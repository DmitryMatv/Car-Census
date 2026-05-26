from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import orjson

from mmr.trafficeye_parser import parse_mmr_results, parse_mmr_results_by_combination
from models import BBox, MMRResult


@dataclass(frozen=True)
class BatchCell:
    index: int
    source_image: Path
    cell_box: BBox
    content_box: BBox


def decode_image(image_bytes: bytes, image_path: Path) -> cv2.typing.MatLike:
    """Decode image bytes for TrafficEye requests.

    Raises:
        RuntimeError: If OpenCV cannot decode the supplied bytes as an image.
    """
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not decode crop for MMR request: {image_path}")
    return image


def build_batch_image(
    image_paths: list[Path],
    *,
    columns: int,
    cell_size_px: int,
    jpeg_quality: int,
) -> tuple[bytes, list[BatchCell]]:
    if not image_paths:
        raise ValueError("image_paths cannot be empty")
    if columns <= 0:
        raise ValueError(f"columns must be positive, got {columns}")
    if cell_size_px <= 0:
        raise ValueError(f"cell_size_px must be positive, got {cell_size_px}")
    grid_columns = min(columns, len(image_paths))
    rows = math.ceil(len(image_paths) / grid_columns)
    canvas = np.full(
        (rows * cell_size_px, grid_columns * cell_size_px, 3), 255, dtype=np.uint8
    )
    cells: list[BatchCell] = []

    for index, image_path in enumerate(image_paths):
        image_bytes = image_path.read_bytes()
        image = decode_image(image_bytes, image_path)
        height, width = image.shape[:2]
        scale = min(cell_size_px / width, cell_size_px / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
        )
        row, column = divmod(index, grid_columns)
        cell_x1 = column * cell_size_px
        cell_y1 = row * cell_size_px
        offset_x = cell_x1 + (cell_size_px - resized_width) // 2
        offset_y = cell_y1 + (cell_size_px - resized_height) // 2
        canvas[
            offset_y : offset_y + resized_height,
            offset_x : offset_x + resized_width,
        ] = resized
        cells.append(
            BatchCell(
                index=index,
                source_image=image_path,
                cell_box=BBox(
                    x1=cell_x1,
                    y1=cell_y1,
                    x2=cell_x1 + cell_size_px,
                    y2=cell_y1 + cell_size_px,
                ),
                content_box=BBox(
                    x1=offset_x,
                    y1=offset_y,
                    x2=offset_x + resized_width,
                    y2=offset_y + resized_height,
                ),
            )
        )

    ok, encoded = cv2.imencode(
        ".jpg",
        canvas,
        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
    )
    if not ok:
        raise RuntimeError("Could not encode MMR batch image")
    return encoded.tobytes(), cells


def batch_request_payload(
    cells: list[BatchCell], *, mmr_preference: str
) -> dict[str, Any]:
    return {
        "tasks": ["MMR"],
        "mmrPreference": mmr_preference,
        "combinations": [
            {
                "roadUsers": [
                    {
                        "box": {
                            "position": {
                                "topLeftCol": cell.content_box.x1,
                                "topLeftRow": cell.content_box.y1,
                                "bottomRightCol": cell.content_box.x2,
                                "bottomRightRow": cell.content_box.y2,
                            }
                        }
                    }
                ]
            }
            for cell in cells
        ],
    }


def _intersection_area(first: BBox, second: BBox) -> float:
    x1 = max(first.x1, second.x1)
    y1 = max(first.y1, second.y1)
    x2 = min(first.x2, second.x2)
    y2 = min(first.y2, second.y2)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _box_center_inside(box: BBox, region: BBox) -> bool:
    center_x, center_y = box.center
    return region.x1 <= center_x < region.x2 and region.y1 <= center_y < region.y2


def _matched_cell_index(result: MMRResult, cells: list[BatchCell]) -> int | None:
    box = result.detection_box
    if box is None:
        return None
    for cell in cells:
        if _box_center_inside(box, cell.cell_box):
            return cell.index

    overlaps = [(_intersection_area(box, cell.cell_box), cell.index) for cell in cells]
    overlap, index = max(overlaps, default=(0.0, None))
    if overlap <= 0.0:
        return None
    return index


def _result_rank(result: MMRResult) -> tuple[float, float, float, float]:
    return (
        result.detection_box.area if result.detection_box is not None else 0.0,
        result.detection_confidence or 0.0,
        result.model_confidence or 0.0,
        result.make_confidence or 0.0,
    )


def match_batch_results(
    payload: dict[str, Any],
    cells: list[BatchCell],
    batch_image_path: Path,
    accept_result: Callable[[MMRResult], MMRResult],
) -> list[MMRResult]:
    matched_by_cell: dict[int, MMRResult] = {}
    ordered_matched_result_ids: set[int] = set()
    ordered_results = parse_mmr_results_by_combination(
        payload, source_image=batch_image_path
    )
    if len(ordered_results) == len(cells):
        for cell, result in zip(cells, ordered_results, strict=True):
            if result is not None:
                matched_by_cell[cell.index] = result
                ordered_matched_result_ids.add(id(result))

    parsed_results = [
        result for result in ordered_results if result is not None
    ] or parse_mmr_results(payload, source_image=batch_image_path)
    for result in sorted(parsed_results, key=_result_rank, reverse=True):
        if id(result) in ordered_matched_result_ids:
            continue
        cell_index = _matched_cell_index(result, cells)
        if cell_index is None or cell_index in matched_by_cell:
            continue
        matched_by_cell[cell_index] = result

    results: list[MMRResult] = []
    for cell in cells:
        result = matched_by_cell.get(cell.index)
        if result is None:
            result = MMRResult(
                raw={
                    "skipped_reason": "batch_no_mmr_result",
                    "batch_image": str(batch_image_path),
                    "batch_cell_index": cell.index,
                    "batch_cell_box": cell.cell_box.model_dump(mode="json"),
                    "batch_content_box": cell.content_box.model_dump(mode="json"),
                },
                source_image=cell.source_image,
            )
        else:
            result = result.model_copy(
                update={
                    "source_image": cell.source_image,
                    "raw": {
                        "batch_image": str(batch_image_path),
                        "batch_cell_index": cell.index,
                        "batch_cell_box": cell.cell_box.model_dump(mode="json"),
                        "batch_content_box": cell.content_box.model_dump(mode="json"),
                        "response": result.raw,
                    },
                }
            )
        results.append(accept_result(result))
    return results


def write_batch_debug_artifacts(
    *,
    batch_grids_dir: Path,
    cache_key: str,
    image_bytes: bytes,
    cells: list[BatchCell],
    request_payload: dict[str, Any],
) -> Path:
    batch_image_path = batch_grids_dir / f"{cache_key}.jpg"
    if not batch_image_path.exists():
        batch_image_path.write_bytes(image_bytes)
    manifest_path = batch_grids_dir / f"{cache_key}.json"
    if not manifest_path.exists():
        manifest = {
            "image": batch_image_path.name,
            "request": request_payload,
            "cells": [
                {
                    "index": cell.index,
                    "source_image": str(cell.source_image),
                    "cell_box": cell.cell_box.model_dump(mode="json"),
                    "content_box": cell.content_box.model_dump(mode="json"),
                }
                for cell in cells
            ],
        }
        manifest_path.write_bytes(orjson.dumps(manifest, option=orjson.OPT_INDENT_2))
    return batch_image_path
