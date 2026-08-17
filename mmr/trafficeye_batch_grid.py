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


BATCH_DETECTION_BOX_COORDINATES_KEY = "batch_detection_box_coordinates"
SOURCE_CROP_COORDINATES = "source_crop"


def normalize_batch_detection_box(
    box: BBox,
    content_box: BBox,
    *,
    image_width: int,
    image_height: int,
) -> BBox:
    scale_x = image_width / content_box.width
    scale_y = image_height / content_box.height
    return BBox(
        x1=(box.x1 - content_box.x1) * scale_x,
        y1=(box.y1 - content_box.y1) * scale_y,
        x2=(box.x2 - content_box.x1) * scale_x,
        y2=(box.y2 - content_box.y1) * scale_y,
    )


def normalize_batch_result_for_source_crop(
    result: MMRResult,
    *,
    image_width: int,
    image_height: int,
    content_box: BBox | None = None,
    source_image: Path | None = None,
) -> MMRResult:
    """Normalize a batch result once and record its coordinate space."""
    box = result.detection_box
    updates: dict[str, object] = {}
    if source_image is not None:
        updates["source_image"] = source_image
    if (
        result.raw.get(BATCH_DETECTION_BOX_COORDINATES_KEY) == SOURCE_CROP_COORDINATES
        or box is None
    ):
        return result.model_copy(update=updates) if updates else result

    if content_box is None:
        content_box_payload = result.raw.get("batch_content_box")
        if not isinstance(content_box_payload, dict):
            return result.model_copy(update=updates) if updates else result
        try:
            content_box = BBox.model_validate(content_box_payload)
        except ValueError:
            return result.model_copy(update=updates) if updates else result

    if content_box.width <= 0 or content_box.height <= 0:
        return result.model_copy(update=updates) if updates else result

    updates["detection_box"] = normalize_batch_detection_box(
        box,
        content_box,
        image_width=image_width,
        image_height=image_height,
    )
    raw = dict(result.raw)
    raw[BATCH_DETECTION_BOX_COORDINATES_KEY] = SOURCE_CROP_COORDINATES
    updates["raw"] = raw
    return result.model_copy(update=updates)


def decode_image(image_bytes: bytes, image_path: Path) -> cv2.typing.MatLike:
    """Decode image bytes for TrafficEye requests.

    Raises:
        RuntimeError: If OpenCV cannot decode the supplied bytes as an image.
    """
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not decode crop for MMR request: {image_path}")
    return image


def _resize_for_batch_cell(
    image: cv2.typing.MatLike,
    size: tuple[int, int],
) -> cv2.typing.MatLike:
    height, width = image.shape[:2]
    resized_width, resized_height = size
    if (width, height) == (resized_width, resized_height):
        return image
    return cv2.resize(
        image,
        size,
        interpolation=cv2.INTER_LANCZOS4,
    )


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
        resized = _resize_for_batch_cell(image, (resized_width, resized_height))
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


def _matched_cell_index(result: MMRResult, cells: list[BatchCell]) -> int | None:
    box = result.detection_box
    if box is None:
        return None
    for cell in cells:
        if cell.cell_box.contains_point(box.center, inclusive=False):
            return cell.index

    overlaps = [(box.intersection_area(cell.cell_box), cell.index) for cell in cells]
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
    batch_grids_dir.mkdir(parents=True, exist_ok=True)
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
