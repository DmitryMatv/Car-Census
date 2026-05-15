from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
import orjson

from config import AppConfig
from models import BBox, MMRResult


@dataclass(frozen=True)
class _BatchCell:
    index: int
    source_image: Path
    cell_box: BBox
    content_box: BBox


def _hash_request(image_bytes: bytes, request_payload: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(image_bytes)
    digest.update(orjson.dumps(request_payload))
    return digest.hexdigest()


def _coerce_confidence(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        score = value.get("score") or value.get("confidence")
        if isinstance(score, (int, float)):
            return float(score)
    return None


def _coerce_label(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, dict):
        nested = value.get("value") or value.get("name") or value.get("label")
        if isinstance(nested, str):
            stripped = nested.strip()
            return stripped or None
    return None


def _coerce_box(value: Any) -> BBox | None:
    if not isinstance(value, dict):
        return None
    try:
        box = BBox(
            x1=float(value["topLeftCol"]),
            y1=float(value["topLeftRow"]),
            x2=float(value["bottomRightCol"]),
            y2=float(value["bottomRightRow"]),
        )
    except (KeyError, TypeError, ValueError):
        return None
    if box.width <= 0 or box.height <= 0:
        return None
    return box


def _recognition_payload(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data")
    if isinstance(data, dict) and isinstance(data.get("combinations"), list):
        return data
    return payload


def _road_user_box_position(road_user: dict[str, Any]) -> dict[str, Any] | None:
    box = road_user.get("box")
    if isinstance(box, dict) and isinstance(box.get("position"), dict):
        return box["position"]
    mmr = road_user.get("mmr")
    if not isinstance(mmr, dict):
        return None
    input_data = mmr.get("input")
    if isinstance(input_data, dict) and isinstance(input_data.get("box"), dict):
        return input_data["box"]
    return None


def _iter_road_users(payload: dict[str, Any]) -> list[dict[str, Any]]:
    road_users: list[dict[str, Any]] = []
    combinations = _recognition_payload(payload).get("combinations")
    if not isinstance(combinations, list):
        return road_users
    for combination in combinations:
        if not isinstance(combination, dict):
            continue
        items = combination.get("roadUsers")
        if not isinstance(items, list):
            continue
        road_users.extend(item for item in items if isinstance(item, dict))
    return road_users


def _selected_road_user(payload: dict[str, Any]) -> dict[str, Any] | None:
    candidates = [
        road_user
        for road_user in _iter_road_users(payload)
        if isinstance(road_user.get("mmr"), dict)
    ]
    if not candidates:
        return None

    def rank(road_user: dict[str, Any]) -> tuple[float, float, float, float]:
        position = _road_user_box_position(road_user)
        box = _coerce_box(position)
        mmr = road_user.get("mmr")
        assert isinstance(mmr, dict)
        return (
            box.area if box is not None else 0.0,
            _coerce_confidence(mmr.get("model")) or 0.0,
            _coerce_confidence(mmr.get("make")) or 0.0,
            _coerce_confidence(position) or 0.0,
        )

    return max(candidates, key=rank)


def _ranked_road_user(road_users: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        road_user for road_user in road_users if isinstance(road_user.get("mmr"), dict)
    ]
    if not candidates:
        return None

    def rank(road_user: dict[str, Any]) -> tuple[float, float, float, float]:
        position = _road_user_box_position(road_user)
        box = _coerce_box(position)
        mmr = road_user.get("mmr")
        assert isinstance(mmr, dict)
        return (
            _coerce_confidence(mmr.get("model")) or 0.0,
            _coerce_confidence(mmr.get("make")) or 0.0,
            _coerce_confidence(position) or 0.0,
            box.area if box is not None else 0.0,
        )

    return max(candidates, key=rank)


def _mmr_result_from_road_user(
    road_user: dict[str, Any],
    raw: dict[str, Any],
    source_image: Path | None = None,
) -> MMRResult:
    mmr = road_user["mmr"]
    position = _road_user_box_position(road_user)
    tags = mmr.get("tags")

    return MMRResult(
        make=_coerce_label(mmr.get("make")),
        model=_coerce_label(mmr.get("model")),
        make_confidence=_coerce_confidence(mmr.get("make")),
        model_confidence=_coerce_confidence(mmr.get("model")),
        category=_coerce_label(mmr.get("category")),
        category_confidence=_coerce_confidence(mmr.get("category")),
        generation=_coerce_label(mmr.get("generation")),
        generation_confidence=_coerce_confidence(mmr.get("generation")),
        variation=_coerce_label(mmr.get("variation")),
        variation_confidence=_coerce_confidence(mmr.get("variation")),
        color=_coerce_label(mmr.get("color")),
        color_confidence=_coerce_confidence(mmr.get("color")),
        view=_coerce_label(mmr.get("view")),
        view_confidence=_coerce_confidence(mmr.get("view")),
        view8=_coerce_label(mmr.get("view8")),
        view8_confidence=_coerce_confidence(mmr.get("view8")),
        tags=[tag for tag in tags if isinstance(tag, dict)]
        if isinstance(tags, list)
        else [],
        detection_box=_coerce_box(position),
        detection_confidence=_coerce_confidence(position),
        raw=raw,
        source_image=source_image,
    )


def parse_mmr_results(
    payload: dict[str, Any], source_image: Path | None = None
) -> list[MMRResult]:
    return [
        _mmr_result_from_road_user(road_user, raw=payload, source_image=source_image)
        for road_user in _iter_road_users(payload)
        if isinstance(road_user.get("mmr"), dict)
    ]


def parse_mmr_results_by_combination(
    payload: dict[str, Any], source_image: Path | None = None
) -> list[MMRResult | None]:
    combinations = _recognition_payload(payload).get("combinations")
    if not isinstance(combinations, list):
        return []

    results: list[MMRResult | None] = []
    for combination in combinations:
        if not isinstance(combination, dict):
            results.append(None)
            continue
        road_users = combination.get("roadUsers")
        if not isinstance(road_users, list):
            results.append(None)
            continue
        road_user = _ranked_road_user(
            [item for item in road_users if isinstance(item, dict)]
        )
        results.append(
            _mmr_result_from_road_user(
                road_user, raw=payload, source_image=source_image
            )
            if road_user is not None
            else None
        )
    return results


def parse_mmr_response(
    payload: dict[str, Any], source_image: Path | None = None
) -> MMRResult:
    road_user = _selected_road_user(payload)
    if road_user is None:
        return MMRResult(raw=payload, source_image=source_image)

    return _mmr_result_from_road_user(road_user, raw=payload, source_image=source_image)


def _intersection_area(first: BBox, second: BBox) -> float:
    x1 = max(first.x1, second.x1)
    y1 = max(first.y1, second.y1)
    x2 = min(first.x2, second.x2)
    y2 = min(first.y2, second.y2)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _box_center_inside(box: BBox, region: BBox) -> bool:
    center_x, center_y = box.center
    return region.x1 <= center_x < region.x2 and region.y1 <= center_y < region.y2


def _matched_cell_index(result: MMRResult, cells: list[_BatchCell]) -> int | None:
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
        result.model_confidence or 0.0,
        result.make_confidence or 0.0,
        result.detection_confidence or 0.0,
        result.detection_box.area if result.detection_box is not None else 0.0,
    )


class TrafficEyeClient:
    def __init__(self, config: AppConfig, cache_dir: Path) -> None:
        api_key = os.getenv(config.mmr.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing TrafficEye API key. Set environment variable {config.mmr.api_key_env}."
            )
        self.api_url = config.mmr.api_url
        self.api_key = api_key
        self.timeout = config.mmr.timeout_seconds
        self.cache_dir = cache_dir
        self.accept_model_confidence = config.mmr.accept_model_confidence
        self.tasks = config.mmr.tasks
        self.requested_detection_types = config.mmr.requested_detection_types
        self.mmr_preference = config.mmr.mmr_preference
        self.batch_size = config.mmr.batch_size
        self.batch_grid_columns = config.mmr.batch_grid_columns
        self.batch_cell_size_px = config.mmr.batch_cell_size_px
        self.jpeg_quality = config.analysis.crop_jpeg_quality
        self.batch_grids_dir = self.cache_dir.parent / "batch_grids"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_grids_dir.mkdir(parents=True, exist_ok=True)

    def _request_payload(self) -> dict[str, Any]:
        return {
            "tasks": self.tasks,
            "requestedDetectionTypes": self.requested_detection_types,
            "mmrPreference": self.mmr_preference,
        }

    def _batch_request_payload(self, cells: list[_BatchCell]) -> dict[str, Any]:
        return {
            "tasks": ["MMR"],
            "mmrPreference": self.mmr_preference,
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

    def _decode_image(self, image_bytes: bytes, image_path: Path) -> cv2.typing.MatLike:
        image = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if image is None:
            raise RuntimeError(f"Could not decode crop for MMR request: {image_path}")
        return image

    def _load_or_request(
        self,
        image_bytes: bytes,
        filename: str,
        request_payload: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        cache_key = _hash_request(image_bytes, request_payload)
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            return orjson.loads(cache_path.read_bytes()), cache_key

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                self.api_url,
                headers={"apikey": self.api_key},
                files={
                    "file": (filename, image_bytes, "image/jpeg"),
                    "request": (
                        None,
                        orjson.dumps(request_payload).decode("utf-8"),
                    ),
                },
            )
            response.raise_for_status()
            payload = response.json()
            cache_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
            return payload, cache_key

    def _mark_acceptance(self, result: MMRResult) -> MMRResult:
        confidence = result.model_confidence or 0.0
        result.accepted = confidence >= self.accept_model_confidence and bool(
            result.make or result.model
        )
        return result

    def _build_batch_image(
        self, image_paths: list[Path]
    ) -> tuple[bytes, list[_BatchCell]]:
        columns = min(self.batch_grid_columns, len(image_paths))
        rows = math.ceil(len(image_paths) / columns)
        cell_size = self.batch_cell_size_px
        canvas = np.full(
            (rows * cell_size, columns * cell_size, 3), 255, dtype=np.uint8
        )
        cells: list[_BatchCell] = []

        for index, image_path in enumerate(image_paths):
            image_bytes = image_path.read_bytes()
            image = self._decode_image(image_bytes, image_path)
            height, width = image.shape[:2]
            scale = min(cell_size / width, cell_size / height)
            resized_width = max(1, int(round(width * scale)))
            resized_height = max(1, int(round(height * scale)))
            resized = cv2.resize(
                image,
                (resized_width, resized_height),
                interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
            )
            row, column = divmod(index, columns)
            cell_x1 = column * cell_size
            cell_y1 = row * cell_size
            offset_x = cell_x1 + (cell_size - resized_width) // 2
            offset_y = cell_y1 + (cell_size - resized_height) // 2
            canvas[
                offset_y : offset_y + resized_height,
                offset_x : offset_x + resized_width,
            ] = resized
            cells.append(
                _BatchCell(
                    index=index,
                    source_image=image_path,
                    cell_box=BBox(
                        x1=cell_x1,
                        y1=cell_y1,
                        x2=cell_x1 + cell_size,
                        y2=cell_y1 + cell_size,
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
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            raise RuntimeError("Could not encode MMR batch image")
        return encoded.tobytes(), cells

    def _match_batch_results(
        self,
        payload: dict[str, Any],
        cells: list[_BatchCell],
        batch_image_path: Path,
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
                            "batch_content_box": cell.content_box.model_dump(
                                mode="json"
                            ),
                            "response": result.raw,
                        },
                    }
                )
            results.append(self._mark_acceptance(result))
        return results

    def _write_batch_debug_artifacts(
        self,
        cache_key: str,
        image_bytes: bytes,
        cells: list[_BatchCell],
        request_payload: dict[str, Any],
    ) -> Path:
        batch_image_path = self.batch_grids_dir / f"{cache_key}.jpg"
        if not batch_image_path.exists():
            batch_image_path.write_bytes(image_bytes)
        manifest_path = self.batch_grids_dir / f"{cache_key}.json"
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
            manifest_path.write_bytes(
                orjson.dumps(manifest, option=orjson.OPT_INDENT_2)
            )
        return batch_image_path

    def recognize_vehicle_crop(self, image_path: Path) -> MMRResult:
        image_bytes = image_path.read_bytes()
        self._decode_image(image_bytes, image_path)
        request_payload = self._request_payload()
        payload, _cache_key = self._load_or_request(
            image_bytes=image_bytes,
            filename=image_path.name,
            request_payload=request_payload,
        )
        result = parse_mmr_response(payload, source_image=image_path)
        return self._mark_acceptance(result)

    def recognize_vehicle_crops(self, image_paths: list[Path]) -> list[MMRResult]:
        if not image_paths:
            return []
        if self.batch_size <= 1:
            return [
                self.recognize_vehicle_crop(image_path) for image_path in image_paths
            ]

        image_bytes, cells = self._build_batch_image(image_paths)
        request_payload = self._batch_request_payload(cells)
        cache_key = _hash_request(image_bytes, request_payload)
        batch_image_path = self._write_batch_debug_artifacts(
            cache_key=cache_key,
            image_bytes=image_bytes,
            cells=cells,
            request_payload=request_payload,
        )
        payload, _cache_key = self._load_or_request(
            image_bytes=image_bytes,
            filename="mmr_batch.jpg",
            request_payload=request_payload,
        )
        return self._match_batch_results(payload, cells, batch_image_path)
