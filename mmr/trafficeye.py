from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
import orjson

from config import AppConfig
from models import BBox, MMRResult


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


def parse_mmr_response(
    payload: dict[str, Any], source_image: Path | None = None
) -> MMRResult:
    road_user = _selected_road_user(payload)
    if road_user is None:
        return MMRResult(raw=payload, source_image=source_image)

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
        raw=payload,
        source_image=source_image,
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
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def recognize_vehicle_crop(self, image_path: Path) -> MMRResult:
        image_bytes = image_path.read_bytes()
        image = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if image is None:
            raise RuntimeError(f"Could not decode crop for MMR request: {image_path}")
        request_payload = {
            "tasks": self.tasks,
            "requestedDetectionTypes": self.requested_detection_types,
            "mmrPreference": self.mmr_preference,
        }
        cache_key = _hash_request(image_bytes, request_payload)
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            payload = orjson.loads(cache_path.read_bytes())
            result = parse_mmr_response(payload, source_image=image_path)
        else:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    self.api_url,
                    headers={"apikey": self.api_key},
                    files={
                        "file": (image_path.name, image_bytes, "image/jpeg"),
                        "request": (
                            None,
                            orjson.dumps(request_payload).decode("utf-8"),
                        ),
                    },
                )
                response.raise_for_status()
                payload = response.json()
                cache_path.write_bytes(
                    orjson.dumps(payload, option=orjson.OPT_INDENT_2)
                )
                result = parse_mmr_response(payload, source_image=image_path)

        confidence = result.model_confidence or 0.0
        result.accepted = confidence >= self.accept_model_confidence and bool(
            result.make or result.model
        )
        return result
