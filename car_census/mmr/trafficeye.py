from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
import orjson

from car_census.config import AppConfig
from car_census.types import MMRResult


def _hash_request(image_bytes: bytes, request_payload: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(image_bytes)
    digest.update(orjson.dumps(request_payload))
    return digest.hexdigest()


def _find_nested_first(data: Any, keys: tuple[str, ...]) -> tuple[Any, str | None]:
    if isinstance(data, dict):
        for key, value in data.items():
            key_lower = key.lower()
            if any(candidate in key_lower for candidate in keys):
                return value, key
            found, found_key = _find_nested_first(value, keys)
            if found is not None:
                return found, found_key
    elif isinstance(data, list):
        for item in data:
            found, found_key = _find_nested_first(item, keys)
            if found is not None:
                return found, found_key
    return None, None


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


def parse_mmr_response(payload: dict[str, Any], source_image: Path | None = None) -> MMRResult:
    make_value, _ = _find_nested_first(payload, ("make", "manufacturer", "brand"))
    model_value, _ = _find_nested_first(payload, ("model",))
    make_conf_value, _ = _find_nested_first(payload, ("makescore", "makeconfidence"))
    model_conf_value, _ = _find_nested_first(payload, ("modelscore", "modelconfidence"))

    if isinstance(make_value, dict) and make_conf_value is None:
        make_conf_value = make_value
    if isinstance(model_value, dict) and model_conf_value is None:
        model_conf_value = model_value

    return MMRResult(
        make=_coerce_label(make_value),
        model=_coerce_label(model_value),
        make_confidence=_coerce_confidence(make_conf_value),
        model_confidence=_coerce_confidence(model_conf_value),
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
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def recognize_vehicle_crop(self, image_path: Path) -> MMRResult:
        image_bytes = image_path.read_bytes()
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not decode crop for MMR request: {image_path}")
        height, width = image.shape[:2]
        request_payload = {
            "tasks": ["MMR"],
            "combinations": [
                {
                    "roadUsers": [
                        {
                            "box": {
                                "position": {
                                    "topLeftCol": 0,
                                    "topLeftRow": 0,
                                    "bottomRightCol": width,
                                    "bottomRightRow": height,
                                }
                            }
                        }
                    ]
                }
            ],
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
                        "request": (None, orjson.dumps(request_payload).decode("utf-8")),
                    },
                )
                response.raise_for_status()
                payload = response.json()
                cache_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
                result = parse_mmr_response(payload, source_image=image_path)

        confidence = result.model_confidence or 0.0
        result.accepted = confidence >= self.accept_model_confidence and bool(result.make or result.model)
        return result
