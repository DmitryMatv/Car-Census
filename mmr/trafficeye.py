from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx

from config import AppConfig
from mmr.trafficeye_batch_grid import (
    batch_request_payload,
    build_batch_image,
    decode_image,
    match_batch_results,
    write_batch_debug_artifacts,
)
from mmr.trafficeye_cache import TrafficEyeCacheClient, hash_request
from mmr.trafficeye_parser import (
    parse_mmr_response,
    parse_mmr_results,
    parse_mmr_results_by_combination,
)
from models import MMRResult


class TrafficEyeClient:
    def __init__(
        self, config: AppConfig, cache_dir: Path, require_api_key: bool = True
    ) -> None:
        api_key = os.getenv(config.mmr.api_key_env)
        if require_api_key and not api_key:
            raise RuntimeError(
                f"Missing TrafficEye API key. Set environment variable {config.mmr.api_key_env}."
            )
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
        self._cache_client = TrafficEyeCacheClient(
            api_url=config.mmr.api_url,
            api_key=api_key or "",
            timeout=config.mmr.timeout_seconds,
            cache_dir=self.cache_dir,
            http_client_factory=httpx.Client,
        )

    def _request_payload(self) -> dict[str, Any]:
        return {
            "tasks": self.tasks,
            "requestedDetectionTypes": self.requested_detection_types,
            "mmrPreference": self.mmr_preference,
        }

    def _mark_acceptance(self, result: MMRResult) -> MMRResult:
        make_confidence = result.make_confidence or 0.0
        model_confidence = result.model_confidence or 0.0
        result.accepted = (
            make_confidence >= self.accept_model_confidence
            and model_confidence >= self.accept_model_confidence
            and bool(result.make)
            and bool(result.model)
        )
        return result

    def write_vehicle_crop_grid(self, image_paths: list[Path]) -> Path | None:
        if not image_paths:
            return None
        image_bytes, cells = build_batch_image(
            image_paths,
            columns=self.batch_grid_columns,
            cell_size_px=self.batch_cell_size_px,
            jpeg_quality=self.jpeg_quality,
        )
        request_payload = batch_request_payload(
            cells, mmr_preference=self.mmr_preference
        )
        cache_key = hash_request(image_bytes, request_payload)
        return write_batch_debug_artifacts(
            batch_grids_dir=self.batch_grids_dir,
            cache_key=cache_key,
            image_bytes=image_bytes,
            cells=cells,
            request_payload=request_payload,
        )

    def recognize_vehicle_crop(self, image_path: Path) -> MMRResult:
        image_bytes = image_path.read_bytes()
        decode_image(image_bytes, image_path)
        request_payload = self._request_payload()
        payload, _cache_key = self._cache_client.load_or_request(
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

        image_bytes, cells = build_batch_image(
            image_paths,
            columns=self.batch_grid_columns,
            cell_size_px=self.batch_cell_size_px,
            jpeg_quality=self.jpeg_quality,
        )
        request_payload = batch_request_payload(
            cells, mmr_preference=self.mmr_preference
        )
        cache_key = hash_request(image_bytes, request_payload)
        batch_image_path = write_batch_debug_artifacts(
            batch_grids_dir=self.batch_grids_dir,
            cache_key=cache_key,
            image_bytes=image_bytes,
            cells=cells,
            request_payload=request_payload,
        )
        payload, _cache_key = self._cache_client.load_or_request(
            image_bytes=image_bytes,
            filename="mmr_batch.jpg",
            request_payload=request_payload,
        )
        return match_batch_results(
            payload, cells, batch_image_path, self._mark_acceptance
        )
