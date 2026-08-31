from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx

from config import AppConfig
from mmr.embeddings import ImageEmbeddingProvider, build_embedding_provider
from mmr.retrieval_cache import (
    MMRRetrievalStore,
    RetrievalLookup,
    image_sha256,
)
from mmr.trafficeye_batch_grid import (
    BatchCell,
    batch_request_payload,
    build_batch_image,
    decode_image,
    match_batch_results,
    normalize_batch_result_for_source_crop,
    write_batch_debug_artifacts,
)
from mmr.trafficeye_cache import TrafficEyeCacheClient, hash_request
from mmr.trafficeye_parser import (
    parse_mmr_response,
    parse_mmr_results,
    parse_mmr_results_by_combination,
)
from models import BBox, MMRResult


def build_single_request_payload(
    width: int,
    height: int,
    mmr_preference: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "tasks": ["MMR"],
        "mmrPreference": mmr_preference,
    }
    if width > 0 and height > 0:
        payload["combinations"] = [
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
        ]
    return payload


class TrafficEyeClient:
    def __init__(
        self,
        config: AppConfig,
        cache_dir: Path,
        require_api_key: bool = True,
        embedding_provider: ImageEmbeddingProvider | None = None,
    ) -> None:
        api_key = os.getenv(config.mmr.api_key_env)
        self.cache_dir = cache_dir
        self.responses_dir = cache_dir / "responses"
        self._retrieval_mode = config.mmr.retrieval_mode
        self.accept_model_confidence = config.mmr.accept_model_confidence
        self.mmr_preference = config.mmr.mmr_preference
        self.batch_size = config.mmr.batch_size
        self.batch_grid_columns = config.mmr.batch_grid_columns
        self.batch_cell_size_px = config.mmr.batch_cell_size_px
        self.jpeg_quality = config.analysis.crop_jpeg_quality
        self.batch_grids_dir = self.cache_dir / "batch_grids"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_client = TrafficEyeCacheClient(
            api_url=config.mmr.api_url,
            api_key=api_key or "",
            timeout=config.mmr.timeout_seconds,
            cache_dir=self.responses_dir,
            legacy_cache_dir=self.cache_dir,
            require_api_key=require_api_key,
            http_client_factory=httpx.Client,
        )
        self._embedding_provider = embedding_provider or build_embedding_provider(
            config, self.cache_dir
        )
        self._retrieval_store = MMRRetrievalStore.from_config(
            config,
            self.cache_dir,
            embedding_provider=self._embedding_provider,
        )

    def _request_payload(self, width: int = 0, height: int = 0) -> dict[str, Any]:
        return build_single_request_payload(
            width=width,
            height=height,
            mmr_preference=self.mmr_preference,
        )

    def _mark_acceptance(self, result: MMRResult) -> MMRResult:
        make_confidence = result.make_confidence or 0.0
        result.accepted = (
            make_confidence >= self.accept_model_confidence
            and bool(result.make)
            and bool(result.model)
        )
        return result

    def _mark_api_provenance(self, result: MMRResult) -> MMRResult:
        return result.model_copy(
            update={
                "evidence_source": result.evidence_source or "api_confirmed",
                "resolution_method": result.resolution_method or "external_api",
            }
        )

    def _mark_retrieval_result(
        self,
        result: MMRResult,
        image_path: Path,
        lookup: RetrievalLookup,
    ) -> MMRResult:
        match = lookup.match
        assert match is not None
        return result.model_copy(
            update={
                "source_image": image_path,
                "resolution_method": (
                    "exact_retrieval"
                    if match.kind == "exact"
                    else "embedding_retrieval"
                ),
                "retrieval_record_id": match.record.record_id,
                "retrieval_distance": match.distance,
                "retrieval_neighbor_count": match.neighbor_count,
            }
        )

    def _crop_request(self, image_path: Path) -> tuple[bytes, dict[str, Any], str]:
        image_bytes = image_path.read_bytes()
        image = decode_image(image_bytes, image_path)
        height, width = image.shape[:2]
        request_payload = self._request_payload(width=width, height=height)
        return image_bytes, request_payload, hash_request(image_bytes, request_payload)

    def _normalize_batch_result_for_crop(
        self,
        result: MMRResult,
        cell: BatchCell,
        image_path: Path,
    ) -> MMRResult:
        image = decode_image(image_path.read_bytes(), image_path)
        source_height, source_width = image.shape[:2]
        return normalize_batch_result_for_source_crop(
            result,
            content_box=cell.content_box,
            image_width=source_width,
            image_height=source_height,
            source_image=image_path,
        )

    def _lookup_retrieval(
        self,
        image_bytes: bytes,
        image_path: Path,
        request_hash: str,
        request_payload: dict[str, Any],
    ) -> tuple[MMRResult | None, RetrievalLookup]:
        lookup = self._retrieval_store.lookup(
            image_bytes=image_bytes,
            request_hash=request_hash,
            request_payload=request_payload,
            include_ineligible_match=self._retrieval_mode == "enforce",
        )
        match = lookup.match
        if match is None:
            self._retrieval_store.append_lookup_audit(
                image_sha256_value=image_sha256(image_bytes),
                request_hash=request_hash,
                lookup=lookup,
                action="api_fallback",
            )
            return None, lookup
        if match.kind == "embedding" and self._retrieval_mode != "enforce":
            self._retrieval_store.append_lookup_audit(
                image_sha256_value=image_sha256(image_bytes),
                request_hash=request_hash,
                lookup=lookup,
                action="shadow",
            )
            return None, lookup
        self._retrieval_store.append_lookup_audit(
            image_sha256_value=image_sha256(image_bytes),
            request_hash=request_hash,
            lookup=lookup,
            action="reuse",
        )
        return self._mark_retrieval_result(match.result, image_path, lookup), lookup

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
        image_bytes, request_payload, request_hash = self._crop_request(image_path)
        lookup: RetrievalLookup | None = None
        if self._retrieval_mode != "disabled":
            retrieval_result, lookup = self._lookup_retrieval(
                image_bytes=image_bytes,
                image_path=image_path,
                request_hash=request_hash,
                request_payload=request_payload,
            )
            if retrieval_result is not None:
                return retrieval_result
        force_api_request = lookup is not None and lookup.reason in {
            "exact_conflict",
            "exact_ineligible",
            "embedding_conflict",
            "embedding_ineligible",
        }
        payload, _cache_key = self._cache_client.load_or_request(
            image_bytes=image_bytes,
            filename=image_path.name,
            request_payload=request_payload,
            allow_cache=not force_api_request,
        )
        result = parse_mmr_response(payload, source_image=image_path)
        result = self._mark_api_provenance(self._mark_acceptance(result))
        self._retrieval_store.record_api_result(
            image_bytes=image_bytes,
            request_hash=request_hash,
            request_payload=request_payload,
            result=result,
        )
        return result

    def recognize_vehicle_crops(self, image_paths: list[Path]) -> list[MMRResult]:
        if not image_paths:
            return []
        if self.batch_size <= 1:
            return [
                self.recognize_vehicle_crop(image_path) for image_path in image_paths
            ]

        resolved: list[MMRResult | None] = [None] * len(image_paths)
        unresolved_indices: list[int] = []
        unresolved_paths: list[Path] = []
        crop_requests: dict[int, tuple[bytes, dict[str, Any], str]] = {}
        force_api_request = False
        for index, image_path in enumerate(image_paths):
            crop_request = self._crop_request(image_path)
            crop_requests[index] = crop_request
            if self._retrieval_mode == "disabled":
                unresolved_indices.append(index)
                unresolved_paths.append(image_path)
                continue
            image_bytes, request_payload, request_hash = crop_request
            retrieval_result, lookup = self._lookup_retrieval(
                image_bytes=image_bytes,
                image_path=image_path,
                request_hash=request_hash,
                request_payload=request_payload,
            )
            if retrieval_result is None:
                unresolved_indices.append(index)
                unresolved_paths.append(image_path)
                force_api_request = force_api_request or lookup.reason in {
                    "exact_conflict",
                    "exact_ineligible",
                    "embedding_conflict",
                    "embedding_ineligible",
                }
            else:
                resolved[index] = retrieval_result

        if not unresolved_paths:
            return [result for result in resolved if result is not None]

        image_bytes, cells = build_batch_image(
            unresolved_paths,
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
            allow_cache=not force_api_request,
        )
        results = match_batch_results(
            payload, cells, batch_image_path, self._mark_acceptance
        )
        finalized: list[MMRResult] = []
        for index, image_path, cell, result in zip(
            unresolved_indices,
            unresolved_paths,
            cells,
            results,
            strict=True,
        ):
            image_bytes, single_payload, single_request_hash = crop_requests[index]
            result = self._normalize_batch_result_for_crop(result, cell, image_path)
            result = self._mark_api_provenance(result)
            finalized.append(result)
            self._retrieval_store.record_api_result(
                image_bytes=image_bytes,
                request_hash=single_request_hash,
                request_payload=single_payload,
                result=result,
            )
        for index, result in zip(unresolved_indices, finalized, strict=True):
            resolved[index] = result
        return [result for result in resolved if result is not None]
