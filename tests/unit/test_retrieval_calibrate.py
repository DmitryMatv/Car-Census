from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from mmr.retrieval_cache import MMRRetrievalStore
from mmr.retrieval_calibrate import calibrate_retrieval_cache
from mmr.retrieval_calibration_artifact import (
    RetrievalCalibrationArtifact,
    load_calibration_artifact,
    save_calibration_artifact,
)
from mmr.trafficeye_cache import hash_request
from models import MMRResult


class _Provider:
    model = "google/gemini-embedding-2"
    dimensions = 768

    def __init__(self) -> None:
        self.vectors: dict[bytes, list[float]] = {}

    def embed(self, image_bytes: bytes) -> list[float]:
        return self.vectors[image_bytes]


def _image_bytes(value: int) -> bytes:
    image = np.full((40, 60, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def _vector(x: float, y: float) -> list[float]:
    return [x, y] + [0.0] * 766


def test_calibration_reports_threshold_only_with_separated_evidence(
    config_factory, tmp_path: Path
) -> None:
    provider = _Provider()
    store = MMRRetrievalStore(
        tmp_path / "cache" / "retrieval",
        retrieval_mode="shadow",
        embedding_model="google/gemini-embedding-2",
        embedding_dimensions=768,
        embedding_distance_threshold=0.2,
        phash_max_hamming_distance=64,
        min_neighbors=1,
        embedding_provider=provider,
    )
    request_payload = {"tasks": ["MMR"], "mmrPreference": "BOX"}
    same_vectors = [_vector(1.0, 0.0), _vector(0.99, 0.01), _vector(0.98, 0.02)]
    conflicting_vectors = [_vector(0.0, 1.0)] * 3
    for index, vector in enumerate(same_vectors + conflicting_vectors):
        image_bytes = _image_bytes(20 + index * 30)
        provider.vectors[image_bytes] = vector
        store.record_api_result(
            image_bytes=image_bytes,
            request_hash=hash_request(image_bytes, request_payload),
            request_payload=request_payload,
            result=MMRResult(
                make="Toyota" if index < 3 else "Honda",
                model="Corolla" if index < 3 else "Civic",
                generation="E210" if index < 3 else "FE1",
                make_confidence=0.95,
                accepted=True,
            ),
        )

    config = config_factory(
        {
            "mmr": {
                "retrieval_calibration_min_same_identity": 3,
                "retrieval_calibration_min_conflicting_identity": 3,
            }
        }
    )
    report = calibrate_retrieval_cache(config=config, cache_dir=tmp_path / "cache")

    assert report.same_identity_pairs == 6
    assert report.conflicting_identity_pairs == 9
    assert report.usable_threshold is not None
    assert report.usable_threshold < (report.minimum_conflicting_identity_distance or 0)
    artifact = load_calibration_artifact(
        tmp_path / "cache" / "retrieval",
        embedding_model="google/gemini-embedding-2",
        embedding_dimensions=768,
        phash_max_hamming_distance=4,
    )
    assert artifact is not None
    assert artifact.threshold == report.usable_threshold


def test_failed_calibration_keeps_existing_artifact(
    config_factory, tmp_path: Path
) -> None:
    store_root = tmp_path / "cache" / "retrieval"
    save_calibration_artifact(
        store_root,
        RetrievalCalibrationArtifact(
            schema_version=1,
            created_at="2026-01-01T00:00:00+00:00",
            embedding_model="google/gemini-embedding-2",
            embedding_dimensions=768,
            phash_max_hamming_distance=4,
            threshold=0.05,
            same_identity_pairs=3,
            conflicting_identity_pairs=3,
            maximum_same_identity_distance=0.05,
            minimum_conflicting_identity_distance=0.2,
        ),
    )
    provider = _Provider()
    store = MMRRetrievalStore(
        store_root,
        retrieval_mode="shadow",
        embedding_model="google/gemini-embedding-2",
        embedding_dimensions=768,
        embedding_distance_threshold=0.2,
        phash_max_hamming_distance=64,
        min_neighbors=1,
        embedding_provider=provider,
    )
    request_payload = {"tasks": ["MMR"], "mmrPreference": "BOX"}
    image_bytes = _image_bytes(20)
    provider.vectors[image_bytes] = _vector(1.0, 0.0)
    store.record_api_result(
        image_bytes=image_bytes,
        request_hash=hash_request(image_bytes, request_payload),
        request_payload=request_payload,
        result=MMRResult(
            make="Toyota",
            model="Corolla",
            generation="E210",
            make_confidence=0.95,
            accepted=True,
        ),
    )

    config = config_factory(None)
    report = calibrate_retrieval_cache(config=config, cache_dir=tmp_path / "cache")

    assert report.usable_threshold is None
    artifact = load_calibration_artifact(
        store_root,
        embedding_model="google/gemini-embedding-2",
        embedding_dimensions=768,
        phash_max_hamming_distance=4,
    )
    assert artifact is not None
    assert artifact.threshold == 0.05
