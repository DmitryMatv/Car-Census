from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import orjson

from mmr.retrieval_cache import MMRRetrievalStore
from mmr.retrieval_calibration_artifact import (
    RetrievalCalibrationArtifact,
    save_calibration_artifact,
)
from mmr.trafficeye_cache import hash_request
from models import BBox, MMRResult


def _image_bytes(quality: int = 95) -> bytes:
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[:, :] = [80, 140, 200]
    image[20:60, 30:90] = [200, 100, 40]
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return encoded.tobytes()


class _EmbeddingProvider:
    model = "google/gemini-embedding-2"
    dimensions = 768

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, image_bytes: bytes) -> list[float]:
        _ = image_bytes
        self.calls += 1
        return [1.0] + [0.0] * (self.dimensions - 1)


def _store(
    tmp_path: Path,
    *,
    retrieval_mode: Literal["disabled", "shadow", "enforce"] = "shadow",
    min_neighbors: int = 1,
    min_make_confidence: float = 0.0,
    embedding_provider: _EmbeddingProvider | None = None,
) -> MMRRetrievalStore:
    return MMRRetrievalStore(
        tmp_path / "retrieval",
        retrieval_mode=retrieval_mode,
        embedding_model="google/gemini-embedding-2",
        embedding_dimensions=768,
        embedding_distance_threshold=0.02,
        phash_max_hamming_distance=4,
        min_neighbors=min_neighbors,
        min_make_confidence=min_make_confidence,
        embedding_provider=embedding_provider or _EmbeddingProvider(),
    )


def _request_payload() -> dict[str, object]:
    return {"tasks": ["MMR"], "mmrPreference": "BOX"}


def _result(**updates: object) -> MMRResult:
    values: dict[str, object] = {
        "make": "Toyota",
        "model": "Corolla",
        "make_confidence": 0.95,
        "model_confidence": 0.90,
        "generation": "E210",
        "variation": "Hybrid",
        "color": "BLUE",
        "view": "frontal",
        "tags": [{"name": "taxi", "value": "no"}],
        "detection_box": BBox(x1=1, y1=2, x2=70, y2=60),
        "detection_confidence": 0.99,
        "accepted": True,
    }
    values.update(updates)
    return MMRResult.model_validate(values)


def test_exact_lookup_round_trips_full_api_result(tmp_path: Path) -> None:
    store = _store(tmp_path)
    image_bytes = _image_bytes()
    payload = _request_payload()
    request_hash = hash_request(image_bytes, payload)

    record = store.record_api_result(
        image_bytes=image_bytes,
        request_hash=request_hash,
        request_payload=payload,
        result=_result(raw={"response": {"data": {"combinations": ["many cars"]}}}),
    )

    lookup = store.lookup(
        image_bytes=image_bytes,
        request_hash=request_hash,
        request_payload=payload,
    )

    assert lookup.reason == "exact_match"
    assert lookup.match is not None
    assert lookup.match.record.record_id == record.record_id
    assert lookup.match.kind == "exact"
    assert lookup.match.record.request_contract == payload
    assert lookup.match.result.color == "BLUE"
    assert lookup.match.result.detection_box is not None
    assert lookup.match.result.raw == {
        "response": {"data": {"combinations": ["many cars"]}}
    }


def test_embedding_lookup_reuses_identity_but_not_observation_fields(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    stored_bytes = _image_bytes(quality=95)
    query_bytes = _image_bytes(quality=90)
    payload = _request_payload()

    store.record_api_result(
        image_bytes=stored_bytes,
        request_hash=hash_request(stored_bytes, payload),
        request_payload=payload,
        result=_result(vehicle_index=2, api_classification_index=3),
    )

    lookup = store.lookup(
        image_bytes=query_bytes,
        request_hash=hash_request(query_bytes, payload),
        request_payload=payload,
    )

    assert lookup.reason == "embedding_match"
    assert lookup.match is not None
    assert lookup.match.kind == "embedding"
    assert lookup.match.result.make == "Toyota"
    assert lookup.match.result.model == "Corolla"
    assert lookup.match.result.generation == "E210"
    assert lookup.match.result.variation == "Hybrid"
    assert lookup.match.result.color is None
    assert lookup.match.result.view is None
    assert lookup.match.result.tags == []
    assert lookup.match.result.detection_box is None
    assert lookup.match.result.vehicle_index is None
    assert lookup.match.result.api_classification_index is None


def test_exact_lookup_rejects_conflicting_api_evidence(tmp_path: Path) -> None:
    store = _store(tmp_path)
    image_bytes = _image_bytes()
    payload = _request_payload()
    request_hash = hash_request(image_bytes, payload)

    store.record_api_result(
        image_bytes=image_bytes,
        request_hash=request_hash,
        request_payload=payload,
        result=_result(),
    )
    store.record_api_result(
        image_bytes=image_bytes,
        request_hash=request_hash,
        request_payload=payload,
        result=_result(make="Honda", model="Civic"),
    )

    lookup = store.lookup(
        image_bytes=image_bytes,
        request_hash=request_hash,
        request_payload=payload,
    )

    assert lookup.match is None
    assert lookup.reason == "exact_conflict"
    assert lookup.candidate_count == 2


def test_adjudication_supersedes_api_evidence_without_mutating_it(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    image_bytes = _image_bytes()
    payload = _request_payload()
    request_hash = hash_request(image_bytes, payload)
    original = store.record_api_result(
        image_bytes=image_bytes,
        request_hash=request_hash,
        request_payload=payload,
        result=_result(),
    )

    corrected = store.record_adjudication(
        record_id=original.record_id,
        result=_result(make="Honda", model="Civic"),
    )
    lookup = store.lookup(
        image_bytes=image_bytes,
        request_hash=request_hash,
        request_payload=payload,
    )

    assert corrected.supersedes_record_id == original.record_id
    assert lookup.match is not None
    assert lookup.match.result.make == "Honda"
    assert lookup.match.result.evidence_source == "human_adjudicated"
    assert lookup.match.result.resolution_method == "human_adjudication"
    assert original.result.make == "Toyota"


def test_embedding_lookup_requires_configured_neighbor_count(tmp_path: Path) -> None:
    store = _store(tmp_path, min_neighbors=2)
    image_bytes = _image_bytes(quality=95)
    query_bytes = _image_bytes(quality=90)
    payload = _request_payload()

    store.record_api_result(
        image_bytes=image_bytes,
        request_hash=hash_request(image_bytes, payload),
        request_payload=payload,
        result=_result(),
    )

    lookup = store.lookup(
        image_bytes=query_bytes,
        request_hash=hash_request(query_bytes, payload),
        request_payload=payload,
    )

    assert lookup.match is None
    assert lookup.reason == "insufficient_neighbors"


def test_lookup_rechecks_api_confidence_under_current_policy(tmp_path: Path) -> None:
    store = _store(tmp_path, min_make_confidence=0.90)
    image_bytes = _image_bytes()
    payload = _request_payload()
    request_hash = hash_request(image_bytes, payload)

    store.record_api_result(
        image_bytes=image_bytes,
        request_hash=request_hash,
        request_payload=payload,
        result=_result(make_confidence=0.89),
    )

    lookup = store.lookup(
        image_bytes=image_bytes,
        request_hash=request_hash,
        request_payload=payload,
    )

    assert lookup.match is None
    assert lookup.reason == "exact_ineligible"


def test_lookup_exact_ineligible_can_report_match_for_enforce(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path, min_make_confidence=0.90)
    image_bytes = _image_bytes()
    payload = _request_payload()
    request_hash = hash_request(image_bytes, payload)

    store.record_api_result(
        image_bytes=image_bytes,
        request_hash=request_hash,
        request_payload=payload,
        result=_result(make_confidence=0.89, accepted=False),
    )

    lookup = store.lookup(
        image_bytes=image_bytes,
        request_hash=request_hash,
        request_payload=payload,
        include_ineligible_match=True,
    )

    assert lookup.reason == "exact_ineligible"
    assert lookup.match is not None
    assert lookup.match.kind == "exact"
    assert lookup.match.result.make == "Toyota"
    assert lookup.match.result.accepted is False
    assert lookup.match.result.make_confidence == 0.89


def test_exact_ineligible_evidence_blocks_approximate_reuse(tmp_path: Path) -> None:
    store = _store(tmp_path, min_make_confidence=0.90)
    image_bytes = _image_bytes()
    payload = _request_payload()
    request_hash = hash_request(image_bytes, payload)

    store.record_api_result(
        image_bytes=image_bytes,
        request_hash=request_hash,
        request_payload=payload,
        result=_result(make_confidence=0.89),
    )

    lookup = store.lookup(
        image_bytes=image_bytes,
        request_hash=request_hash,
        request_payload=payload,
    )

    assert lookup.match is None
    assert lookup.reason == "exact_ineligible"


def test_embedding_ineligible_neighbor_blocks_local_reuse(tmp_path: Path) -> None:
    store = _store(tmp_path, min_make_confidence=0.90)
    payload = _request_payload()
    ineligible_bytes = _image_bytes(quality=95)
    eligible_bytes = _image_bytes(quality=90)
    query_bytes = _image_bytes(quality=80)

    store.record_api_result(
        image_bytes=ineligible_bytes,
        request_hash=hash_request(ineligible_bytes, payload),
        request_payload=payload,
        result=_result(make_confidence=0.89),
    )
    store.record_api_result(
        image_bytes=eligible_bytes,
        request_hash=hash_request(eligible_bytes, payload),
        request_payload=payload,
        result=_result(make_confidence=0.95),
    )

    lookup = store.lookup(
        image_bytes=query_bytes,
        request_hash=hash_request(query_bytes, payload),
        request_payload=payload,
    )

    assert lookup.match is None
    assert lookup.reason == "embedding_ineligible"


def test_compact_records_normalize_coordinates_and_keep_raw_response(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    image_bytes = _image_bytes()
    payload = _request_payload()
    result = _result(
        detection_box=BBox(x1=512, y1=1601, x2=1024, y2=1983),
        raw={
            "batch_image": "old-grid.jpg",
            "batch_cell_index": 16,
            "batch_content_box": {
                "x1": 512,
                "y1": 1601,
                "x2": 1024,
                "y2": 1983,
            },
            "response": {"data": {"combinations": ["many cars"]}},
        },
    )
    request_hash = hash_request(image_bytes, payload)
    record = store.record_api_result(
        image_bytes=image_bytes,
        request_hash=request_hash,
        request_payload=payload,
        result=result,
    )
    record_path = store.records_dir / f"{record.record_id}.json"
    legacy_payload = orjson.loads(record_path.read_bytes())
    legacy_payload["result"]["detection_box"] = {
        "x1": 512,
        "y1": 1601,
        "x2": 1024,
        "y2": 1983,
    }
    legacy_payload["result"]["raw"] = result.raw
    record_path.write_bytes(orjson.dumps(legacy_payload))

    changed = store.compact_records()
    compacted = record_path.read_bytes()
    lookup = store.lookup(
        image_bytes=image_bytes,
        request_hash=request_hash,
        request_payload=payload,
    )

    assert changed == 1
    assert b"batch_image" in compacted
    assert b"many cars" in compacted
    assert lookup.match is not None
    assert lookup.match.result.detection_box == BBox(x1=0, y1=0, x2=120, y2=80)
    assert lookup.match.result.raw["response"]["data"]["combinations"] == ["many cars"]


def test_phash_gate_runs_before_embedding_provider(tmp_path: Path) -> None:
    provider = _EmbeddingProvider()
    provider.calls = 0
    store = MMRRetrievalStore(
        tmp_path / "retrieval",
        retrieval_mode="shadow",
        embedding_model="google/gemini-embedding-2",
        embedding_dimensions=768,
        embedding_distance_threshold=0.2,
        phash_max_hamming_distance=0,
        min_neighbors=1,
        embedding_provider=provider,
    )
    stored_bytes = _image_bytes()
    query_image = np.zeros((80, 120, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", query_image)
    assert ok
    query_bytes = encoded.tobytes()
    payload = _request_payload()
    store.record_api_result(
        image_bytes=stored_bytes,
        request_hash=hash_request(stored_bytes, payload),
        request_payload=payload,
        result=_result(),
    )
    provider.calls = 0

    lookup = store.lookup(
        image_bytes=query_bytes,
        request_hash=hash_request(query_bytes, payload),
        request_payload=payload,
    )

    assert lookup.reason == "no_match"
    assert provider.calls == 0


def test_unavailable_embedding_is_still_exact_matchable(tmp_path: Path) -> None:
    class FailingProvider(_EmbeddingProvider):
        def embed(self, image_bytes: bytes) -> list[float]:
            _ = image_bytes
            raise RuntimeError("offline")

    store = MMRRetrievalStore(
        tmp_path / "retrieval",
        retrieval_mode="shadow",
        embedding_model="google/gemini-embedding-2",
        embedding_dimensions=768,
        embedding_distance_threshold=0.2,
        phash_max_hamming_distance=4,
        min_neighbors=1,
        embedding_provider=FailingProvider(),
    )
    image_bytes = _image_bytes()
    payload = _request_payload()
    request_hash = hash_request(image_bytes, payload)
    record = store.record_api_result(
        image_bytes=image_bytes,
        request_hash=request_hash,
        request_payload=payload,
        result=_result(),
    )

    lookup = store.lookup(
        image_bytes=image_bytes,
        request_hash=request_hash,
        request_payload=payload,
    )

    assert record.embedding is None
    assert lookup.reason == "exact_match"
    assert lookup.match is not None


def test_embedding_migration_supersedes_legacy_record(tmp_path: Path) -> None:
    legacy_provider = _EmbeddingProvider()
    legacy_provider.model = "normalized_pixels_v1"
    legacy_store = _store(tmp_path)
    image_bytes = _image_bytes()
    payload = _request_payload()
    original = legacy_store.record_api_result(
        image_bytes=image_bytes,
        request_hash=hash_request(image_bytes, payload),
        request_payload=payload,
        result=_result(),
    )
    original_path = legacy_store.records_dir / f"{original.record_id}.json"
    payload_on_disk = orjson.loads(original_path.read_bytes())
    payload_on_disk["embedding_model"] = legacy_provider.model
    original_path.write_bytes(orjson.dumps(payload_on_disk))

    current_store = MMRRetrievalStore(
        tmp_path / "retrieval",
        retrieval_mode="shadow",
        embedding_model="google/gemini-embedding-2",
        embedding_dimensions=768,
        embedding_distance_threshold=0.2,
        phash_max_hamming_distance=4,
        min_neighbors=1,
        embedding_provider=_EmbeddingProvider(),
    )
    migrated, unavailable = current_store.migrate_embeddings()

    assert (migrated, unavailable) == (1, 0)
    assert original_path.exists()
    active = current_store.active_records()
    assert len(active) == 1
    assert active[0].supersedes_record_id == original.record_id
    assert active[0].embedding_model == "google/gemini-embedding-2"


def _calibration_artifact(
    *, threshold: float, **updates: object
) -> RetrievalCalibrationArtifact:
    values: dict[str, object] = {
        "schema_version": 1,
        "created_at": "2026-01-01T00:00:00+00:00",
        "embedding_model": "google/gemini-embedding-2",
        "embedding_dimensions": 768,
        "phash_max_hamming_distance": 4,
        "threshold": threshold,
        "same_identity_pairs": 3,
        "conflicting_identity_pairs": 3,
        "maximum_same_identity_distance": threshold,
        "minimum_conflicting_identity_distance": 0.5,
    }
    values.update(updates)
    return RetrievalCalibrationArtifact.model_validate(values)


class _QueueEmbeddingProvider(_EmbeddingProvider):
    def __init__(self, vectors: list[list[float]]) -> None:
        super().__init__()
        self._vectors = list(vectors)

    def embed(self, image_bytes: bytes) -> list[float]:
        _ = image_bytes
        self.calls += 1
        return self._vectors.pop(0)


def test_enforce_without_calibration_refuses_embedding_reuse(
    tmp_path: Path,
) -> None:
    provider = _EmbeddingProvider()
    store = _store(tmp_path, retrieval_mode="enforce", embedding_provider=provider)
    stored_bytes = _image_bytes(quality=95)
    query_bytes = _image_bytes(quality=90)
    payload = _request_payload()
    store.record_api_result(
        image_bytes=stored_bytes,
        request_hash=hash_request(stored_bytes, payload),
        request_payload=payload,
        result=_result(),
    )
    calls_after_record = provider.calls
    assert calls_after_record == 1

    lookup = store.lookup(
        image_bytes=query_bytes,
        request_hash=hash_request(query_bytes, payload),
        request_payload=payload,
    )

    assert lookup.match is None
    assert lookup.reason == "calibration_missing"
    assert lookup.candidate_count == 1
    assert provider.calls == calls_after_record


def test_enforce_uses_calibrated_threshold(tmp_path: Path) -> None:
    recording_provider = _QueueEmbeddingProvider([[1.0, 0.0] + [0.0] * 766])
    recording_store = _store(
        tmp_path, retrieval_mode="shadow", embedding_provider=recording_provider
    )
    stored_bytes = _image_bytes(quality=95)
    payload = _request_payload()
    recording_store.record_api_result(
        image_bytes=stored_bytes,
        request_hash=hash_request(stored_bytes, payload),
        request_payload=payload,
        result=_result(),
    )
    save_calibration_artifact(
        tmp_path / "retrieval",
        _calibration_artifact(threshold=0.1),
    )

    query_provider = _QueueEmbeddingProvider([[0.92, 0.3919] + [0.0] * 766])
    store = _store(
        tmp_path, retrieval_mode="enforce", embedding_provider=query_provider
    )
    lookup = store.lookup(
        image_bytes=_image_bytes(quality=90),
        request_hash=hash_request(_image_bytes(quality=90), payload),
        request_payload=payload,
    )

    assert lookup.match is not None
    assert lookup.reason == "embedding_match"
    assert lookup.match.distance > 0.02


def test_stale_calibration_artifact_is_ignored(tmp_path: Path) -> None:
    save_calibration_artifact(
        tmp_path / "retrieval",
        _calibration_artifact(
            threshold=0.1,
            embedding_model="previous_embedding_model_v1",
        ),
    )
    provider = _EmbeddingProvider()
    store = _store(tmp_path, retrieval_mode="enforce", embedding_provider=provider)
    stored_bytes = _image_bytes(quality=95)
    payload = _request_payload()
    store.record_api_result(
        image_bytes=stored_bytes,
        request_hash=hash_request(stored_bytes, payload),
        request_payload=payload,
        result=_result(),
    )

    lookup = store.lookup(
        image_bytes=_image_bytes(quality=90),
        request_hash=hash_request(_image_bytes(quality=90), payload),
        request_payload=payload,
    )

    assert lookup.match is None
    assert lookup.reason == "calibration_missing"
