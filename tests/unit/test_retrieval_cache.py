from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import orjson

from mmr.retrieval_cache import MMRRetrievalStore
from mmr.trafficeye_cache import hash_request
from models import BBox, MMRResult


def _image_bytes(quality: int = 95) -> bytes:
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[:, :] = [80, 140, 200]
    image[20:60, 30:90] = [200, 100, 40]
    ok, encoded = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    assert ok
    return encoded.tobytes()


def _store(
    tmp_path: Path,
    *,
    min_neighbors: int = 1,
    min_make_confidence: float = 0.0,
) -> MMRRetrievalStore:
    return MMRRetrievalStore(
        tmp_path / "retrieval",
        embedding_distance_threshold=0.02,
        phash_max_hamming_distance=4,
        min_neighbors=min_neighbors,
        min_make_confidence=min_make_confidence,
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
        result=_result(),
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
        result=_result(),
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


def test_compact_records_remove_batch_trace_and_normalize_coordinates(
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
    assert b"batch_image" not in compacted
    assert b"many cars" not in compacted
    assert lookup.match is not None
    assert lookup.match.result.detection_box == BBox(
        x1=0, y1=0, x2=120, y2=80
    )
