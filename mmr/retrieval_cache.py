from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import orjson
from pydantic import BaseModel, Field

from models import BBox, MMRResult

EMBEDDING_MODEL = "normalized_pixels_v1"
_EMBEDDING_SIZE = 32


def image_sha256(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()


def request_contract(request_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "tasks": request_payload.get("tasks"),
        "mmrPreference": request_payload.get("mmrPreference"),
    }


def request_contract_hash(request_payload: dict[str, Any]) -> str:
    contract = request_contract(request_payload)
    return hashlib.sha256(orjson.dumps(contract)).hexdigest()


def _decode_image(image_bytes: bytes) -> cv2.typing.MatLike:
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Could not decode image for retrieval embedding")
    return image


def image_embedding(image_bytes: bytes) -> list[float]:
    image = _decode_image(image_bytes)
    resized = cv2.resize(
        image,
        (_EMBEDDING_SIZE, _EMBEDDING_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    vector = rgb.reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        return [0.0] * int(vector.size)
    return (vector / norm).tolist()


def perceptual_hash(image_bytes: bytes) -> int:
    image = _decode_image(image_bytes)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(
        grayscale,
        (32, 32),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)
    coefficients = cv2.dct(resized)[:8, :8]
    low_frequency = coefficients.flatten()
    threshold = float(np.median(low_frequency[1:]))
    value = 0
    for coefficient in low_frequency:
        value = (value << 1) | int(coefficient > threshold)
    return value


def _cosine_distance(left: list[float], right: list[float]) -> float:
    left_array = np.asarray(left, dtype=np.float32)
    right_array = np.asarray(right, dtype=np.float32)
    if left_array.shape != right_array.shape:
        return float("inf")
    left_norm = float(np.linalg.norm(left_array))
    right_norm = float(np.linalg.norm(right_array))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0 if np.array_equal(left_array, right_array) else float("inf")
    similarity = float(np.dot(left_array, right_array) / (left_norm * right_norm))
    return max(0.0, 1.0 - similarity)


def _hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _normalized(value: str | None) -> str:
    return "" if value is None else value.strip().casefold()


def _identity_signature(result: MMRResult) -> tuple[str, str, str]:
    return (
        _normalized(result.make),
        _normalized(result.model),
        _normalized(result.generation),
    )


def _exact_label_signature(result: MMRResult) -> bytes:
    return orjson.dumps(
        [
            result.make,
            result.model,
            result.make_confidence,
            result.model_confidence,
            result.category,
            result.category_confidence,
            result.generation,
            result.generation_confidence,
            result.variation,
            result.variation_confidence,
            result.color,
            result.color_confidence,
            result.view,
            result.view_confidence,
            result.view8,
            result.view8_confidence,
            result.tags,
            result.detection_box.model_dump(mode="json")
            if result.detection_box is not None
            else None,
            result.detection_confidence,
        ]
    )


def _is_eligible(result: MMRResult, min_make_confidence: float) -> bool:
    if not result.accepted or not result.make or not result.model:
        return False
    if result.evidence_source == "human_adjudicated":
        return True
    return (result.make_confidence or 0.0) >= min_make_confidence


def _compact_result_raw(result: MMRResult) -> dict[str, Any]:
    return result.model_dump(
        mode="json",
        exclude={
            "raw",
            "source_image",
            "vehicle_index",
            "api_classification_index",
            "evidence_source",
            "resolution_method",
            "retrieval_record_id",
            "retrieval_distance",
            "retrieval_neighbor_count",
        },
    )


def compact_result_for_cache(result: MMRResult) -> MMRResult:
    return result.model_copy(update={"raw": _compact_result_raw(result)})


def normalize_legacy_batch_result(
    result: MMRResult,
    image_bytes: bytes,
) -> MMRResult:
    """Convert an old grid-relative detection box to source-crop coordinates."""
    content_box_payload = result.raw.get("batch_content_box")
    if not isinstance(content_box_payload, dict) or result.detection_box is None:
        return result
    try:
        content_box = BBox.model_validate(content_box_payload)
    except ValueError:
        return result
    box = result.detection_box
    if (
        content_box.width <= 0
        or content_box.height <= 0
        or not content_box.contains_point(box.center)
    ):
        return result
    image = _decode_image(image_bytes)
    image_height, image_width = image.shape[:2]
    scale_x = image_width / content_box.width
    scale_y = image_height / content_box.height
    return result.model_copy(
        update={
            "detection_box": BBox(
                x1=(box.x1 - content_box.x1) * scale_x,
                y1=(box.y1 - content_box.y1) * scale_y,
                x2=(box.x2 - content_box.x1) * scale_x,
                y2=(box.y2 - content_box.y1) * scale_y,
            )
        }
    )


class RetrievalRecord(BaseModel):
    record_id: str
    created_at: str
    image_sha256: str
    request_hash: str
    request_contract_hash: str
    request_contract: dict[str, Any] = Field(default_factory=dict)
    embedding_model: str
    embedding: list[float]
    perceptual_hash: int
    result: MMRResult
    supersedes_record_id: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievalMatch:
    record: RetrievalRecord
    result: MMRResult
    kind: Literal["exact", "embedding"]
    distance: float
    perceptual_distance: int
    neighbor_count: int


@dataclass(frozen=True, slots=True)
class RetrievalLookup:
    match: RetrievalMatch | None
    candidate_count: int
    reason: str


class MMRRetrievalStore:
    """Durable, auditable evidence store for exact and near-duplicate reuse."""

    def __init__(
        self,
        root: Path,
        *,
        embedding_distance_threshold: float,
        phash_max_hamming_distance: int,
        min_neighbors: int,
        min_make_confidence: float = 0.0,
    ) -> None:
        self.root = root
        self.records_dir = root / "records"
        self.images_dir = root / "images"
        self.audit_path = root / "lookup_audit.jsonl"
        self.embedding_distance_threshold = embedding_distance_threshold
        self.phash_max_hamming_distance = phash_max_hamming_distance
        self.min_neighbors = min_neighbors
        self.min_make_confidence = min_make_confidence
        self._record_cache: dict[str, RetrievalRecord] | None = None
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def _record_path(self, record_id: str) -> Path:
        return self.records_dir / f"{record_id}.json"

    def _iter_records(self) -> Iterable[RetrievalRecord]:
        if self._record_cache is None:
            self._record_cache = {}
            for path in sorted(self.records_dir.glob("*.json")):
                try:
                    record = RetrievalRecord.model_validate(
                        orjson.loads(path.read_bytes())
                    )
                except (OSError, ValueError):
                    continue
                self._record_cache[record.record_id] = record
        yield from self._record_cache.values()

    def _active_records(self) -> list[RetrievalRecord]:
        records = list(self._iter_records())
        superseded_ids = {
            record.supersedes_record_id
            for record in records
            if record.supersedes_record_id is not None
        }
        return [record for record in records if record.record_id not in superseded_ids]

    def _write_record(self, record: RetrievalRecord) -> None:
        path = self._record_path(record.record_id)
        if path.exists():
            return
        temporary_path = path.with_suffix(".tmp")
        temporary_path.write_bytes(
            orjson.dumps(record.model_dump(mode="json"), option=orjson.OPT_INDENT_2)
        )
        temporary_path.replace(path)
        if self._record_cache is not None:
            self._record_cache[record.record_id] = record

    def _replace_record(self, record: RetrievalRecord) -> None:
        path = self._record_path(record.record_id)
        temporary_path = path.with_suffix(".tmp")
        temporary_path.write_bytes(
            orjson.dumps(record.model_dump(mode="json"), option=orjson.OPT_INDENT_2)
        )
        temporary_path.replace(path)
        if self._record_cache is not None:
            self._record_cache[record.record_id] = record

    def record_api_result(
        self,
        *,
        image_bytes: bytes,
        request_hash: str,
        request_payload: dict[str, Any],
        result: MMRResult,
    ) -> RetrievalRecord:
        image_digest = image_sha256(image_bytes)
        stored_image_path = self.images_dir / f"{image_digest}.jpg"
        if not stored_image_path.exists():
            stored_image_path.write_bytes(image_bytes)

        stored_result = result.model_copy(
            update={
                "source_image": stored_image_path,
                "evidence_source": result.evidence_source or "api_confirmed",
                "resolution_method": result.resolution_method or "external_api",
            }
        )
        stored_result = compact_result_for_cache(stored_result)
        contract = request_contract(request_payload)
        contract_hash = request_contract_hash(request_payload)
        embedding = image_embedding(image_bytes)
        image_hash = perceptual_hash(image_bytes)
        record_payload = {
            "image_sha256": image_digest,
            "request_hash": request_hash,
            "request_contract": contract,
            "request_contract_hash": contract_hash,
            "embedding_model": EMBEDDING_MODEL,
            "embedding": embedding,
            "perceptual_hash": image_hash,
            "result": stored_result.model_dump(mode="json"),
            "supersedes_record_id": None,
        }
        record_id = hashlib.sha256(orjson.dumps(record_payload)).hexdigest()
        record = RetrievalRecord(
            record_id=record_id,
            created_at=datetime.now(UTC).isoformat(),
            image_sha256=image_digest,
            request_hash=request_hash,
            request_contract_hash=contract_hash,
            request_contract=contract,
            embedding_model=EMBEDDING_MODEL,
            embedding=embedding,
            perceptual_hash=image_hash,
            result=stored_result,
            supersedes_record_id=None,
        )
        self._write_record(record)
        return record

    def record_adjudication(
        self,
        *,
        record_id: str,
        result: MMRResult,
    ) -> RetrievalRecord:
        original = next(
            (
                record
                for record in self._iter_records()
                if record.record_id == record_id
            ),
            None,
        )
        if original is None:
            raise KeyError(f"Retrieval record not found: {record_id}")

        stored_image_path = self.images_dir / f"{original.image_sha256}.jpg"
        if not stored_image_path.exists():
            raise FileNotFoundError(
                f"Stored image for retrieval record does not exist: {stored_image_path}"
            )
        stored_result = result.model_copy(
            update={
                "source_image": stored_image_path,
                "evidence_source": "human_adjudicated",
                "resolution_method": "human_adjudication",
            }
        )
        stored_result = compact_result_for_cache(stored_result)
        record_payload = {
            "image_sha256": original.image_sha256,
            "request_hash": original.request_hash,
            "request_contract": original.request_contract,
            "request_contract_hash": original.request_contract_hash,
            "embedding_model": original.embedding_model,
            "embedding": original.embedding,
            "perceptual_hash": original.perceptual_hash,
            "result": stored_result.model_dump(mode="json"),
            "supersedes_record_id": original.record_id,
        }
        corrected_record = RetrievalRecord(
            record_id=hashlib.sha256(orjson.dumps(record_payload)).hexdigest(),
            created_at=datetime.now(UTC).isoformat(),
            image_sha256=original.image_sha256,
            request_hash=original.request_hash,
            request_contract_hash=original.request_contract_hash,
            request_contract=original.request_contract,
            embedding_model=original.embedding_model,
            embedding=original.embedding,
            perceptual_hash=original.perceptual_hash,
            result=stored_result,
            supersedes_record_id=original.record_id,
        )
        self._write_record(corrected_record)
        return corrected_record

    def compact_records(self) -> int:
        changed = 0
        for path in sorted(self.records_dir.glob("*.json")):
            try:
                record = RetrievalRecord.model_validate(orjson.loads(path.read_bytes()))
            except (OSError, ValueError):
                continue
            image_path = self.images_dir / f"{record.image_sha256}.jpg"
            if not image_path.exists():
                continue
            image_bytes = image_path.read_bytes()
            result = compact_result_for_cache(
                normalize_legacy_batch_result(record.result, image_bytes)
            )
            if result == record.result:
                continue
            self._replace_record(record.model_copy(update={"result": result}))
            changed += 1
        return changed

    def lookup(
        self,
        *,
        image_bytes: bytes,
        request_hash: str,
        request_payload: dict[str, Any],
    ) -> RetrievalLookup:
        records = self._active_records()
        image_digest = image_sha256(image_bytes)
        exact_records = [
            record
            for record in records
            if record.image_sha256 == image_digest
            and record.request_hash == request_hash
        ]
        if exact_records:
            if any(
                not _is_eligible(record.result, self.min_make_confidence)
                for record in exact_records
            ):
                return RetrievalLookup(
                    match=None,
                    candidate_count=len(exact_records),
                    reason="exact_ineligible",
                )
            signatures = {
                _exact_label_signature(record.result) for record in exact_records
            }
            if len(signatures) != 1:
                return RetrievalLookup(
                    match=None,
                    candidate_count=len(exact_records),
                    reason="exact_conflict",
                )
            return RetrievalLookup(
                match=RetrievalMatch(
                    record=exact_records[0],
                    result=exact_records[0].result,
                    kind="exact",
                    distance=0.0,
                    perceptual_distance=0,
                    neighbor_count=len(exact_records),
                ),
                candidate_count=len(exact_records),
                reason="exact_match",
            )

        query_embedding = image_embedding(image_bytes)
        query_hash = perceptual_hash(image_bytes)
        contract_hash = request_contract_hash(request_payload)
        nearby_candidates = [
            (
                record,
                _cosine_distance(query_embedding, record.embedding),
                _hamming_distance(query_hash, record.perceptual_hash),
            )
            for record in records
            if record.embedding_model == EMBEDDING_MODEL
            and record.request_contract_hash == contract_hash
        ]
        nearby_candidates = [
            candidate
            for candidate in nearby_candidates
            if candidate[1] <= self.embedding_distance_threshold
            and candidate[2] <= self.phash_max_hamming_distance
        ]
        if any(
            not _is_eligible(record.result, self.min_make_confidence)
            for record, _distance, _phash_distance in nearby_candidates
        ):
            return RetrievalLookup(
                match=None,
                candidate_count=len(nearby_candidates),
                reason="embedding_ineligible",
            )

        candidates_by_image: dict[str, list[tuple[RetrievalRecord, float, int]]] = {}
        for candidate in nearby_candidates:
            candidates_by_image.setdefault(candidate[0].image_sha256, []).append(
                candidate
            )
        for image_candidates in candidates_by_image.values():
            if (
                len({_identity_signature(item[0].result) for item in image_candidates})
                > 1
            ):
                return RetrievalLookup(
                    match=None,
                    candidate_count=len(nearby_candidates),
                    reason="embedding_conflict",
                )

        candidates = [
            min(
                image_candidates, key=lambda item: (item[1], item[2], item[0].record_id)
            )
            for image_candidates in candidates_by_image.values()
        ]
        candidates.sort(key=lambda item: (item[1], item[2], item[0].record_id))
        if not candidates:
            return RetrievalLookup(match=None, candidate_count=0, reason="no_match")
        if len(candidates) < self.min_neighbors:
            return RetrievalLookup(
                match=None,
                candidate_count=len(candidates),
                reason="insufficient_neighbors",
            )

        identity_signatures = {
            _identity_signature(candidate[0].result) for candidate in candidates
        }
        if len(identity_signatures) != 1:
            return RetrievalLookup(
                match=None,
                candidate_count=len(candidates),
                reason="embedding_conflict",
            )

        record, distance, phash_distance = candidates[0]
        variation_values = {
            _normalized(record.result.variation)
            for record, _distance, _phash_distance in nearby_candidates
        }
        variation_is_agreed = len(variation_values) == 1 and bool(
            next(iter(variation_values), "")
        )
        nearest_result = record.result.model_copy(
            update={
                "category": None,
                "category_confidence": None,
                "color": None,
                "color_confidence": None,
                "view": None,
                "view_confidence": None,
                "view8": None,
                "view8_confidence": None,
                "tags": [],
                "detection_box": None,
                "detection_confidence": None,
                "variation": (record.result.variation if variation_is_agreed else None),
                "variation_confidence": (
                    record.result.variation_confidence if variation_is_agreed else None
                ),
                "raw": {
                    "retrieval_source_record_id": record.record_id,
                    "retrieval_source_response": record.result.raw,
                },
            }
        )
        return RetrievalLookup(
            match=RetrievalMatch(
                record=record,
                result=nearest_result,
                kind="embedding",
                distance=distance,
                perceptual_distance=phash_distance,
                neighbor_count=len(candidates),
            ),
            candidate_count=len(candidates),
            reason="embedding_match",
        )

    def append_lookup_audit(
        self,
        *,
        image_sha256_value: str,
        request_hash: str,
        lookup: RetrievalLookup,
        action: Literal["reuse", "shadow", "api_fallback"],
    ) -> None:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "image_sha256": image_sha256_value,
            "request_hash": request_hash,
            "action": action,
            "reason": lookup.reason,
            "candidate_count": lookup.candidate_count,
            "match": {
                "record_id": lookup.match.record.record_id,
                "kind": lookup.match.kind,
                "distance": lookup.match.distance,
                "perceptual_distance": lookup.match.perceptual_distance,
                "neighbor_count": lookup.match.neighbor_count,
            }
            if lookup.match is not None
            else None,
        }
        with self.audit_path.open("ab") as handle:
            handle.write(orjson.dumps(payload))
            handle.write(b"\n")
