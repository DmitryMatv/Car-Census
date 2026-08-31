from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from config import AppConfig
from mmr.retrieval_cache import MMRRetrievalStore, RetrievalRecord, _is_eligible
from mmr.retrieval_calibration_artifact import (
    RetrievalCalibrationArtifact,
    save_calibration_artifact,
)
from mmr.retrieval_similarity import blockwise_cosine_distances, normalized_identity

_CALIBRATION_BLOCK_SIZE = 256
_BYTE_POPCOUNT = np.asarray([value.bit_count() for value in range(256)], dtype=np.uint8)


@dataclass(frozen=True, slots=True)
class RetrievalCalibrationReport:
    same_identity_pairs: int
    conflicting_identity_pairs: int
    maximum_same_identity_distance: float | None
    minimum_conflicting_identity_distance: float | None
    usable_threshold: float | None


def _identity(record: RetrievalRecord) -> tuple[str, str, str]:
    result = record.result
    return normalized_identity(result.make, result.model, result.generation)


def _calibration_pair_stats(
    records: list[RetrievalRecord],
    *,
    phash_max_hamming_distance: int,
) -> tuple[int, int, float | None, float | None]:
    if len(records) < 2:
        return 0, 0, None, None

    embeddings = np.asarray(
        [record.embedding or [] for record in records], dtype=np.float32
    )

    identity_ids: dict[tuple[str, str, str], int] = {}
    identity_codes = np.empty(len(records), dtype=np.intp)
    contract_ids: dict[str, int] = {}
    contract_codes = np.empty(len(records), dtype=np.intp)
    perceptual_hashes = np.asarray(
        [record.perceptual_hash for record in records], dtype=np.uint64
    )
    for index, record in enumerate(records):
        identity = _identity(record)
        identity_codes[index] = identity_ids.setdefault(identity, len(identity_ids))
        contract = record.request_contract_hash
        contract_codes[index] = contract_ids.setdefault(contract, len(contract_ids))

    same_count = 0
    conflicting_count = 0
    maximum_same: float | None = None
    minimum_conflicting: float | None = None
    for start in range(0, len(records), _CALIBRATION_BLOCK_SIZE):
        end = min(start + _CALIBRATION_BLOCK_SIZE, len(records))
        distances = blockwise_cosine_distances(embeddings, start, end)

        hash_xor = np.bitwise_xor(
            perceptual_hashes[start:end, None], perceptual_hashes[None, :]
        )
        hash_bytes = hash_xor.view(np.uint8).reshape(hash_xor.shape + (8,))
        hamming_distances = _BYTE_POPCOUNT[hash_bytes].sum(axis=2)
        row_indices = np.arange(start, end)[:, None]
        column_indices = np.arange(start, len(records))[None, :]
        upper_triangle = column_indices > row_indices
        calibration_candidates = (
            upper_triangle
            & (hamming_distances[:, start:] <= phash_max_hamming_distance)
            & (contract_codes[start:end, None] == contract_codes[None, start:])
        )
        same_identity = calibration_candidates & (
            identity_codes[start:end, None] == identity_codes[None, start:]
        )
        conflicting_identity = calibration_candidates & ~same_identity
        distances = distances[:, start:]

        same_distances = distances[same_identity]
        conflicting_distances = distances[conflicting_identity]
        same_count += int(same_distances.size)
        conflicting_count += int(conflicting_distances.size)
        if same_distances.size:
            candidate = float(same_distances.max())
            maximum_same = (
                candidate if maximum_same is None else max(maximum_same, candidate)
            )
        if conflicting_distances.size:
            candidate = float(conflicting_distances.min())
            minimum_conflicting = (
                candidate
                if minimum_conflicting is None
                else min(minimum_conflicting, candidate)
            )

    return (
        same_count,
        conflicting_count,
        maximum_same,
        minimum_conflicting,
    )


def calibrate_retrieval_cache(
    *, config: AppConfig, cache_dir: Path
) -> RetrievalCalibrationReport:
    store = MMRRetrievalStore.from_config(config, cache_dir)
    records = [
        record
        for record in store.active_records()
        if record.embedding_model == config.mmr.retrieval_embedding_model
        and record.embedding is not None
        and len(record.embedding) == config.mmr.retrieval_embedding_dimensions
        and _is_eligible(record.result, config.mmr.accept_model_confidence)
    ]
    (
        same_count,
        conflicting_count,
        maximum_same,
        minimum_conflicting,
    ) = _calibration_pair_stats(
        records,
        phash_max_hamming_distance=config.mmr.retrieval_phash_max_hamming_distance,
    )
    enough_evidence = (
        same_count >= config.mmr.retrieval_calibration_min_same_identity
        and conflicting_count
        >= config.mmr.retrieval_calibration_min_conflicting_identity
    )
    usable_threshold = (
        maximum_same
        if enough_evidence
        and maximum_same is not None
        and minimum_conflicting is not None
        and maximum_same < minimum_conflicting
        else None
    )
    report = RetrievalCalibrationReport(
        same_identity_pairs=same_count,
        conflicting_identity_pairs=conflicting_count,
        maximum_same_identity_distance=maximum_same,
        minimum_conflicting_identity_distance=minimum_conflicting,
        usable_threshold=usable_threshold,
    )
    if (
        usable_threshold is not None
        and maximum_same is not None
        and minimum_conflicting is not None
    ):
        save_calibration_artifact(
            store.root,
            RetrievalCalibrationArtifact(
                schema_version=1,
                created_at=datetime.now(UTC).isoformat(),
                embedding_model=config.mmr.retrieval_embedding_model,
                embedding_dimensions=config.mmr.retrieval_embedding_dimensions,
                phash_max_hamming_distance=config.mmr.retrieval_phash_max_hamming_distance,
                threshold=usable_threshold,
                same_identity_pairs=same_count,
                conflicting_identity_pairs=conflicting_count,
                maximum_same_identity_distance=maximum_same,
                minimum_conflicting_identity_distance=minimum_conflicting,
            ),
        )
    return report
