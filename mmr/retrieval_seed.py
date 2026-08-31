from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config import AppConfig
from mmr.embeddings import ImageEmbeddingProvider, build_embedding_provider
from mmr.retrieval_cache import MMRRetrievalStore
from mmr.trafficeye import build_single_request_payload
from mmr.trafficeye_batch_grid import (
    decode_image,
    normalize_batch_result_for_source_crop,
)
from mmr.trafficeye_cache import hash_request
from models import MMRResult, TrackSummary
from storage.run_store import RunStore


@dataclass(frozen=True, slots=True)
class RetrievalSeedSummary:
    run_dir: Path
    imported: int
    skipped_unaccepted: int
    skipped_missing_image: int


def _build_store(
    config: AppConfig,
    cache_dir: Path,
    embedding_provider: ImageEmbeddingProvider | None = None,
) -> MMRRetrievalStore:
    provider = embedding_provider or build_embedding_provider(config, cache_dir)
    return MMRRetrievalStore.from_config(config, cache_dir, provider)


def _resolve_image_path(
    store: RunStore,
    track_id: int,
    result: MMRResult,
    summaries_by_track: dict[int, TrackSummary],
) -> Path | None:
    candidates: list[Path] = []
    if result.source_image is not None:
        candidates.extend(
            [
                result.source_image,
                store.root / result.source_image,
                store.crops_dir / result.source_image.name,
            ]
        )
    summary = summaries_by_track.get(track_id)
    if summary is not None:
        candidates.extend(candidate.image_path for candidate in summary.candidates)
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def seed_retrieval_cache(
    *,
    run_dirs: list[Path],
    config: AppConfig,
    cache_dir: Path,
    embedding_provider: ImageEmbeddingProvider | None = None,
) -> list[RetrievalSeedSummary]:
    store = _build_store(config, cache_dir, embedding_provider)
    summaries: list[RetrievalSeedSummary] = []
    for run_dir in run_dirs:
        run_store = RunStore.from_existing(run_dir)
        imported = 0
        skipped_unaccepted = 0
        skipped_missing_image = 0
        summaries_by_track = {
            summary.track_id: summary for summary in run_store.tracks.iter()
        }
        for track_id, result in run_store.labels.read().items():
            if not result.accepted or not result.make or not result.model:
                skipped_unaccepted += 1
                continue
            image_path = _resolve_image_path(
                run_store, track_id, result, summaries_by_track
            )
            if image_path is None:
                skipped_missing_image += 1
                continue
            image_bytes = image_path.read_bytes()
            image = decode_image(image_bytes, image_path)
            height, width = image.shape[:2]
            request_payload = build_single_request_payload(
                width=width,
                height=height,
                mmr_preference=config.mmr.mmr_preference,
            )
            store.record_api_result(
                image_bytes=image_bytes,
                request_hash=hash_request(image_bytes, request_payload),
                request_payload=request_payload,
                result=normalize_batch_result_for_source_crop(
                    result,
                    image_width=width,
                    image_height=height,
                ),
            )
            imported += 1
        summaries.append(
            RetrievalSeedSummary(
                run_dir=run_dir,
                imported=imported,
                skipped_unaccepted=skipped_unaccepted,
                skipped_missing_image=skipped_missing_image,
            )
        )
    return summaries
