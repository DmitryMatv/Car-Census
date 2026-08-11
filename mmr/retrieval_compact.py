from __future__ import annotations

from pathlib import Path

from config import AppConfig
from mmr.retrieval_cache import MMRRetrievalStore


def compact_retrieval_cache(*, config: AppConfig, cache_dir: Path) -> int:
    store = MMRRetrievalStore(
        cache_dir / "retrieval",
        embedding_distance_threshold=config.mmr.retrieval_embedding_distance_threshold,
        phash_max_hamming_distance=config.mmr.retrieval_phash_max_hamming_distance,
        min_neighbors=config.mmr.retrieval_min_neighbors,
        min_make_confidence=config.mmr.accept_model_confidence,
    )
    return store.compact_records()
