from __future__ import annotations

from pathlib import Path

from config import AppConfig
from mmr.embeddings import ImageEmbeddingProvider, build_embedding_provider
from mmr.retrieval_cache import MMRRetrievalStore


def compact_retrieval_cache(
    *,
    config: AppConfig,
    cache_dir: Path,
    embedding_provider: ImageEmbeddingProvider | None = None,
) -> int:
    provider = embedding_provider or build_embedding_provider(config, cache_dir)
    store = MMRRetrievalStore(
        cache_dir / "retrieval",
        retrieval_mode=config.mmr.retrieval_mode,
        embedding_model=config.mmr.retrieval_embedding_model,
        embedding_dimensions=config.mmr.retrieval_embedding_dimensions,
        embedding_distance_threshold=config.mmr.retrieval_embedding_distance_threshold,
        phash_max_hamming_distance=config.mmr.retrieval_phash_max_hamming_distance,
        min_neighbors=config.mmr.retrieval_min_neighbors,
        min_make_confidence=config.mmr.accept_model_confidence,
        embedding_provider=provider,
    )
    return store.compact_records()
