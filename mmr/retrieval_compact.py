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
    store = MMRRetrievalStore.from_config(config, cache_dir, provider)
    return store.compact_records()
