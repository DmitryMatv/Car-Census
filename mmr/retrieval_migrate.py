from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config import AppConfig
from mmr.embeddings import ImageEmbeddingProvider, build_embedding_provider
from mmr.retrieval_cache import MMRRetrievalStore


@dataclass(frozen=True, slots=True)
class RetrievalMigrationSummary:
    migrated: int
    unavailable: int


def migrate_retrieval_embeddings(
    *,
    config: AppConfig,
    cache_dir: Path,
    embedding_provider: ImageEmbeddingProvider | None = None,
) -> RetrievalMigrationSummary:
    provider = embedding_provider or build_embedding_provider(config, cache_dir)
    store = MMRRetrievalStore.from_config(config, cache_dir, provider)
    migrated, unavailable = store.migrate_embeddings()
    return RetrievalMigrationSummary(migrated=migrated, unavailable=unavailable)
