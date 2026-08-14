from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config import AppConfig
from mmr.embeddings import ImageEmbeddingProvider, OpenRouterEmbeddingProvider
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
    provider = embedding_provider or OpenRouterEmbeddingProvider(
        api_key_env=config.mmr.retrieval_embedding_api_key_env,
        model=config.mmr.retrieval_embedding_model,
        dimensions=config.mmr.retrieval_embedding_dimensions,
        cache_dir=cache_dir / "embeddings",
        timeout=config.mmr.timeout_seconds,
    )
    store = MMRRetrievalStore(
        cache_dir / "retrieval",
        embedding_distance_threshold=config.mmr.retrieval_embedding_distance_threshold,
        phash_max_hamming_distance=config.mmr.retrieval_phash_max_hamming_distance,
        min_neighbors=config.mmr.retrieval_min_neighbors,
        min_make_confidence=config.mmr.accept_model_confidence,
        embedding_provider=provider,
    )
    migrated, unavailable = store.migrate_embeddings()
    return RetrievalMigrationSummary(migrated=migrated, unavailable=unavailable)
