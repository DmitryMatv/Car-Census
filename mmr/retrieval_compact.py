from __future__ import annotations

from pathlib import Path

from config import AppConfig
from mmr.retrieval_cache import MMRRetrievalStore


def compact_retrieval_cache(*, config: AppConfig, cache_dir: Path) -> int:
    store = MMRRetrievalStore.from_config(config, cache_dir)
    return store.compact_records()
