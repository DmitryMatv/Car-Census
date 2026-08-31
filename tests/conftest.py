from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from config import AppConfig, build_effective_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ConfigFactory = Callable[[dict[str, Any] | None], AppConfig]


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def config_factory() -> ConfigFactory:
    def build(overrides: dict[str, Any] | None = None) -> AppConfig:
        return build_effective_config(root=PROJECT_ROOT, overrides=overrides)

    return build


@pytest.fixture
def default_config(config_factory: ConfigFactory) -> AppConfig:
    return config_factory(None)
