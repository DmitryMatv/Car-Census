import numpy as np
import pytest
from config import AppConfig
from reid import (
    TorchvisionEmbedder,
    TrackAppearanceMemory,
    build_embedder,
)


def _vec(*values: float) -> np.ndarray:
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_memory_starts_empty() -> None:
    memory = TrackAppearanceMemory(history_length=4)

    assert memory.known(1) is False
    assert memory.similarity(1, _vec(1, 0)) is None


def test_memory_observe_and_similarity() -> None:
    memory = TrackAppearanceMemory(history_length=4)
    vector = _vec(1.0, 0.0)
    memory.observe(3, vector)

    assert memory.known(3) is True
    assert memory.similarity(3, vector) == pytest.approx(1.0)
    assert memory.similarity(3, _vec(0.0, 1.0)) == pytest.approx(0.0)


def test_memory_similarity_returns_best_of_history() -> None:
    memory = TrackAppearanceMemory(history_length=4)
    memory.observe(3, _vec(0.0, 1.0))
    memory.observe(3, _vec(1.0, 0.0))

    assert memory.similarity(3, _vec(1.0, 0.0)) == pytest.approx(1.0)
    assert memory.similarity(3, _vec(0.0, 1.0)) == pytest.approx(1.0)


def test_memory_history_is_bounded() -> None:
    memory = TrackAppearanceMemory(history_length=2)
    memory.observe(3, _vec(1.0, 0.0))
    memory.observe(3, _vec(0.0, 1.0))
    memory.observe(3, _vec(1.0, 1.0))

    # 只保留最近 2 条：与最早一条的相似度不再可达
    assert memory.similarity(3, _vec(0.0, 1.0)) == pytest.approx(1.0)


def test_memory_forget_clears_track() -> None:
    memory = TrackAppearanceMemory(history_length=4)
    memory.observe(3, _vec(1.0, 0.0))
    memory.forget(3)

    assert memory.known(3) is False
    assert memory.similarity(3, _vec(1.0, 0.0)) is None


def test_memory_ignores_zero_vector() -> None:
    memory = TrackAppearanceMemory(history_length=4)
    memory.observe(3, np.zeros(3, dtype=np.float32))

    assert memory.known(3) is False


def test_build_embedder_returns_none_when_disabled() -> None:
    class FakeReidConfig:
        enabled = False
        device = "auto"
        batch_size = 16

    assert build_embedder(FakeReidConfig()) is None


def test_build_embedder_constructs_without_network() -> None:
    class FakeReidConfig:
        enabled = True
        device = "cpu"
        batch_size = 8

    embedder = build_embedder(FakeReidConfig())

    assert embedder is not None
    assert isinstance(embedder, TorchvisionEmbedder)


def test_torchvision_embedder_contract() -> None:
    """契约测试：真实模型在首次 embed 时才构建（懒加载）。"""
    embedder = TorchvisionEmbedder(device="cpu", batch_size=4)

    # 构造后模型未加载，不触发任何下载
    assert embedder._model is None
