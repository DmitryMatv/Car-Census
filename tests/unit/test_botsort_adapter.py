import builtins

import numpy as np
import pytest

from config import AppConfig
from trackers.botsort import (
    BotSortAdapter,
    _create_botsort_tracker,
    _resolve_cmc_method,
)
from models import BBox, Detection


class FakeTracker:
    def __init__(self, tracks):
        self.tracks = tracks
        self.calls = []

    def update(self, detections, frame):
        self.calls.append((detections, frame))
        return self.tracks


def test_botsort_adapter_calls_tracker_for_empty_detections() -> None:
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    fake_tracker = FakeTracker(np.empty((0, 8), dtype=np.float32))
    adapter = BotSortAdapter(AppConfig(), tracker=fake_tracker)

    tracked = adapter.update([], frame)

    assert fake_tracker.calls[0][0].shape == (0, 6)
    assert fake_tracker.calls[0][1] is frame
    assert len(tracked) == 0
    assert tracked.tracker_id is not None


def test_botsort_adapter_converts_detections_and_tracks() -> None:
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    fake_tracker = FakeTracker(
        np.array([[1, 2, 11, 12, 42, 0.91, 2, 0]], dtype=np.float32)
    )
    adapter = BotSortAdapter(AppConfig(), tracker=fake_tracker)
    detections = [
        Detection(
            bbox=BBox(x1=1, y1=2, x2=11, y2=12),
            confidence=0.88,
            class_id=2,
            class_name="car",
        )
    ]

    tracked = adapter.update(detections, frame)

    np.testing.assert_allclose(
        fake_tracker.calls[0][0],
        np.array([[1, 2, 11, 12, 0.88, 2]], dtype=np.float32),
    )
    assert tracked.tracker_id.tolist() == [42]
    assert tracked.confidence.tolist() == pytest.approx([0.91])
    assert tracked.class_id.tolist() == [2]
    assert tracked.data["class_name"].tolist() == ["car"]


def test_botsort_adapter_handles_missing_class_id() -> None:
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    fake_tracker = FakeTracker(np.empty((0, 8), dtype=np.float32))
    adapter = BotSortAdapter(AppConfig(), tracker=fake_tracker)
    detections = [
        Detection(
            bbox=BBox(x1=1, y1=2, x2=11, y2=12),
            confidence=0.88,
        )
    ]

    adapter.update(detections, frame)

    assert fake_tracker.calls[0][0][0, 5] == -1


def test_create_botsort_tracker_raises_clear_install_error(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "boxmot.trackers":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="BoT-SORT tracking requires BoxMOT"):
        _create_botsort_tracker(AppConfig(), frame_rate=30)


def test_resolve_cmc_method_disables_none_string_for_older_configs() -> None:
    config = AppConfig.model_validate({"tracker": {"cmc_method": "none"}})

    assert _resolve_cmc_method(config) is None


def test_resolve_cmc_method_keeps_supported_method() -> None:
    config = AppConfig.model_validate({"tracker": {"cmc_method": "ecc"}})

    assert _resolve_cmc_method(config) == "ecc"
