import numpy as np
import pytest
import supervision as sv
from trackers import BoTSORTTracker

from config import AppConfig
from models import BBox, Detection
from tracking_adapters.botsort import (
    BotSortAdapter,
    _create_botsort_tracker,
    _effective_frame_rate,
)


class FakeTracker:
    def __init__(self, tracks: sv.Detections) -> None:
        self.tracks = tracks
        self.calls = []

    def update(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
        self.calls.append((detections, frame))
        return self.tracks


def _empty_tracks() -> sv.Detections:
    tracks = sv.Detections.empty()
    tracks.tracker_id = np.empty((0,), dtype=np.int32)
    return tracks


def test_botsort_adapter_calls_tracker_for_empty_detections() -> None:
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    fake_tracker = FakeTracker(_empty_tracks())
    adapter = BotSortAdapter(AppConfig(), tracker=fake_tracker)

    tracked = adapter.update([], frame)

    sent_detections = fake_tracker.calls[0][0]
    assert len(sent_detections) == 0
    assert sent_detections.xyxy.shape == (0, 4)
    assert sent_detections.confidence is not None
    assert sent_detections.class_id is not None
    assert sent_detections.data["class_name"].tolist() == []
    assert fake_tracker.calls[0][1] is frame
    assert len(tracked) == 0
    assert tracked.tracker_id is not None


def test_botsort_adapter_converts_detections_to_supervision() -> None:
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    fake_tracker = FakeTracker(_empty_tracks())
    adapter = BotSortAdapter(AppConfig(), tracker=fake_tracker)
    detections = [
        Detection(
            bbox=BBox(x1=1, y1=2, x2=11, y2=12),
            confidence=0.88,
            class_id=2,
            class_name="car",
        )
    ]

    adapter.update(detections, frame)

    sent_detections = fake_tracker.calls[0][0]
    np.testing.assert_allclose(
        sent_detections.xyxy,
        np.array([[1, 2, 11, 12]], dtype=np.float32),
    )
    assert sent_detections.confidence.tolist() == pytest.approx([0.88])
    assert sent_detections.class_id.tolist() == [2]
    assert sent_detections.data["class_name"].tolist() == ["car"]


def test_botsort_adapter_handles_missing_class_id_and_class_name() -> None:
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    fake_tracker = FakeTracker(_empty_tracks())
    adapter = BotSortAdapter(AppConfig(), tracker=fake_tracker)
    detections = [
        Detection(
            bbox=BBox(x1=1, y1=2, x2=11, y2=12),
            confidence=0.88,
        )
    ]

    adapter.update(detections, frame)

    sent_detections = fake_tracker.calls[0][0]
    assert sent_detections.class_id.tolist() == [-1]
    assert sent_detections.data["class_name"].tolist() == [""]


def test_create_botsort_tracker_uses_analysis_frame_rate() -> None:
    tracker = _create_botsort_tracker(AppConfig(), frame_rate=10)

    assert isinstance(tracker, BoTSORTTracker)
    assert _effective_frame_rate(AppConfig(), frame_rate=10) == 10.0
    assert tracker.maximum_frames_without_update == 10


def test_create_botsort_tracker_allows_configured_frame_rate_override() -> None:
    config = AppConfig.model_validate({"tracker": {"frame_rate": 15}})
    tracker = _create_botsort_tracker(config, frame_rate=10)

    assert _effective_frame_rate(config, frame_rate=10) == 15.0
    assert tracker.maximum_frames_without_update == 15


def test_create_botsort_tracker_enables_cmc_by_default() -> None:
    tracker = _create_botsort_tracker(AppConfig(), frame_rate=30)

    assert tracker.enable_cmc is True
    assert tracker.cmc is not None


def test_create_botsort_tracker_can_disable_cmc() -> None:
    config = AppConfig.model_validate({"tracker": {"enable_cmc": False}})
    tracker = _create_botsort_tracker(config, frame_rate=30)

    assert tracker.enable_cmc is False
    assert tracker.cmc is None
