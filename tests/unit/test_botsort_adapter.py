import numpy as np
import pytest
import supervision as sv
from trackers import BoTSORTTracker

from config import AppConfig
from tracking_adapters.botsort import (
    BotSortAdapter,
    _create_botsort_tracker,
    _effective_frame_rate,
)


class FakeTracker:
    def __init__(self, tracks: sv.Detections) -> None:
        self.tracks = tracks
        self.calls: list[tuple[sv.Detections, np.ndarray]] = []

    def update(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
        self.calls.append((detections, frame))
        return self.tracks


class FakeInternalTrack:
    def __init__(self, tracker_id: int) -> None:
        self.tracker_id = tracker_id


class FakeLegacyInternalTrack:
    def __init__(self, track_id: int) -> None:
        self.track_id = track_id


class FakeTrackListTracker:
    def __init__(self) -> None:
        self.tracks = [
            FakeInternalTrack(1),
            FakeInternalTrack(2),
            FakeInternalTrack(-1),
        ]

    def update(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
        _ = detections, frame
        return _empty_tracks()


def _empty_tracks() -> sv.Detections:
    tracks = sv.Detections.empty()
    tracks.tracker_id = np.empty((0,), dtype=np.int32)
    return tracks


def test_botsort_adapter_calls_tracker_for_empty_detections() -> None:
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    fake_tracker = FakeTracker(_empty_tracks())
    adapter = BotSortAdapter(AppConfig(), tracker=fake_tracker)
    detections = sv.Detections(
        xyxy=np.empty((0, 4), dtype=np.float32),
        confidence=np.empty((0,), dtype=np.float32),
        class_id=np.empty((0,), dtype=np.int32),
        data={"class_name": np.empty((0,), dtype=object)},
    )

    tracked = adapter.update(detections, frame)

    sent_detections = fake_tracker.calls[0][0]
    assert sent_detections is detections
    assert len(sent_detections) == 0
    assert sent_detections.xyxy.shape == (0, 4)
    assert sent_detections.confidence is not None
    assert sent_detections.class_id is not None
    assert sent_detections.data["class_name"].tolist() == []
    assert fake_tracker.calls[0][1] is frame
    assert len(tracked) == 0
    assert tracked.tracker_id is not None


def test_botsort_adapter_passes_detections_to_tracker() -> None:
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    fake_tracker = FakeTracker(_empty_tracks())
    adapter = BotSortAdapter(AppConfig(), tracker=fake_tracker)
    detections = sv.Detections(
        xyxy=np.array([[1, 2, 11, 12]], dtype=np.float32),
        confidence=np.array([0.88], dtype=np.float32),
        class_id=np.array([2], dtype=np.int32),
        data={"class_name": np.array(["car"], dtype=object)},
    )

    adapter.update(detections, frame)

    sent_detections = fake_tracker.calls[0][0]
    assert sent_detections is detections


def test_botsort_adapter_drop_tracks_removes_matching_internal_tracklets() -> None:
    tracker = FakeTrackListTracker()
    adapter = BotSortAdapter(AppConfig(), tracker=tracker)

    adapter.drop_tracks({2})

    assert [track.tracker_id for track in tracker.tracks] == [1, -1]


def test_botsort_adapter_drop_tracks_also_supports_track_id_attribute() -> None:
    tracker = FakeTrackListTracker()
    tracker.tracks = [
        FakeLegacyInternalTrack(1),
        FakeLegacyInternalTrack(2),
        FakeLegacyInternalTrack(3),
    ]
    adapter = BotSortAdapter(AppConfig(), tracker=tracker)

    adapter.drop_tracks({2})

    assert [track.track_id for track in tracker.tracks] == [1, 3]


def test_botsort_adapter_drop_tracks_ignores_wrappers_without_track_list() -> None:
    fake_tracker = FakeTracker(_empty_tracks())
    adapter = BotSortAdapter(AppConfig(), tracker=fake_tracker)

    adapter.drop_tracks({2})

    assert fake_tracker.tracks is not None


def test_create_botsort_tracker_uses_analysis_frame_rate() -> None:
    tracker = _create_botsort_tracker(AppConfig(), frame_rate=10)

    assert isinstance(tracker, BoTSORTTracker)
    assert _effective_frame_rate(AppConfig(), frame_rate=10) == 10.0
    assert tracker.maximum_frames_without_update == 4


def test_create_botsort_tracker_allows_configured_frame_rate_override() -> None:
    config = AppConfig.model_validate({"tracker": {"frame_rate": 15}})
    tracker = _create_botsort_tracker(config, frame_rate=10)

    assert _effective_frame_rate(config, frame_rate=10) == 15.0
    assert tracker.maximum_frames_without_update == 6


def test_create_botsort_tracker_disables_cmc_by_default() -> None:
    tracker = _create_botsort_tracker(AppConfig(), frame_rate=30)

    assert tracker.enable_cmc is False
    assert tracker.cmc is None


def test_create_botsort_tracker_can_enable_cmc() -> None:
    config = AppConfig.model_validate({"tracker": {"enable_cmc": True}})
    tracker = _create_botsort_tracker(config, frame_rate=30)

    assert tracker.enable_cmc is True
    assert tracker.cmc is not None


def test_create_botsort_tracker_passes_instant_first_frame_activation() -> None:
    config = AppConfig.model_validate(
        {"tracker": {"instant_first_frame_activation": False}}
    )
    tracker = _create_botsort_tracker(config, frame_rate=30)

    assert tracker.instant_first_frame_activation is False
