from typing import Any

import numpy as np
import supervision as sv
from config import AppConfig
from trackers import BoTSORTTracker
from tracking_adapters.botsort import (
    BotSortAdapter,
    _create_botsort_tracker,
    _effective_frame_rate,
    _track_identity,
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


def test_botsort_adapter_calls_tracker_for_empty_detections(default_config) -> None:
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    fake_tracker = FakeTracker(_empty_tracks())
    adapter = BotSortAdapter(default_config, tracker=fake_tracker)
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


def test_botsort_adapter_passes_detections_to_tracker(default_config) -> None:
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    fake_tracker = FakeTracker(_empty_tracks())
    adapter = BotSortAdapter(default_config, tracker=fake_tracker)
    detections = sv.Detections(
        xyxy=np.array([[1, 2, 11, 12]], dtype=np.float32),
        confidence=np.array([0.88], dtype=np.float32),
        class_id=np.array([2], dtype=np.int32),
        data={"class_name": np.array(["car"], dtype=object)},
    )

    adapter.update(detections, frame)

    sent_detections = fake_tracker.calls[0][0]
    assert sent_detections is detections


def test_botsort_adapter_drop_tracks_removes_matching_internal_tracklets(
    default_config,
) -> None:
    tracker = FakeTrackListTracker()
    adapter = BotSortAdapter(default_config, tracker=tracker)

    adapter.drop_tracks({2})

    assert [track.tracker_id for track in tracker.tracks] == [1, -1]


def test_botsort_adapter_drop_tracks_also_supports_track_id_attribute(
    default_config,
) -> None:
    tracker = FakeTrackListTracker()
    tracker.tracks = [
        FakeLegacyInternalTrack(1),
        FakeLegacyInternalTrack(2),
        FakeLegacyInternalTrack(3),
    ]
    adapter = BotSortAdapter(default_config, tracker=tracker)

    adapter.drop_tracks({2})

    assert [track.track_id for track in tracker.tracks] == [1, 3]


def test_botsort_adapter_drop_tracks_ignores_wrappers_without_track_list(
    default_config,
) -> None:
    fake_tracker = FakeTracker(_empty_tracks())
    adapter = BotSortAdapter(default_config, tracker=fake_tracker)

    adapter.drop_tracks({2})

    assert fake_tracker.tracks is not None


def test_create_botsort_tracker_uses_analysis_frame_rate(default_config) -> None:
    analysis_fps = default_config.analysis.fps
    tracker = _create_botsort_tracker(default_config, frame_rate=analysis_fps)

    assert isinstance(tracker, BoTSORTTracker)
    assert _effective_frame_rate(default_config, frame_rate=analysis_fps) == 10.0
    assert tracker.maximum_frames_without_update == 20


def test_create_botsort_tracker_allows_configured_frame_rate_override(
    config_factory,
) -> None:
    config = config_factory({"tracker": {"frame_rate": 15}})
    tracker = _create_botsort_tracker(config, frame_rate=10)

    assert _effective_frame_rate(config, frame_rate=10) == 15.0
    assert tracker.maximum_frames_without_update == 30


def test_create_botsort_tracker_disables_cmc_by_default(default_config) -> None:
    tracker = _create_botsort_tracker(default_config, frame_rate=30)

    assert tracker.enable_cmc is False
    assert tracker.cmc is None


def test_create_botsort_tracker_can_enable_cmc(config_factory) -> None:
    config = config_factory({"tracker": {"enable_cmc": True}})
    tracker = _create_botsort_tracker(config, frame_rate=30)

    assert tracker.enable_cmc is True
    assert tracker.cmc is not None


def test_create_botsort_tracker_passes_instant_first_frame_activation(
    config_factory,
) -> None:
    config = config_factory({"tracker": {"instant_first_frame_activation": False}})
    tracker = _create_botsort_tracker(config, frame_rate=30)

    assert tracker.instant_first_frame_activation is False


class _RescueStubTracker:
    """Stub exposing a BoT-SORT-like internal tracklet list."""

    def __init__(self, tracks: list, output: sv.Detections | None = None) -> None:
        self.tracks = tracks
        self.output = output

    def update(self, detections, frame, timestamp=None):
        _ = detections, frame, timestamp
        if self.output is not None:
            return self.output
        return _empty_tracks()


def _world_transformer() -> Any:
    from roi.transform import ViewTransformer

    return ViewTransformer(
        source_points=[[0, 0], [100, 0], [100, 100], [0, 100]],
        target_points=[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
    )


def _old_tracklet(track_id: int) -> Any:
    from trackers.core.botsort.tracklet import BoTSORTTracklet

    tracklet = BoTSORTTracklet(initial_bbox=np.array([60.0, 20.0, 100.0, 50.0]))
    tracklet.tracker_id = track_id
    tracklet.time_since_update = 3
    tracklet.number_of_successful_updates = 10
    return tracklet


def _fresh_spawn_tracklet() -> Any:
    from trackers.core.botsort.tracklet import BoTSORTTracklet

    return BoTSORTTracklet(initial_bbox=np.array([60.0, 20.0, 100.0, 50.0]))


def _spawn_output() -> sv.Detections:
    return sv.Detections(
        xyxy=np.array([[60, 20, 100, 50]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([2], dtype=np.int32),
        tracker_id=np.array([-1], dtype=np.int32),
        data={"class_name": np.array(["car"], dtype=object)},
    )


def _seed_trajectory(adapter: BotSortAdapter, track_id: int) -> None:
    # World velocity: 10 m/s along +x (pixel scale: 10 px = 1 m).
    adapter._rescue.observe(track_id, 0.0, (50.0, 50.0))
    adapter._rescue.observe(track_id, 0.1, (60.0, 50.0))


def test_adapter_rescue_takes_over_fresh_spawn(default_config) -> None:
    old = _old_tracklet(5)
    spawn = _fresh_spawn_tracklet()
    stub = _RescueStubTracker([old, spawn])
    adapter = BotSortAdapter(
        default_config, tracker=stub, view_transformer=_world_transformer()
    )
    _seed_trajectory(adapter, 5)

    adapter.update(
        _spawn_output(), np.zeros((10, 20, 3), dtype=np.uint8), timestamp=0.2
    )

    assert spawn.tracker_id == 5
    assert all(_track_identity(track) != 5 or track is spawn for track in stub.tracks)
    accepted = [e for e in adapter._events if e["outcome"] == "accepted"]
    assert len(accepted) == 1
    assert accepted[0]["old_track_id"] == 5


def test_adapter_rescue_rejects_implausible_handoff(default_config) -> None:
    # The fresh spawn sits far from the old track's predicted position:
    # bottom-center (400, 50) px = (40, 5) m, i.e. 340 m/s implied — rejected.
    from trackers.core.botsort.tracklet import BoTSORTTracklet

    spawn = BoTSORTTracklet(initial_bbox=np.array([380.0, 20.0, 420.0, 50.0]))
    stub = _RescueStubTracker([spawn])
    adapter = BotSortAdapter(
        default_config, tracker=stub, view_transformer=_world_transformer()
    )
    _seed_trajectory(adapter, 5)
    # The old tracklet itself is gone; only its trajectory memory remains.

    far = sv.Detections(
        xyxy=np.array([[380, 20, 420, 50]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([2], dtype=np.int32),
        tracker_id=np.array([-1], dtype=np.int32),
        data={"class_name": np.array(["car"], dtype=object)},
    )
    adapter.update(far, np.zeros((10, 20, 3), dtype=np.uint8), timestamp=0.2)

    assert spawn.tracker_id == -1
    rejected = [e for e in adapter._events if e["outcome"] == "rejected"]
    assert len(rejected) == 1
    assert rejected[0]["old_track_id"] == 5


def test_adapter_rescue_inert_without_transformer(default_config) -> None:
    spawn = _fresh_spawn_tracklet()
    stub = _RescueStubTracker([spawn])
    adapter = BotSortAdapter(default_config, tracker=stub, view_transformer=None)
    _seed_trajectory(adapter, 5)

    adapter.update(
        _spawn_output(), np.zeros((10, 20, 3), dtype=np.uint8), timestamp=0.2
    )

    assert spawn.tracker_id == -1
    assert adapter._events == []


def test_adapter_rescue_skips_ids_matched_this_frame(default_config) -> None:
    old = _old_tracklet(5)
    spawn = _fresh_spawn_tracklet()
    busy_output = sv.Detections(
        xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([2], dtype=np.int32),
        tracker_id=np.array([5], dtype=np.int32),
        data={"class_name": np.array(["car"], dtype=object)},
    )
    # The stub returns the busy output so track 5 counts as matched.
    stub = _RescueStubTracker([old, spawn], output=busy_output)
    adapter = BotSortAdapter(
        default_config, tracker=stub, view_transformer=_world_transformer()
    )
    _seed_trajectory(adapter, 5)

    adapter.update(busy_output, np.zeros((10, 20, 3), dtype=np.uint8), timestamp=0.2)

    assert spawn.tracker_id == -1
    assert old in stub.tracks
    assert adapter._events == []
