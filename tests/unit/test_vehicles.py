import statistics
from pathlib import Path

import numpy as np
import pytest

from config import AppConfig
from models import BBox, FrameRecord, TrackedObject
from pipeline.analysis_crops import save_candidate
from pipeline.analysis_tracking import MutableTrackState
from pipeline.vehicles import (
    compute_track_world_speeds,
    discard_track_artifacts,
    finalize_vehicle_identities,
    staged_track_crop_dir,
    track_summary_from_state,
    vehicle_crop_path,
)
from roi.transform import ViewTransformer
from storage.run_store import RunStore


class DummyRunStore:
    def __init__(self, root: Path) -> None:
        self.crops_dir = root / "crops"
        self.crops_dir.mkdir(parents=True)


def _track(track_id: int, vehicle_index: int | None = None) -> TrackedObject:
    bbox = BBox(x1=1, y1=2, x2=11, y2=12)
    return TrackedObject(
        track_id=track_id,
        vehicle_index=vehicle_index,
        frame_index=1,
        timestamp_seconds=0.1,
        bbox=bbox,
        confidence=0.9,
        class_id=2,
        class_name="car",
        centroid=bbox.center,
        bottom_center=bbox.bottom_center,
        inside_roi=True,
    )


def test_vehicle_crop_path_helpers(tmp_path) -> None:
    crops_dir = tmp_path / "crops"

    assert staged_track_crop_dir(crops_dir, 42) == (
        crops_dir / ".track_candidates" / "track_000042"
    )
    assert vehicle_crop_path(crops_dir, 13, Path("frame.jpg")) == (
        crops_dir / "vehicle_000013.jpg"
    )


def test_finalize_vehicle_identities_compacts_crop_eligible_tracks(default_config, tmp_path) -> None:
    store = DummyRunStore(tmp_path)
    no_crop_state = MutableTrackState(
        track_id=10,
        first_frame_index=0,
        last_frame_index=5,
    )
    later_state = MutableTrackState(
        track_id=30,
        first_frame_index=20,
        last_frame_index=30,
    )
    earlier_state = MutableTrackState(
        track_id=20,
        first_frame_index=10,
        last_frame_index=15,
    )
    frame = np.full((20, 20, 3), 255, dtype=np.uint8)
    for state, frame_index in [(later_state, 20), (earlier_state, 10)]:
        save_candidate(
            store=store,
            track_state=state,
            frame=frame,
            bbox=BBox(x1=2, y1=3, x2=12, y2=13),
            frame_index=frame_index,
            timestamp_seconds=frame_index / 10.0,
            config=default_config,
        )

    vehicle_index_by_track = finalize_vehicle_identities(
        store, [no_crop_state, later_state, earlier_state]
    )

    assert vehicle_index_by_track == {20: 1, 30: 2}
    assert no_crop_state.vehicle_index is None
    assert earlier_state.vehicle_index == 1
    assert later_state.vehicle_index == 2
    assert earlier_state.candidates[0].vehicle_index == 1
    assert later_state.candidates[0].vehicle_index == 2
    assert earlier_state.candidates[0].image_path == store.crops_dir / (
        "vehicle_000001.jpg"
    )
    assert later_state.candidates[0].image_path == store.crops_dir / (
        "vehicle_000002.jpg"
    )
    assert sorted(path.name for path in store.crops_dir.iterdir()) == [
        "vehicle_000001.jpg",
        "vehicle_000002.jpg",
    ]
    assert (store.crops_dir / "vehicle_000001.jpg").is_file()
    assert (store.crops_dir / "vehicle_000002.jpg").is_file()


def test_rewrite_frame_vehicle_indices_marks_only_crop_eligible_tracks(
    tmp_path,
) -> None:
    store = RunStore(tmp_path)
    store.ensure_directories()
    record = FrameRecord(
        frame_index=1,
        timestamp_seconds=0.1,
        tracks=[_track(10), _track(20), _track(30)],
    )
    store.frames.write_all([record])

    store.frames.rewrite_vehicle_indices({20: 1, 30: 2})

    rewritten = store.frames.read_all()[0]
    assert [track.vehicle_index for track in rewritten.tracks] == [None, 1, 2]


def test_rewrite_frame_vehicle_indices_preserves_existing_unmapped_indices(
    tmp_path,
) -> None:
    store = RunStore(tmp_path)
    store.ensure_directories()
    record = FrameRecord(
        frame_index=1,
        timestamp_seconds=0.1,
        tracks=[_track(10, vehicle_index=7), _track(20)],
    )
    store.frames.write_all([record])

    store.frames.rewrite_vehicle_indices({20: 1})

    rewritten = store.frames.read_all()[0]
    assert [track.vehicle_index for track in rewritten.tracks] == [7, 1]


def test_discard_track_artifacts_removes_empty_temp_crop_dir(default_config, tmp_path) -> None:
    store = DummyRunStore(tmp_path)
    state = MutableTrackState(
        track_id=42,
        first_frame_index=1,
        last_frame_index=1,
    )
    frame = np.full((20, 20, 3), 255, dtype=np.uint8)

    save_candidate(
        store=store,
        track_state=state,
        frame=frame,
        bbox=BBox(x1=2, y1=3, x2=12, y2=13),
        frame_index=5,
        timestamp_seconds=0.5,
        config=default_config,
    )

    discard_track_artifacts(state, store.crops_dir)

    assert not staged_track_crop_dir(store.crops_dir, 42).exists()


def test_track_summary_from_state_preserves_vehicle_index(tmp_path) -> None:
    store = DummyRunStore(tmp_path)
    state = MutableTrackState(
        track_id=42,
        vehicle_index=7,
        first_frame_index=1,
        last_frame_index=3,
        frames_seen=3,
        min_box_width_px=40,
        max_box_width_px=100,
        counted=True,
    )

    summary = track_summary_from_state(state)

    assert summary.track_id == 42
    assert summary.vehicle_index == 7
    assert summary.first_frame_index == 1
    assert summary.last_frame_index == 3
    assert summary.frames_seen == 3
    assert summary.min_box_width_px == 40
    assert summary.max_box_width_px == 100
    assert summary.counted is True
    assert summary.candidates == []
    assert store.crops_dir.exists()


def _moving_track(
    track_id: int,
    timestamp_seconds: float,
    bottom_center: tuple[float, float],
) -> TrackedObject:
    x, y = bottom_center
    bbox = BBox(x1=x - 10, y1=y - 20, x2=x + 10, y2=y)
    return TrackedObject(
        track_id=track_id,
        vehicle_index=None,
        frame_index=int(round(timestamp_seconds * 10)),
        timestamp_seconds=timestamp_seconds,
        bbox=bbox,
        confidence=0.9,
        class_id=2,
        class_name="car",
        centroid=bbox.center,
        bottom_center=bbox.bottom_center,
        inside_roi=True,
    )


def test_compute_track_world_speeds_median_and_max() -> None:
    source_points = [[0, 0], [2000, 0], [2000, 1200], [0, 1200]]
    target_points = [[0.0, 0.0], [200.0, 0.0], [200.0, 120.0], [0.0, 120.0]]
    transformer = ViewTransformer(source_points, target_points)
    records = [
        FrameRecord(
            frame_index=index,
            timestamp_seconds=timestamp,
            tracks=[_moving_track(7, timestamp, position)],
        )
        for index, (timestamp, position) in enumerate(
            [(0.0, (100.0, 100.0)), (0.5, (160.0, 100.0)), (1.0, (300.0, 100.0))]
        )
    ]

    speeds = compute_track_world_speeds(records, transformer)

    assert speeds[7][0] == pytest.approx(statistics.median([12.0, 28.0]))
    assert speeds[7][1] == pytest.approx(28.0)


def test_compute_track_world_speeds_without_transformer_returns_empty() -> None:
    records = [
        FrameRecord(
            frame_index=0,
            timestamp_seconds=0.0,
            tracks=[_moving_track(7, 0.0, (100.0, 100.0))],
        )
    ]

    assert compute_track_world_speeds(records, None) == {}


def test_compute_track_world_speeds_skips_non_positive_elapsed_time() -> None:
    source_points = [[0, 0], [2000, 0], [2000, 1200], [0, 1200]]
    target_points = [[0.0, 0.0], [200.0, 0.0], [200.0, 120.0], [0.0, 120.0]]
    transformer = ViewTransformer(source_points, target_points)
    record = FrameRecord(
        frame_index=0,
        timestamp_seconds=1.0,
        tracks=[
            _moving_track(7, 1.0, (100.0, 100.0)),
            _moving_track(7, 1.0, (500.0, 100.0)),
        ],
    )

    assert compute_track_world_speeds([record], transformer) == {}
