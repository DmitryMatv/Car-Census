from pathlib import Path

from models import BBox, CountEvent, FrameRecord, MMRResult, RunManifest, TrackedObject
from storage.run_layout import RunLayout
from storage.run_store import RunStore


def _manifest(root: Path) -> RunManifest:
    return RunManifest(
        run_id="test",
        video_path=root / "input.mp4",
        camera_id="__full_frame__",
        root_dir=root,
        source_fps=30.0,
        analysis_fps=10.0,
        width=32,
        height=32,
    )


def _track(track_id: int, vehicle_index: int | None = None) -> TrackedObject:
    bbox = BBox(x1=1, y1=2, x2=10, y2=12)
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


def test_run_layout_exposes_expected_paths(tmp_path) -> None:
    layout = RunLayout(tmp_path)

    assert layout.analysis_dir == tmp_path / "analysis"
    assert layout.crops_dir == tmp_path / "crops"
    assert layout.mmr_cache_dir == tmp_path / "mmr" / "cache"
    assert layout.mmr_batch_grids_dir == tmp_path / "mmr" / "batch_grids"
    assert layout.manifest_path == tmp_path / "run.json"
    assert layout.frames_path == tmp_path / "analysis" / "frames.jsonl"
    assert layout.render_frames_path == tmp_path / "analysis" / "render_frames.jsonl"
    assert layout.output_video_path == tmp_path / "annotated.mp4"
    assert layout.report_csv_path == tmp_path / "report.csv"


def test_run_store_repositories_round_trip_artifacts(tmp_path) -> None:
    store = RunStore(tmp_path)
    store.ensure_directories()
    manifest = _manifest(tmp_path)
    frame = FrameRecord(frame_index=1, timestamp_seconds=0.1, tracks=[_track(10)])
    event = CountEvent(
        track_id=10,
        frame_index=1,
        timestamp_seconds=0.1,
        direction="IN",
    )
    result = MMRResult(make="Toyota", model="Corolla", vehicle_index=1)

    store.manifest.write(manifest)
    store.frames.append(frame)
    with store.frames.open_writer(smoothed=True) as writer:
        writer.write(frame)
    store.counts.append(event)
    store.labels.write({10: result})
    store.detection_stats.write({"raw_candidate_rows": 3})

    assert store.manifest.read() == manifest
    assert store.frames.read_all() == [frame]
    assert store.frames.read_all(smoothed=True) == [frame]
    assert store.counts.read_all() == [event]
    assert store.labels.read() == {10: result}
    assert store.detection_stats.read() == {"raw_candidate_rows": 3}


def test_frame_repository_rewrites_vehicle_indices(tmp_path) -> None:
    store = RunStore(tmp_path)
    store.ensure_directories()
    frame = FrameRecord(
        frame_index=1,
        timestamp_seconds=0.1,
        tracks=[_track(10), _track(20), _track(30)],
    )
    store.frames.write_all([frame])

    store.frames.rewrite_vehicle_indices({20: 1, 30: 2})

    rewritten = store.frames.read_all()[0]
    assert [track.vehicle_index for track in rewritten.tracks] == [None, 1, 2]


def test_missing_optional_artifacts_read_as_empty_dicts(tmp_path) -> None:
    store = RunStore(tmp_path)
    store.ensure_directories()

    assert store.labels.read() == {}
    assert store.detection_stats.read() == {}
