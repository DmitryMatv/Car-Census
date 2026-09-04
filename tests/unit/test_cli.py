from pathlib import Path

import orjson
from typer.testing import CliRunner

from car_census_cli import app
from models import BBox, FrameRecord, RunManifest, TrackedObject, TrackSummary


def test_analyze_help_exposes_transactional_overwrite_option() -> None:
    result = CliRunner().invoke(app, ["analyze", "--help"])

    assert result.exit_code == 0
    assert "--run-dir" in result.stdout
    assert "--overwrite" in result.stdout


def test_analyze_rejects_overwrite_without_explicit_run_directory() -> None:
    result = CliRunner().invoke(app, ["analyze", "input.mp4", "--overwrite"])

    assert result.exit_code == 2
    assert "--overwrite requires --run-dir" in result.stderr


def test_analyze_rejects_camera_id_with_path_traversal() -> None:
    result = CliRunner().invoke(app, ["analyze", "input.mp4", "--camera-id", "../evil"])

    assert result.exit_code == 2
    assert "Invalid camera id" in result.stderr
    assert "--camera-id" in result.stderr


def test_roi_edit_rejects_camera_id_with_path_traversal() -> None:
    result = CliRunner().invoke(
        app, ["roi", "edit", "input.mp4", "--camera-id", "../../evil"]
    )

    assert result.exit_code == 2
    assert "Invalid camera id" in result.stderr


def _observation(track_id: int) -> TrackedObject:
    bbox = BBox(x1=10, y1=10, x2=40, y2=40)
    return TrackedObject(
        track_id=track_id,
        vehicle_index=None,
        frame_index=0,
        timestamp_seconds=0.0,
        bbox=bbox,
        confidence=0.9,
        class_id=3,
        class_name="car",
        centroid=bbox.center,
        bottom_center=bbox.bottom_center,
        inside_roi=True,
    )


def _fabricate_run(run_dir: Path, *, with_labels: bool) -> Path:
    analysis = run_dir / "analysis"
    analysis.mkdir(parents=True)
    manifest = RunManifest(
        run_id=run_dir.name,
        video_path=run_dir / "input.mp4",
        camera_id="__full_frame__",
        root_dir=run_dir,
        source_fps=30.0,
        analysis_fps=10.0,
        width=64,
        height=64,
    )
    (run_dir / "run.json").write_bytes(orjson.dumps(manifest.model_dump(mode="json")))
    record = FrameRecord(
        frame_index=0,
        timestamp_seconds=0.0,
        tracks=[_observation(0)],
    )
    (analysis / "frames.jsonl").write_bytes(
        orjson.dumps(record.model_dump(mode="json")) + b"\n"
    )
    summary = TrackSummary(
        track_id=0,
        vehicle_index=None,
        first_frame_index=0,
        last_frame_index=0,
        frames_seen=1,
        max_box_width_px=30.0,
    )
    (analysis / "tracks.jsonl").write_bytes(
        orjson.dumps(summary.model_dump(mode="json")) + b"\n"
    )
    (analysis / "detection_stats.json").write_bytes(b"{}")
    if with_labels:
        labels_dir = run_dir / "mmr"
        labels_dir.mkdir(parents=True)
        (labels_dir / "labels.json").write_bytes(b"{}")
    return run_dir


def test_link_refuses_to_relink_after_classification(tmp_path) -> None:
    run_dir = _fabricate_run(tmp_path / "run", with_labels=True)

    result = CliRunner().invoke(app, ["link", "--run-dir", str(run_dir)])

    assert result.exit_code == 1
    assert "Refusing to re-link" in result.stderr
    assert "--force" in result.stderr
    assert not (run_dir / "analysis" / "links.json").exists()


def test_link_force_relinks_after_classification_with_warning(tmp_path) -> None:
    run_dir = _fabricate_run(tmp_path / "run", with_labels=True)

    result = CliRunner().invoke(app, ["link", "--run-dir", str(run_dir), "--force"])

    assert result.exit_code == 0
    assert "labels.json exists and may become stale" in result.stdout
    assert "skipped_no_homography" in result.stdout
    assert (run_dir / "analysis" / "links.json").exists()


def test_link_runs_without_labels_guard(tmp_path) -> None:
    run_dir = _fabricate_run(tmp_path / "run", with_labels=False)

    result = CliRunner().invoke(app, ["link", "--run-dir", str(run_dir)])

    assert result.exit_code == 0
    assert "Refusing to re-link" not in result.stdout
    assert (run_dir / "analysis" / "links.json").exists()
