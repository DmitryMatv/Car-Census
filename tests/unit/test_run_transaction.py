from __future__ import annotations

from pathlib import Path

import pytest

from models import CropCandidate, RunManifest, TrackSummary
from storage.run_store import RunStore
from storage.run_transaction import AnalysisRunTransaction, RunDirectoryError


def _seed_analysis(
    root: Path,
    *,
    video_path: Path,
    camera_id: str = "camera",
    with_candidate: bool = False,
) -> RunStore:
    store = RunStore(root)
    store.ensure_directories()
    store.manifest.write(
        RunManifest(
            run_id=root.name,
            video_path=video_path,
            camera_id=camera_id,
            root_dir=root,
            source_fps=30.0,
            analysis_fps=10.0,
            width=1920,
            height=1080,
            frame_count=100,
        )
    )
    store.frames.write_all([])
    candidates: list[CropCandidate] = []
    if with_candidate:
        crop_path = store.crops_dir / "vehicle_000001.jpg"
        crop_path.write_bytes(b"new crop")
        candidates.append(
            CropCandidate(
                track_id=7,
                vehicle_index=1,
                frame_index=10,
                timestamp_seconds=1.0,
                bbox={"x1": 1, "y1": 2, "x2": 10, "y2": 12},
                image_path=crop_path,
                sharpness=1.0,
                edge_margin_score=1.0,
                area_score=1.0,
            )
        )
    store.tracks.write_all(
        [
            TrackSummary(
                track_id=7,
                vehicle_index=1 if candidates else None,
                first_frame_index=0,
                last_frame_index=10,
                frames_seen=11,
                max_box_width_px=100,
                candidates=candidates,
            )
        ]
    )
    store.detection_stats.write({"complete": True})
    return store


def _transaction(
    target: Path,
    video_path: Path,
    *,
    overwrite: bool = False,
    camera_id: str = "camera",
) -> AnalysisRunTransaction:
    return AnalysisRunTransaction(
        run_dir=target,
        overwrite=overwrite,
        video_path=video_path,
        camera_id=camera_id,
    )


def test_transaction_creates_missing_explicit_run_directory(tmp_path: Path) -> None:
    target = tmp_path / "explicit-run"
    video_path = tmp_path / "input.mp4"

    with _transaction(target, video_path) as staging_store:
        _seed_analysis(
            staging_store.root,
            video_path=video_path,
            with_candidate=True,
        )

    promoted = RunStore.from_existing(target)
    summary = promoted.tracks.read_all()[0]
    assert promoted.manifest.read().run_id == target.name
    assert promoted.manifest.read().root_dir == target.resolve()
    assert summary.candidates[0].image_path == (
        target.resolve() / "crops" / "vehicle_000001.jpg"
    )
    assert summary.candidates[0].image_path.read_bytes() == b"new crop"


def test_transaction_accepts_empty_explicit_run_directory(tmp_path: Path) -> None:
    target = tmp_path / "empty-run"
    target.mkdir()
    video_path = tmp_path / "input.mp4"

    with _transaction(target, video_path) as staging_store:
        _seed_analysis(staging_store.root, video_path=video_path)

    assert RunStore.from_existing(target).detection_stats.read() == {"complete": True}


def test_transaction_refuses_completed_run_without_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "existing-run"
    video_path = tmp_path / "input.mp4"
    _seed_analysis(target, video_path=video_path)

    with pytest.raises(RunDirectoryError, match="--overwrite"):
        with _transaction(target, video_path):
            pass


def test_transaction_refuses_invalid_nonempty_directory_even_with_overwrite(
    tmp_path: Path,
) -> None:
    target = tmp_path / "not-a-run"
    target.mkdir()
    (target / "unrelated.txt").write_text("keep me", encoding="utf-8")

    with pytest.raises(RunDirectoryError, match="not a valid Car-Census run"):
        with _transaction(target, tmp_path / "input.mp4", overwrite=True):
            pass

    assert (target / "unrelated.txt").read_text(encoding="utf-8") == "keep me"


@pytest.mark.parametrize(
    ("video_name", "camera_id", "match"),
    [
        ("different.mp4", "camera", "different video"),
        ("input.mp4", "different-camera", "different camera profile"),
    ],
)
def test_transaction_refuses_source_or_camera_mismatch(
    tmp_path: Path,
    video_name: str,
    camera_id: str,
    match: str,
) -> None:
    target = tmp_path / "existing-run"
    original_video = tmp_path / "input.mp4"
    _seed_analysis(target, video_path=original_video)

    with pytest.raises(RunDirectoryError, match=match):
        with _transaction(
            target,
            tmp_path / video_name,
            overwrite=True,
            camera_id=camera_id,
        ):
            pass


def test_analysis_failure_leaves_existing_run_untouched(tmp_path: Path) -> None:
    target = tmp_path / "existing-run"
    video_path = tmp_path / "input.mp4"
    _seed_analysis(target, video_path=video_path)
    sentinel = target / "annotated.mp4"
    sentinel.write_bytes(b"old video")
    before = {
        path.relative_to(target): path.read_bytes()
        for path in target.rglob("*")
        if path.is_file()
    }

    with pytest.raises(RuntimeError, match="analysis failed"):
        with _transaction(target, video_path, overwrite=True):
            raise RuntimeError("analysis failed")

    after = {
        path.relative_to(target): path.read_bytes()
        for path in target.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_successful_overwrite_discards_all_stale_downstream_artifacts(
    tmp_path: Path,
) -> None:
    target = tmp_path / "existing-run"
    video_path = tmp_path / "input.mp4"
    old_store = _seed_analysis(target, video_path=video_path)
    old_store.labels.write({})
    old_store.render_frames_path.write_bytes(b"old smoothing")
    old_store.output_video_path.write_bytes(b"old video")
    old_store.report_csv_path.write_bytes(b"old report")

    with _transaction(target, video_path, overwrite=True) as staging_store:
        _seed_analysis(
            staging_store.root,
            video_path=video_path,
            with_candidate=True,
        )

    promoted = RunStore.from_existing(target)
    assert promoted.detection_stats.read() == {"complete": True}
    assert not promoted.labels_path.exists()
    assert not promoted.render_frames_path.exists()
    assert not promoted.output_video_path.exists()
    assert not promoted.report_csv_path.exists()


def test_promotion_failure_restores_previous_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = (tmp_path / "existing-run").resolve()
    video_path = tmp_path / "input.mp4"
    _seed_analysis(target, video_path=video_path)
    sentinel = target / "annotated.mp4"
    sentinel.write_bytes(b"old video")
    original_rename = Path.rename

    def fail_staging_promotion(path: Path, destination: Path) -> Path:
        if ".staging-" in path.name and Path(destination) == target:
            raise OSError("simulated promotion failure")
        return original_rename(path, destination)

    monkeypatch.setattr(Path, "rename", fail_staging_promotion)

    with pytest.raises(OSError, match="simulated promotion failure"):
        with _transaction(target, video_path, overwrite=True) as staging_store:
            _seed_analysis(staging_store.root, video_path=video_path)

    assert RunStore.from_existing(target).manifest.read().video_path == video_path
    assert sentinel.read_bytes() == b"old video"


def test_from_existing_rejects_missing_and_manifestless_directories(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        RunStore.from_existing(tmp_path / "missing")

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="manifest does not exist"):
        RunStore.from_existing(empty)
