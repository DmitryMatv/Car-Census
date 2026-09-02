from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from config import build_full_frame_profile
from models import RunManifest
from pipeline.run import run_pipeline
from pipeline.stages import PipelineStages
from storage import run_store as run_store_module
from storage.run_store import (
    RunStore,
    _allocate_run_root,
    _compact_utc_timestamp,
    _run_descriptor,
)


def _write_analysis_artifacts(run_store: RunStore, video_path: Path) -> None:
    run_store.manifest.write(
        RunManifest(
            run_id=run_store.root.name,
            video_path=video_path,
            root_dir=run_store.root,
            source_fps=30.0,
            analysis_fps=10.0,
            width=16,
            height=16,
            frame_count=2,
        )
    )
    run_store.frames.write_all([])
    run_store.tracks.write_all([])
    run_store.detection_stats.write({})


def test_run_store_creates_expected_artifact_directories(tmp_path) -> None:
    store = RunStore.create(
        output_root=tmp_path,
        camera_id="test-camera",
        video_stem="video",
    )

    assert store.root.exists()
    assert store.analysis_dir.is_dir()
    assert store.crops_dir.is_dir()
    assert store.mmr_dir.is_dir()
    assert not store.mmr_cache_dir.exists()
    assert not store.mmr_batch_grids_dir.exists()
    assert not (store.root / "render").exists()


@pytest.mark.parametrize(
    ("camera_id", "video_stem", "expected"),
    [
        ("__full_frame__", "IMG_5386_1440_20s", "IMG_5386_1440_20s"),
        ("IMG_5581_1440", "IMG_5581_1440", "IMG_5581_1440"),
        (
            "IMG_5458_1440",
            "IMG_5383_1440_20s",
            "IMG_5383_1440_20s--camera-IMG_5458_1440",
        ),
    ],
)
def test_run_descriptor_omits_redundant_camera_names(
    camera_id: str, video_stem: str, expected: str
) -> None:
    assert _run_descriptor(camera_id, video_stem) == expected


@pytest.mark.parametrize(
    "camera_id",
    ["../escape", "..", ".", "", "a/b", "a\\b"],
)
def test_run_descriptor_rejects_unsafe_camera_ids(
    camera_id: str, video_stem: str = "video"
) -> None:
    with pytest.raises(ValueError, match="Invalid camera id"):
        _run_descriptor(camera_id, video_stem)


def test_run_store_create_rejects_traversal_camera_id(tmp_path) -> None:
    with pytest.raises(ValueError, match="Invalid camera id"):
        RunStore.create(
            output_root=tmp_path,
            camera_id="../../outside",
            video_stem="video",
        )
    assert not (tmp_path / "outside").exists()


def test_compact_timestamp_converts_to_utc() -> None:
    riga_summer_time = timezone(timedelta(hours=3))

    assert (
        _compact_utc_timestamp(
            datetime(2026, 6, 11, 10, 9, 23, tzinfo=riga_summer_time)
        )
        == "20260611-070923Z"
    )


def test_run_store_uses_readable_suffixes_for_same_second_collisions(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        run_store_module,
        "_compact_utc_timestamp",
        lambda: "20260611-070923Z",
    )
    first = RunStore.create(
        output_root=tmp_path,
        camera_id="test-camera",
        video_stem="video",
    )
    second = RunStore.create(
        output_root=tmp_path,
        camera_id="test-camera",
        video_stem="video",
    )

    assert first.root.name == "video--camera-test-camera--20260611-070923Z"
    assert second.root.name == "video--camera-test-camera--20260611-070923Z--02"


def test_run_root_collision_suffix_continues_beyond_99(tmp_path: Path) -> None:
    base_run_id = "video--20260611-070923Z"
    for collision_number in range(1, 100):
        run_id = (
            base_run_id
            if collision_number == 1
            else f"{base_run_id}--{collision_number:02d}"
        )
        (tmp_path / run_id).mkdir()

    allocated = _allocate_run_root(tmp_path, base_run_id)

    assert allocated.name == "video--20260611-070923Z--100"


def test_creating_readable_run_does_not_rename_existing_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = tmp_path / "full-frame-video-20260101T010101Z"
    existing.mkdir()
    monkeypatch.setattr(
        run_store_module,
        "_compact_utc_timestamp",
        lambda: "20260611-070923Z",
    )

    created = RunStore.create(
        output_root=tmp_path,
        camera_id="__full_frame__",
        video_stem="video",
    )

    assert existing.is_dir()
    assert created.root.name == "video--20260611-070923Z"


def test_run_pipeline_orders_analyze_classify_render_report(
    default_config, tmp_path
) -> None:
    calls: list[str] = []
    render_kwargs: dict[str, object] = {}
    video_path = tmp_path / "input.mp4"

    def fake_analyze_video(project_root, config, profile, analyzed_video, run_store):
        _ = project_root, config, profile
        calls.append("analyze")
        assert analyzed_video == video_path
        assert run_store.root.name.startswith(".")
        _write_analysis_artifacts(run_store, video_path)
        return run_store

    def fake_classify_tracks(config, run_store, profile):
        _ = config, profile
        calls.append("classify")
        assert isinstance(run_store, RunStore)
        return {}

    def fake_render_video(
        config, profile, rendered_video, run_store, allow_unclassified_annotations
    ):
        _ = config, profile, rendered_video
        calls.append("render")
        render_kwargs["run_store_root"] = run_store.root
        render_kwargs["allow_unclassified_annotations"] = allow_unclassified_annotations
        return run_store.root / "annotated.mp4"

    def fake_generate_reports(run_store):
        calls.append("report")
        assert isinstance(run_store, RunStore)
        return {}

    def fake_write_skipped_classification_batch_grids(config, run_store):
        _ = config, run_store
        raise AssertionError("Skipped classification stage should not run")

    result = run_pipeline(
        project_root=tmp_path,
        config=default_config,
        profile=build_full_frame_profile(width=16, height=16),
        video_path=video_path,
        stages=PipelineStages(
            analyze_video=fake_analyze_video,
            classify_tracks=fake_classify_tracks,
            write_skipped_classification_batch_grids=(
                fake_write_skipped_classification_batch_grids
            ),
            render_video=fake_render_video,
            generate_reports=fake_generate_reports,
        ),
        skip_classification=False,
    )

    assert calls == ["analyze", "classify", "render", "report"]
    output_root = tmp_path / "output"
    assert result.root.parent == output_root
    assert not result.root.name.startswith(".")
    assert result.manifest.read().run_id == result.root.name
    assert result.frames_path.is_file()
    assert [path.name for path in output_root.iterdir()] == [result.root.name]
    assert render_kwargs["allow_unclassified_annotations"] is False
    assert render_kwargs["run_store_root"] != result.root


def test_run_pipeline_allows_unclassified_annotations_when_classification_is_skipped(
    default_config, tmp_path
) -> None:
    calls: list[str] = []
    render_kwargs: dict[str, object] = {}
    video_path = tmp_path / "input.mp4"

    def fake_analyze_video(project_root, config, profile, analyzed_video, run_store):
        _ = project_root, config, profile
        calls.append("analyze")
        _write_analysis_artifacts(run_store, video_path)
        return run_store

    def fake_classify_tracks(config, run_store, profile):
        _ = config, run_store, profile
        raise AssertionError("Classification stage should not run")

    def fake_write_skipped_classification_batch_grids(config, run_store):
        _ = config
        calls.append("batch_grids")
        assert isinstance(run_store, RunStore)
        return []

    def fake_render_video(
        config, profile, rendered_video, run_store, allow_unclassified_annotations
    ):
        _ = config, profile, rendered_video
        calls.append("render")
        render_kwargs["allow_unclassified_annotations"] = allow_unclassified_annotations
        return run_store.root / "annotated.mp4"

    def fake_generate_reports(run_store):
        calls.append("report")
        assert isinstance(run_store, RunStore)
        return {}

    result = run_pipeline(
        project_root=tmp_path,
        config=default_config,
        profile=build_full_frame_profile(width=16, height=16),
        video_path=video_path,
        stages=PipelineStages(
            analyze_video=fake_analyze_video,
            classify_tracks=fake_classify_tracks,
            write_skipped_classification_batch_grids=(
                fake_write_skipped_classification_batch_grids
            ),
            render_video=fake_render_video,
            generate_reports=fake_generate_reports,
        ),
        skip_classification=True,
    )

    assert calls == ["analyze", "batch_grids", "render", "report"]
    assert render_kwargs["allow_unclassified_annotations"] is True
    assert not (tmp_path / "output" / result.root.name / "mmr" / "labels.json").exists()


def test_run_pipeline_can_skip_render_while_still_generating_report(
    default_config, tmp_path
) -> None:
    calls: list[str] = []
    video_path = tmp_path / "input.mp4"

    def fake_analyze_video(project_root, config, profile, analyzed_video, run_store):
        _ = project_root, config, profile
        calls.append("analyze")
        _write_analysis_artifacts(run_store, video_path)
        return run_store

    def fake_classify_tracks(config, run_store, profile):
        _ = config, profile
        calls.append("classify")
        assert isinstance(run_store, RunStore)
        return {}

    def fake_write_skipped_classification_batch_grids(config, run_store):
        _ = config, run_store
        raise AssertionError("Skipped classification stage should not run")

    def fake_render_video(
        config, profile, rendered_video, run_store, allow_unclassified_annotations
    ):
        _ = config, profile, rendered_video, run_store, allow_unclassified_annotations
        raise AssertionError("Render stage should not run")

    def fake_generate_reports(run_store):
        calls.append("report")
        assert isinstance(run_store, RunStore)
        return {}

    result = run_pipeline(
        project_root=tmp_path,
        config=default_config,
        profile=build_full_frame_profile(width=16, height=16),
        video_path=video_path,
        stages=PipelineStages(
            analyze_video=fake_analyze_video,
            classify_tracks=fake_classify_tracks,
            write_skipped_classification_batch_grids=(
                fake_write_skipped_classification_batch_grids
            ),
            render_video=fake_render_video,
            generate_reports=fake_generate_reports,
        ),
        skip_classification=False,
        skip_render=True,
    )

    assert calls == ["analyze", "classify", "report"]
    assert result.frames_path.is_file()


def test_run_pipeline_failure_leaves_no_partial_run_directory(
    default_config, tmp_path
) -> None:
    video_path = tmp_path / "input.mp4"

    def fake_analyze_video(project_root, config, profile, analyzed_video, run_store):
        _ = project_root, config, profile
        _write_analysis_artifacts(run_store, video_path)
        return run_store

    def fake_classify_tracks(config, run_store, profile):
        _ = config, run_store, profile
        return {}

    def fake_write_skipped_classification_batch_grids(config, run_store):
        _ = config, run_store
        return []

    def fake_render_video(
        config, profile, rendered_video, run_store, allow_unclassified_annotations
    ):
        _ = config, profile, rendered_video, run_store, allow_unclassified_annotations
        raise RuntimeError("render failed")

    def fake_generate_reports(run_store):
        _ = run_store
        raise AssertionError("Report stage should not run")

    with pytest.raises(RuntimeError, match="render failed"):
        run_pipeline(
            project_root=tmp_path,
            config=default_config,
            profile=build_full_frame_profile(width=16, height=16),
            video_path=video_path,
            stages=PipelineStages(
                analyze_video=fake_analyze_video,
                classify_tracks=fake_classify_tracks,
                write_skipped_classification_batch_grids=(
                    fake_write_skipped_classification_batch_grids
                ),
                render_video=fake_render_video,
                generate_reports=fake_generate_reports,
            ),
        )

    output_root = tmp_path / "output"
    assert list(output_root.iterdir()) == []
