from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from config import AppConfig, build_full_frame_profile
from pipeline import run as run_module
from pipeline.run import run_pipeline
from pipeline.stages import PipelineStages
from storage import run_store as run_store_module
from storage.run_store import (
    RunStore,
    _allocate_run_root,
    _compact_utc_timestamp,
    _run_descriptor,
)


class DummyRunStore:
    root = Path("output/test-run")


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
    default_config, tmp_path, monkeypatch
) -> None:
    calls: list[str] = []
    render_kwargs = {}
    store = DummyRunStore()

    monkeypatch.setattr(
        run_module.RunStore,
        "create",
        lambda **kwargs: store,
    )

    def fake_analyze_video(project_root, config, profile, video_path, run_store):
        _ = project_root, config, profile, video_path
        calls.append("analyze")
        assert run_store is store
        return store

    def fake_classify_tracks(config, run_store):
        _ = config
        calls.append("classify")
        assert run_store is store
        return {}

    def fake_render_video(
        config, profile, video_path, run_store, allow_unclassified_annotations
    ):
        _ = config, profile, video_path
        calls.append("render")
        render_kwargs["run_store"] = run_store
        render_kwargs["allow_unclassified_annotations"] = allow_unclassified_annotations
        return store.root / "annotated.mp4"

    def fake_generate_reports(run_store):
        calls.append("report")
        assert run_store is store
        return {}

    def fake_write_skipped_classification_batch_grids(config, run_store):
        _ = config, run_store
        raise AssertionError("Skipped classification stage should not run")

    result = run_pipeline(
        project_root=tmp_path,
        config=default_config,
        profile=build_full_frame_profile(width=16, height=16),
        video_path=tmp_path / "input.mp4",
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

    assert result is store
    assert calls == ["analyze", "classify", "render", "report"]
    assert render_kwargs["run_store"] is store
    assert render_kwargs["allow_unclassified_annotations"] is False


def test_run_pipeline_allows_unclassified_annotations_when_classification_is_skipped(
    default_config, tmp_path, monkeypatch
) -> None:
    calls: list[str] = []
    render_kwargs = {}
    store = DummyRunStore()

    monkeypatch.setattr(
        run_module.RunStore,
        "create",
        lambda **kwargs: store,
    )

    def fake_analyze_video(project_root, config, profile, video_path, run_store):
        _ = project_root, config, profile, video_path
        calls.append("analyze")
        assert run_store is store
        return store

    def fake_classify_tracks(config, run_store):
        _ = config, run_store
        calls.append("classify")
        return {}

    def fake_write_skipped_classification_batch_grids(config, run_store):
        _ = config
        calls.append("batch_grids")
        assert run_store is store
        return []

    def fake_render_video(
        config, profile, video_path, run_store, allow_unclassified_annotations
    ):
        _ = config, profile, video_path
        calls.append("render")
        render_kwargs["run_store"] = run_store
        render_kwargs["allow_unclassified_annotations"] = allow_unclassified_annotations
        return store.root / "annotated.mp4"

    def fake_generate_reports(run_store):
        calls.append("report")
        assert run_store is store
        return {}

    result = run_pipeline(
        project_root=tmp_path,
        config=default_config,
        profile=build_full_frame_profile(width=16, height=16),
        video_path=tmp_path / "input.mp4",
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

    assert result is store
    assert calls == ["analyze", "batch_grids", "render", "report"]
    assert render_kwargs["run_store"] is store
    assert render_kwargs["allow_unclassified_annotations"] is True


def test_run_pipeline_can_skip_render_while_still_generating_report(
    default_config, tmp_path, monkeypatch
) -> None:
    calls: list[str] = []
    store = DummyRunStore()

    monkeypatch.setattr(
        run_module.RunStore,
        "create",
        lambda **kwargs: store,
    )

    def fake_analyze_video(project_root, config, profile, video_path, run_store):
        _ = project_root, config, profile, video_path
        calls.append("analyze")
        assert run_store is store
        return store

    def fake_classify_tracks(config, run_store):
        _ = config
        calls.append("classify")
        assert run_store is store
        return {}

    def fake_write_skipped_classification_batch_grids(config, run_store):
        _ = config, run_store
        raise AssertionError("Skipped classification stage should not run")

    def fake_render_video(
        config, profile, video_path, run_store, allow_unclassified_annotations
    ):
        _ = config, profile, video_path, run_store, allow_unclassified_annotations
        raise AssertionError("Render stage should not run")

    def fake_generate_reports(run_store):
        calls.append("report")
        assert run_store is store
        return {}

    result = run_pipeline(
        project_root=tmp_path,
        config=default_config,
        profile=build_full_frame_profile(width=16, height=16),
        video_path=tmp_path / "input.mp4",
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

    assert result is store
    assert calls == ["analyze", "classify", "report"]
