from pathlib import Path

from config import AppConfig, build_full_frame_profile
from pipeline import run as run_module
from pipeline.run import run_pipeline
from pipeline.stages import PipelineStages
from storage.run_store import RunStore


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
    assert store.mmr_cache_dir.is_dir()
    assert store.mmr_batch_grids_dir.is_dir()
    assert not (store.root / "render").exists()


def test_run_pipeline_orders_analyze_classify_render_report(
    tmp_path, monkeypatch
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
        config=AppConfig(),
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
    tmp_path, monkeypatch
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
        config=AppConfig(),
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
    tmp_path, monkeypatch
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
        config=AppConfig(),
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
