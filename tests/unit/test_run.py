from pathlib import Path

from config import AppConfig, build_full_frame_profile
from pipeline import run as run_module
from pipeline.run import run_pipeline


class DummyRunStore:
    root = Path("output/test-run")


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

    def fake_analyze_video(**kwargs) -> None:
        calls.append("analyze")
        assert kwargs["run_store"] is store

    def fake_classify_tracks(**kwargs) -> None:
        calls.append("classify")
        assert kwargs["run_store"] is store

    def fake_render_video(**kwargs) -> None:
        calls.append("render")
        render_kwargs.update(kwargs)

    def fake_generate_reports(**kwargs) -> None:
        calls.append("report")
        assert kwargs["run_store"] is store

    monkeypatch.setattr(run_module, "analyze_video", fake_analyze_video)
    monkeypatch.setattr(run_module, "classify_tracks", fake_classify_tracks)
    monkeypatch.setattr(run_module, "render_video", fake_render_video)
    monkeypatch.setattr(run_module, "generate_reports", fake_generate_reports)

    result = run_pipeline(
        project_root=tmp_path,
        config=AppConfig(),
        profile=build_full_frame_profile(width=16, height=16),
        video_path=tmp_path / "input.mp4",
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

    def fake_analyze_video(**kwargs) -> None:
        calls.append("analyze")
        assert kwargs["run_store"] is store

    def fake_classify_tracks(**kwargs) -> None:
        calls.append("classify")

    def fake_write_skipped_classification_batch_grids(**kwargs) -> None:
        calls.append("batch_grids")
        assert kwargs["run_store"] is store

    def fake_render_video(**kwargs) -> None:
        calls.append("render")
        render_kwargs.update(kwargs)

    def fake_generate_reports(**kwargs) -> None:
        calls.append("report")
        assert kwargs["run_store"] is store

    monkeypatch.setattr(run_module, "analyze_video", fake_analyze_video)
    monkeypatch.setattr(run_module, "classify_tracks", fake_classify_tracks)
    monkeypatch.setattr(
        run_module,
        "write_skipped_classification_batch_grids",
        fake_write_skipped_classification_batch_grids,
    )
    monkeypatch.setattr(run_module, "render_video", fake_render_video)
    monkeypatch.setattr(run_module, "generate_reports", fake_generate_reports)

    result = run_pipeline(
        project_root=tmp_path,
        config=AppConfig(),
        profile=build_full_frame_profile(width=16, height=16),
        video_path=tmp_path / "input.mp4",
        skip_classification=True,
    )

    assert result is store
    assert calls == ["analyze", "batch_grids", "render", "report"]
    assert render_kwargs["run_store"] is store
    assert render_kwargs["allow_unclassified_annotations"] is True
