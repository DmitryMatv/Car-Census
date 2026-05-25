from __future__ import annotations

from pathlib import Path

from config import AppConfig, CameraProfile
from pipeline.analyze import analyze_video
from pipeline.classify import classify_tracks, write_skipped_classification_batch_grids
from pipeline.render import render_video
from pipeline.report import generate_reports
from pipeline.smooth import smooth_render_tracks
from pipeline.stages import PipelineStages
from storage.run_store import RunStore


def _render_video(
    config: AppConfig,
    profile: CameraProfile,
    video_path: Path,
    run_store: RunStore,
    allow_unclassified_annotations: bool,
) -> Path:
    return render_video(
        config=config,
        profile=profile,
        video_path=video_path,
        run_store=run_store,
        allow_unclassified_annotations=allow_unclassified_annotations,
        smooth_render_tracks=smooth_render_tracks,
    )


def default_pipeline_stages() -> PipelineStages:
    return PipelineStages(
        analyze_video=analyze_video,
        classify_tracks=classify_tracks,
        write_skipped_classification_batch_grids=write_skipped_classification_batch_grids,
        render_video=_render_video,
        generate_reports=generate_reports,
    )
