from __future__ import annotations

from pathlib import Path

from config import AppConfig, CameraProfile
from pipeline.stages import PipelineStages
from storage.run_store import RunStore


def run_pipeline(
    project_root: Path,
    config: AppConfig,
    profile: CameraProfile,
    video_path: Path,
    stages: PipelineStages,
    skip_classification: bool = False,
) -> RunStore:
    run_store = RunStore.create(
        output_root=project_root / config.project.output_root,
        camera_id=profile.camera_id,
        video_stem=video_path.stem,
    )
    stages.analyze_video(project_root, config, profile, video_path, run_store)
    if skip_classification:
        stages.write_skipped_classification_batch_grids(config, run_store)
    else:
        stages.classify_tracks(config, run_store)
    stages.render_video(
        config,
        profile,
        video_path,
        run_store,
        skip_classification,
    )
    stages.generate_reports(run_store)
    return run_store
