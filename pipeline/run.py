from __future__ import annotations

import logging
from pathlib import Path

from config import AppConfig, CameraProfile
from pipeline.stages import PipelineStages
from storage.run_store import RunStore
from storage.run_transaction import AnalysisRunTransaction

logger = logging.getLogger(__name__)


def _discard_reserved_run_root(run_root: Path) -> None:
    try:
        if run_root.exists() and not any(run_root.iterdir()):
            run_root.rmdir()
    except OSError:
        logger.warning(
            "Could not remove reserved run directory: %s",
            run_root,
            exc_info=True,
        )


def run_pipeline(
    project_root: Path,
    config: AppConfig,
    profile: CameraProfile,
    video_path: Path,
    stages: PipelineStages,
    skip_classification: bool = False,
    skip_render: bool = False,
) -> RunStore:
    reserved_root = RunStore.reserve_root(
        output_root=project_root / config.project.output_root,
        camera_id=profile.camera_id,
        video_stem=video_path.stem,
    )
    transaction = AnalysisRunTransaction(
        run_dir=reserved_root,
        overwrite=True,
        video_path=video_path,
        camera_id=profile.camera_id,
    )
    try:
        with transaction as run_store:
            stages.analyze_video(project_root, config, profile, video_path, run_store)
            if skip_classification:
                stages.write_skipped_classification_batch_grids(config, run_store)
            else:
                stages.classify_tracks(config, run_store)
            if not skip_render:
                stages.render_video(
                    config,
                    profile,
                    video_path,
                    run_store,
                    skip_classification,
                )
            stages.generate_reports(run_store)
    except BaseException:
        _discard_reserved_run_root(reserved_root)
        raise
    return RunStore.from_existing(reserved_root)
