from __future__ import annotations

from pathlib import Path

from config import AppConfig, CameraProfile
from pipeline.analyze import analyze_video
from pipeline.classify import classify_tracks
from pipeline.render import render_video
from pipeline.report import generate_reports
from storage.run_store import RunStore


def run_pipeline(
    project_root: Path,
    config: AppConfig,
    profile: CameraProfile,
    video_path: Path,
    skip_classification: bool = False,
) -> RunStore:
    run_store = RunStore.create(
        output_root=project_root / config.project.output_root,
        camera_id=profile.camera_id,
        video_stem=video_path.stem,
    )
    analyze_video(
        project_root=project_root,
        config=config,
        profile=profile,
        video_path=video_path,
        run_store=run_store,
    )
    if not skip_classification:
        classify_tracks(config=config, run_store=run_store)
    render_video(
        config=config, profile=profile, video_path=video_path, run_store=run_store
    )
    generate_reports(run_store=run_store)
    return run_store
