from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import typer
from dotenv import load_dotenv

from config import (
    FULL_FRAME_CAMERA_ID,
    AppConfig,
    CameraProfile,
    build_effective_config,
    build_full_frame_profile,
    camera_profile_path,
    load_camera_profile,
)
from mmr.retrieval_calibrate import calibrate_retrieval_cache
from mmr.retrieval_compact import compact_retrieval_cache
from mmr.retrieval_migrate import migrate_retrieval_embeddings
from mmr.retrieval_seed import seed_retrieval_cache
from mmr.trafficeye_cache import migrate_legacy_response_cache
from pipeline.analyze import analyze_video
from pipeline.classify import classify_tracks
from pipeline.default_stages import default_pipeline_stages
from pipeline.render import render_video
from pipeline.report import generate_reports
from pipeline.run import run_pipeline
from pipeline.smooth import smooth_render_tracks
from roi.editor import edit_camera_profile
from storage.run_store import RunStore
from storage.run_transaction import AnalysisRunTransaction, RunDirectoryError
from utils.logging import configure_logging
from utils.video import read_first_frame

app = typer.Typer(no_args_is_help=True)
roi_app = typer.Typer(no_args_is_help=True)
cache_app = typer.Typer(no_args_is_help=True)
app.add_typer(roi_app, name="roi")
app.add_typer(cache_app, name="cache")

SUPPORTED_ACCELERATORS = {"default", "colab-t4"}
SUPPORTED_DEVICES = {"auto", "cpu", "cuda"}


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _accelerator_overrides(accelerator: str) -> dict[str, Any]:
    accelerator = accelerator.strip().lower()
    if accelerator == "default":
        return {}
    if accelerator == "colab-t4":
        return {
            "render": {
                "encode_backend": "auto-nvenc",
                "output_fps": 30.0,
                "nvenc_preset": "p4",
                "nvenc_cq": 23,
            },
        }
    supported = ", ".join(sorted(SUPPORTED_ACCELERATORS))
    raise typer.BadParameter(
        f"Unsupported accelerator '{accelerator}'. Expected one of: {supported}."
    )


def _device_overrides(device: str) -> dict[str, Any]:
    device = device.strip().lower()
    if device not in SUPPORTED_DEVICES:
        supported = ", ".join(sorted(SUPPORTED_DEVICES))
        raise typer.BadParameter(
            f"Unsupported device '{device}'. Expected one of: {supported}."
        )
    if device == "auto":
        return {}
    return {"detector": {"device": device}}


def _load_config_with_accelerator(
    config_path: Optional[Path],
    accelerator: str = "default",
    device: str = "auto",
) -> tuple[Path, AppConfig]:
    project_root = _project_root()
    overrides = _accelerator_overrides(accelerator)
    overrides = {**overrides, **_device_overrides(device)}
    config = build_effective_config(
        root=project_root,
        config_path=config_path,
        overrides=overrides,
    )
    return project_root, config


def _resolve_retrieval_cache_dir(
    project_root: Path, config: AppConfig, cache_dir: Optional[Path]
) -> Path:
    if cache_dir is not None:
        return cache_dir.expanduser().resolve()
    return (
        project_root / config.project.output_root / config.project.retrieval_cache_dir
    ).resolve()


def _resolve_profile(
    project_root: Path, config: AppConfig, video: Path, camera_id: Optional[str]
) -> CameraProfile:
    if camera_id:
        return load_camera_profile(config, camera_id, root=project_root)
    first_frame = read_first_frame(video)
    height, width = first_frame.shape[:2]
    return build_full_frame_profile(width=width, height=height)


@roi_app.command("edit")
def roi_edit(
    video: Path,
    camera_id: str = typer.Option(..., "--camera-id"),
    config_path: Optional[Path] = typer.Option(None, "--config"),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    configure_logging(verbose)
    project_root, config = _load_config_with_accelerator(config_path)
    try:
        output_path = camera_profile_path(config, camera_id, root=project_root)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--camera-id") from exc
    profile = edit_camera_profile(
        video_path=video, camera_id=camera_id, output_path=output_path
    )
    typer.echo(f"Saved camera profile to {output_path}")
    typer.echo(profile.model_dump_json(indent=2))


@app.command()
def analyze(
    video: Path,
    camera_id: Optional[str] = typer.Option(None, "--camera-id"),
    accelerator: str = typer.Option(
        "default",
        "--accelerator",
        help="default or colab-t4",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Detector device: auto, cpu, or cuda.",
    ),
    config_path: Optional[Path] = typer.Option(None, "--config"),
    run_dir: Optional[Path] = typer.Option(
        None,
        "--run-dir",
        help="Write analysis to this exact run directory.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Replace an existing matching run after a successful reanalysis.",
    ),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    load_dotenv()
    configure_logging(verbose)
    if overwrite and run_dir is None:
        raise typer.BadParameter("--overwrite requires --run-dir")
    project_root, config = _load_config_with_accelerator(
        config_path, accelerator, device
    )
    try:
        profile = _resolve_profile(project_root, config, video, camera_id)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--camera-id") from exc
    if run_dir is None:
        store = RunStore.create(
            output_root=project_root / config.project.output_root,
            camera_id=profile.camera_id,
            video_stem=video.stem,
        )
        analyze_video(
            project_root=project_root,
            config=config,
            profile=profile,
            video_path=video,
            run_store=store,
        )
    else:
        transaction = AnalysisRunTransaction(
            run_dir=run_dir,
            overwrite=overwrite,
            video_path=video,
            camera_id=profile.camera_id,
        )
        try:
            with transaction as staging_store:
                analyze_video(
                    project_root=project_root,
                    config=config,
                    profile=profile,
                    video_path=video,
                    run_store=staging_store,
                )
        except RunDirectoryError as exc:
            raise typer.BadParameter(str(exc), param_hint="--run-dir") from exc
        store = RunStore.from_existing(transaction.target)
    typer.echo(str(store.root))


@app.command()
def classify(
    run_dir: Path = typer.Option(..., "--run-dir"),
    config_path: Optional[Path] = typer.Option(None, "--config"),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    load_dotenv()
    configure_logging(verbose)
    project_root, config = _load_config_with_accelerator(config_path)
    _ = project_root
    store = RunStore.from_existing(run_dir)
    store.validate_analysis_artifacts()
    classify_tracks(config=config, run_store=store)
    typer.echo(str(store.labels_path))


@cache_app.command("seed")
def cache_seed(
    run_dirs: list[Path] = typer.Argument(
        ...,
        help="Completed run directories to import into the shared retrieval cache.",
    ),
    cache_dir: Optional[Path] = typer.Option(
        None,
        "--cache-dir",
        help="Override the configured shared retrieval cache directory.",
    ),
    config_path: Optional[Path] = typer.Option(None, "--config"),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    load_dotenv()
    configure_logging(verbose)
    project_root, config = _load_config_with_accelerator(config_path)
    target_cache_dir = _resolve_retrieval_cache_dir(project_root, config, cache_dir)
    summaries = seed_retrieval_cache(
        run_dirs=[run_dir.expanduser().resolve() for run_dir in run_dirs],
        config=config,
        cache_dir=target_cache_dir,
    )
    typer.echo(f"Retrieval cache: {target_cache_dir}")
    for summary in summaries:
        typer.echo(
            f"{summary.run_dir}: imported={summary.imported}, "
            f"skipped_unaccepted={summary.skipped_unaccepted}, "
            f"skipped_missing_image={summary.skipped_missing_image}"
        )


@cache_app.command("organize")
def cache_organize(
    cache_dir: Optional[Path] = typer.Option(
        None,
        "--cache-dir",
        help="Override the configured shared retrieval cache directory.",
    ),
    config_path: Optional[Path] = typer.Option(None, "--config"),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    configure_logging(verbose)
    project_root, config = _load_config_with_accelerator(config_path)
    target_cache_dir = _resolve_retrieval_cache_dir(project_root, config, cache_dir)
    migrated = migrate_legacy_response_cache(target_cache_dir)
    typer.echo(
        f"Moved {migrated} legacy response files into {target_cache_dir / 'responses'}"
    )


@cache_app.command("compact")
def cache_compact(
    cache_dir: Optional[Path] = typer.Option(
        None,
        "--cache-dir",
        help="Override the configured shared retrieval cache directory.",
    ),
    config_path: Optional[Path] = typer.Option(None, "--config"),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    configure_logging(verbose)
    project_root, config = _load_config_with_accelerator(config_path)
    target_cache_dir = _resolve_retrieval_cache_dir(project_root, config, cache_dir)
    changed = compact_retrieval_cache(config=config, cache_dir=target_cache_dir)
    typer.echo(f"Compacted {changed} retrieval records in {target_cache_dir}")


@cache_app.command("migrate-embeddings")
def cache_migrate_embeddings(
    cache_dir: Optional[Path] = typer.Option(
        None,
        "--cache-dir",
        help="Override the configured shared retrieval cache directory.",
    ),
    config_path: Optional[Path] = typer.Option(None, "--config"),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    load_dotenv()
    configure_logging(verbose)
    project_root, config = _load_config_with_accelerator(config_path)
    target_cache_dir = _resolve_retrieval_cache_dir(project_root, config, cache_dir)
    summary = migrate_retrieval_embeddings(config=config, cache_dir=target_cache_dir)
    typer.echo(
        f"Migrated {summary.migrated} retrieval records in {target_cache_dir}; "
        f"unavailable={summary.unavailable}"
    )


@cache_app.command("calibrate")
def cache_calibrate(
    cache_dir: Optional[Path] = typer.Option(
        None,
        "--cache-dir",
        help="Override the configured shared retrieval cache directory.",
    ),
    config_path: Optional[Path] = typer.Option(None, "--config"),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    configure_logging(verbose)
    project_root, config = _load_config_with_accelerator(config_path)
    target_cache_dir = _resolve_retrieval_cache_dir(project_root, config, cache_dir)
    report = calibrate_retrieval_cache(config=config, cache_dir=target_cache_dir)
    typer.echo(report)
    if report.usable_threshold is None:
        raise typer.Exit(code=1)
    typer.echo(
        f"Calibration artifact: {target_cache_dir / 'retrieval' / 'calibration.json'}"
    )


@app.command()
def render(
    run_dir: Path = typer.Option(..., "--run-dir"),
    accelerator: str = typer.Option(
        "default",
        "--accelerator",
        help="default or colab-t4",
    ),
    config_path: Optional[Path] = typer.Option(None, "--config"),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    configure_logging(verbose)
    project_root, config = _load_config_with_accelerator(config_path, accelerator)
    store = RunStore.from_existing(run_dir)
    store.validate_analysis_artifacts()
    manifest = store.manifest.read()
    if manifest.camera_id and manifest.camera_id != FULL_FRAME_CAMERA_ID:
        profile = load_camera_profile(config, manifest.camera_id, root=project_root)
    else:
        profile = build_full_frame_profile(width=manifest.width, height=manifest.height)
    video_path = manifest.video_path.expanduser()
    if not video_path.is_file():
        typer.secho(
            "Source video recorded in the run manifest does not exist: "
            f"{manifest.video_path}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    render_video(
        config=config,
        profile=profile,
        video_path=video_path,
        run_store=store,
        smooth_render_tracks=smooth_render_tracks,
    )
    typer.echo(str(store.output_video_path))


@app.command()
def smooth(
    run_dir: Path = typer.Option(..., "--run-dir"),
    config_path: Optional[Path] = typer.Option(None, "--config"),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    configure_logging(verbose)
    project_root, config = _load_config_with_accelerator(config_path)
    store = RunStore.from_existing(run_dir)
    store.validate_analysis_artifacts()
    manifest = store.manifest.read()
    if manifest.camera_id and manifest.camera_id != FULL_FRAME_CAMERA_ID:
        profile = load_camera_profile(config, manifest.camera_id, root=project_root)
    else:
        profile = build_full_frame_profile(width=manifest.width, height=manifest.height)
    output_path = smooth_render_tracks(config=config, profile=profile, run_store=store)
    typer.echo(str(output_path))


@app.command()
def report(
    run_dir: Path = typer.Option(..., "--run-dir"),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    configure_logging(verbose)
    store = RunStore.from_existing(run_dir)
    store.validate_analysis_artifacts()
    payload = generate_reports(run_store=store)
    typer.echo(typer.style("Report generated", fg=typer.colors.GREEN))
    typer.echo(str(payload))


@app.command()
def run(
    video: Path,
    camera_id: Optional[str] = typer.Option(None, "--camera-id"),
    accelerator: str = typer.Option(
        "default",
        "--accelerator",
        help="default or colab-t4",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Detector device: auto, cpu, or cuda.",
    ),
    config_path: Optional[Path] = typer.Option(None, "--config"),
    skip_classify: bool = typer.Option(
        False, "--skip-classify", help="Skip make/model API calls."
    ),
    skip_render: bool = typer.Option(
        False, "--skip-render", help="Skip annotated video rendering."
    ),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    load_dotenv()
    configure_logging(verbose)
    project_root, config = _load_config_with_accelerator(
        config_path, accelerator, device
    )
    try:
        profile = _resolve_profile(project_root, config, video, camera_id)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--camera-id") from exc
    store = run_pipeline(
        project_root=project_root,
        config=config,
        profile=profile,
        video_path=video,
        stages=default_pipeline_stages(),
        skip_classification=skip_classify,
        skip_render=skip_render,
    )
    typer.echo(str(store.root))


if __name__ == "__main__":
    app()
