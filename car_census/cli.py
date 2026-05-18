from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import typer
from dotenv import load_dotenv

from config import (
    FULL_FRAME_CAMERA_ID,
    CameraProfile,
    build_effective_config,
    build_full_frame_profile,
    camera_profile_path,
    load_camera_profile,
)
from pipeline.analyze import analyze_video
from pipeline.classify import classify_tracks
from pipeline.render import render_video
from pipeline.report import generate_reports
from pipeline.run import run_pipeline
from pipeline.smooth import smooth_render_tracks
from roi.editor import edit_camera_profile
from storage.run_store import RunStore
from utils.logging import configure_logging
from utils.video import read_first_frame

app = typer.Typer(no_args_is_help=True)
roi_app = typer.Typer(no_args_is_help=True)
app.add_typer(roi_app, name="roi")

SUPPORTED_ACCELERATORS = {"default", "colab-t4", "onnx-cuda", "tensorrt"}


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_config(config_path: Optional[Path]) -> tuple[Path, object]:
    project_root = _project_root()
    config = build_effective_config(root=project_root, config_path=config_path)
    return project_root, config


def _accelerator_overrides(accelerator: str, device: str) -> dict[str, Any]:
    accelerator = accelerator.strip().lower()
    if accelerator == "default":
        return {"project": {"device": device}}
    if accelerator == "colab-t4":
        return {
            "project": {"device": "cuda"},
            "detector": {
                "onnx_execution_providers": [
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ],
                "onnx_require_gpu": True,
            },
            "render": {
                "encode_backend": "auto-nvenc",
                "output_fps": 30.0,
                "nvenc_preset": "p4",
                "nvenc_cq": 23,
            },
        }
    if accelerator == "onnx-cuda":
        return {
            "project": {"device": "cuda"},
            "detector": {
                "onnx_execution_providers": [
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ],
                "onnx_require_gpu": True,
            },
        }
    if accelerator == "tensorrt":
        return {
            "project": {"device": "cuda"},
            "detector": {
                "onnx_execution_providers": [
                    "TensorrtExecutionProvider",
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ],
                "onnx_require_gpu": True,
            },
        }
    supported = ", ".join(sorted(SUPPORTED_ACCELERATORS))
    raise typer.BadParameter(
        f"Unsupported accelerator '{accelerator}'. Expected one of: {supported}."
    )


def _load_config_with_accelerator(
    config_path: Optional[Path],
    device: str,
    accelerator: str = "default",
) -> tuple[Path, object]:
    project_root = _project_root()
    config = build_effective_config(
        root=project_root,
        config_path=config_path,
        overrides=_accelerator_overrides(accelerator, device),
    )
    return project_root, config


def _load_config_with_device(
    config_path: Optional[Path], device: str
) -> tuple[Path, object]:
    return _load_config_with_accelerator(config_path, device)


def _resolve_profile(
    project_root: Path, config: object, video: Path, camera_id: Optional[str]
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
    device: str = typer.Option("auto", "--device", help="cpu, cuda, or auto"),
    config_path: Optional[Path] = typer.Option(None, "--config"),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    configure_logging(verbose)
    project_root, config = _load_config_with_device(config_path, device)
    output_path = camera_profile_path(config, camera_id, root=project_root)
    profile = edit_camera_profile(
        video_path=video, camera_id=camera_id, output_path=output_path
    )
    typer.echo(f"Saved camera profile to {output_path}")
    typer.echo(profile.model_dump_json(indent=2))


@app.command()
def analyze(
    video: Path,
    camera_id: Optional[str] = typer.Option(None, "--camera-id"),
    device: str = typer.Option("auto", "--device", help="cpu, cuda, or auto"),
    accelerator: str = typer.Option(
        "default",
        "--accelerator",
        help="default, colab-t4, onnx-cuda, or tensorrt",
    ),
    config_path: Optional[Path] = typer.Option(None, "--config"),
    run_dir: Optional[Path] = typer.Option(None, "--run-dir"),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    load_dotenv()
    configure_logging(verbose)
    project_root, config = _load_config_with_accelerator(
        config_path, device, accelerator
    )
    profile = _resolve_profile(project_root, config, video, camera_id)
    store = (
        run_dir
        and RunStore.from_existing(run_dir)
        or RunStore.create(
            output_root=project_root / config.project.output_root,
            camera_id=profile.camera_id,
            video_stem=video.stem,
        )
    )
    analyze_video(
        project_root=project_root,
        config=config,
        profile=profile,
        video_path=video,
        run_store=store,
    )
    typer.echo(str(store.root))


@app.command()
def classify(
    run_dir: Path = typer.Option(..., "--run-dir"),
    device: str = typer.Option("auto", "--device", help="cpu, cuda, or auto"),
    config_path: Optional[Path] = typer.Option(None, "--config"),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    load_dotenv()
    configure_logging(verbose)
    project_root, config = _load_config_with_device(config_path, device)
    _ = project_root
    store = RunStore.from_existing(run_dir)
    classify_tracks(config=config, run_store=store)
    typer.echo(str(store.labels_path))


@app.command()
def render(
    run_dir: Path = typer.Option(..., "--run-dir"),
    device: str = typer.Option("auto", "--device", help="cpu, cuda, or auto"),
    accelerator: str = typer.Option(
        "default",
        "--accelerator",
        help="default, colab-t4, onnx-cuda, or tensorrt",
    ),
    config_path: Optional[Path] = typer.Option(None, "--config"),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    load_dotenv()
    configure_logging(verbose)
    project_root, config = _load_config_with_accelerator(
        config_path, device, accelerator
    )
    store = RunStore.from_existing(run_dir)
    manifest = store.read_manifest()
    if manifest.camera_id and manifest.camera_id != FULL_FRAME_CAMERA_ID:
        profile = load_camera_profile(config, manifest.camera_id, root=project_root)
    else:
        profile = build_full_frame_profile(width=manifest.width, height=manifest.height)
    render_video(
        config=config, profile=profile, video_path=manifest.video_path, run_store=store
    )
    typer.echo(str(store.output_video_path))


@app.command()
def smooth(
    run_dir: Path = typer.Option(..., "--run-dir"),
    device: str = typer.Option("auto", "--device", help="cpu, cuda, or auto"),
    config_path: Optional[Path] = typer.Option(None, "--config"),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    load_dotenv()
    configure_logging(verbose)
    project_root, config = _load_config_with_device(config_path, device)
    store = RunStore.from_existing(run_dir)
    manifest = store.read_manifest()
    if manifest.camera_id and manifest.camera_id != FULL_FRAME_CAMERA_ID:
        profile = load_camera_profile(config, manifest.camera_id, root=project_root)
    else:
        profile = build_full_frame_profile(width=manifest.width, height=manifest.height)
    output_path = smooth_render_tracks(config=config, profile=profile, run_store=store)
    typer.echo(str(output_path))


@app.command()
def report(
    run_dir: Path = typer.Option(..., "--run-dir"),
    device: str = typer.Option("auto", "--device", help="cpu, cuda, or auto"),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    configure_logging(verbose)
    _ = device
    store = RunStore.from_existing(run_dir)
    payload = generate_reports(run_store=store)
    typer.echo(typer.style("Report generated", fg=typer.colors.GREEN))
    typer.echo(str(payload))


@app.command()
def run(
    video: Path,
    camera_id: Optional[str] = typer.Option(None, "--camera-id"),
    device: str = typer.Option("auto", "--device", help="cpu, cuda, or auto"),
    accelerator: str = typer.Option(
        "default",
        "--accelerator",
        help="default, colab-t4, onnx-cuda, or tensorrt",
    ),
    config_path: Optional[Path] = typer.Option(None, "--config"),
    skip_classify: bool = typer.Option(
        False, "--skip-classify", help="Skip make/model API calls."
    ),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    load_dotenv()
    configure_logging(verbose)
    project_root, config = _load_config_with_accelerator(
        config_path, device, accelerator
    )
    profile = _resolve_profile(project_root, config, video, camera_id)
    store = run_pipeline(
        project_root=project_root,
        config=config,
        profile=profile,
        video_path=video,
        skip_classification=skip_classify,
    )
    typer.echo(str(store.root))
