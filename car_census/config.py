from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    output_root: Path = Path("outputs")
    device: str = "auto"
    camera_profiles_dir: Path = Path("configs/cameras")


class AnalysisConfig(BaseModel):
    fps: float = 0.0
    imgsz: int = 960
    min_track_frames: int = 3
    min_box_height_px: int = 80
    crop_limit_per_track: int = 3
    crop_min_spacing_seconds: float = 0.5
    crop_jpeg_quality: int = 95


class DetectorConfig(BaseModel):
    provider: str = "ultralytics_local"
    weights: str = "weights/yolo11n.pt"
    confidence: float = 0.25
    iou: float = 0.45
    allowed_class_names: list[str] = Field(default_factory=lambda: ["car"])
    allowed_class_ids: list[int] | None = None


class TrackerConfig(BaseModel):
    track_activation_threshold: float = 0.25
    lost_track_buffer: int = 30
    minimum_matching_threshold: float = 0.8
    minimum_consecutive_frames: int = 3
    frame_rate: int = 0


class MMRConfig(BaseModel):
    api_url: str = "https://trafficeye.ai/recognition"
    api_key_env: str = "TRAFFICEYE_API_KEY"
    timeout_seconds: float = 45.0
    max_attempts_per_track: int = 3
    accept_model_confidence: float = 0.60


class RenderConfig(BaseModel):
    codec: str = "mp4v"
    label_font_scale: float = 0.7
    label_thickness: int = 2
    line_thickness: int = 8
    corner_thickness: int = 8
    trace_length: int = 25
    output_fps: float = 0.0
    unknown_label: str = "unknown"


class PolygonZoneConfig(BaseModel):
    points: list[list[int]]


class CountLineConfig(BaseModel):
    start: list[int]
    end: list[int]
    direction: str = "BOTH"


class CameraProfile(BaseModel):
    camera_id: str
    polygon: PolygonZoneConfig
    count_line: CountLineConfig


FULL_FRAME_CAMERA_ID = "__full_frame__"


def build_full_frame_profile(
    width: int,
    height: int,
    camera_id: str = FULL_FRAME_CAMERA_ID,
) -> CameraProfile:
    mid_y = max(0, height // 2)
    max_x = max(0, width - 1)
    max_y = max(0, height - 1)
    return CameraProfile(
        camera_id=camera_id,
        polygon=PolygonZoneConfig(
            points=[
                [0, 0],
                [max_x, 0],
                [max_x, max_y],
                [0, max_y],
            ]
        ),
        count_line=CountLineConfig(
            start=[0, mid_y],
            end=[max_x, mid_y],
            direction="BOTH",
        ),
    )


class AppConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    mmr: MMRConfig = Field(default_factory=MMRConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML file: {path}")
    return data


def load_app_config(path: Path) -> AppConfig:
    return AppConfig.model_validate(load_yaml(path))


def camera_profile_path(config: AppConfig, camera_id: str, root: Path) -> Path:
    return root / config.project.camera_profiles_dir / f"{camera_id}.yaml"


def load_camera_profile(config: AppConfig, camera_id: str, root: Path) -> CameraProfile:
    path = camera_profile_path(config, camera_id, root)
    if not path.exists():
        raise FileNotFoundError(
            f"Camera profile not found for '{camera_id}'. Expected: {path}"
        )
    return CameraProfile.model_validate(load_yaml(path))


def build_effective_config(
    root: Path,
    config_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> AppConfig:
    resolved_config_path = config_path or (root / "configs/default.yaml")
    merged = load_yaml(resolved_config_path)
    if overrides:
        merged = _deep_merge(merged, overrides)
    return AppConfig.model_validate(merged)
