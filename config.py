from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


def _coco_class_names() -> dict[int, str]:
    return {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        4: "airplane",
        5: "bus",
        6: "train",
        7: "truck",
        8: "boat",
        9: "traffic light",
        10: "fire hydrant",
        11: "stop sign",
        12: "parking meter",
        13: "bench",
        14: "bird",
        15: "cat",
        16: "dog",
    }


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProjectConfig(StrictBaseModel):
    output_root: Path = Path("output")
    device: str = "auto"
    camera_profiles_dir: Path = Path("configs/cameras")


class VideoConfig(StrictBaseModel):
    fps: float = Field(default=30.0, gt=0.0)
    fps_tolerance: float = Field(default=0.05, ge=0.0)


class AnalysisConfig(StrictBaseModel):
    fps: float = Field(default=10.0, gt=0.0)
    imgsz: int = 960
    min_track_frames: int = 10
    min_box_height_px: int = 80
    crop_padding_ratio: float = Field(default=0.08, ge=0.0)
    crop_padding_px: int = Field(default=0, ge=0)
    crop_limit_per_track: int = 1
    crop_target_box_range_ratio: float = Field(default=0.70, ge=0.0, le=1.0)
    crop_min_spacing_seconds: float = 0.5
    crop_jpeg_quality: int = 95


class DetectorConfig(StrictBaseModel):
    provider: str = "onnxruntime_local"
    weights: str = "weights/yolo26s.onnx"
    confidence: float = 0.4
    iou: float = 0.4
    allowed_class_names: list[str] = Field(default_factory=lambda: ["car"])
    allowed_class_ids: list[int] | None = None
    class_names: dict[int, str] = Field(default_factory=_coco_class_names)
    onnx_threads: int = 4
    onnx_execution_providers: list[str] = Field(
        default_factory=lambda: ["CPUExecutionProvider"]
    )
    onnx_require_gpu: bool = False


class TrackerConfig(StrictBaseModel):
    provider: str = "botsort"
    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.10
    new_track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.80
    minimum_consecutive_frames: int = 3
    frame_rate: int = 0
    fuse_first_associate: bool = True
    cmc_method: str | None = None
    with_reid: bool = False
    reid_weights: Path = Path("osnet_x0_25_msmt17.pt")
    reid_device: str = "auto"
    reid_half: bool = False
    proximity_thresh: float = 0.50
    appearance_thresh: float = 0.80
    ignore_edge_touches: bool = True
    edge_margin_px: int = 0


class RenderSmoothingConfig(StrictBaseModel):
    enabled: bool = True
    interpolate: bool = True
    interpolation_method: Literal["linear", "polynomial", "hermite"] = "hermite"
    smooth_keyframes: bool = True
    window_seconds: float = 0.25
    max_gap_seconds: float = 0.35
    min_observations: int = 3
    polynomial_order: int = Field(default=2, ge=1, le=3)
    max_center_offset_ratio: float = 0.20
    max_size_delta_ratio: float = 0.20
    reject_short_excursions: bool = True
    max_excursion_observations: int = Field(default=2, ge=1)
    excursion_center_ratio: float = Field(default=1.25, ge=0.0)


class MMRConfig(StrictBaseModel):
    api_url: str = "https://trafficeye.ai/recognition"
    api_key_env: str = "TRAFFICEYE_API_KEY"
    timeout_seconds: float = 45.0
    accept_model_confidence: float = 0.60
    tasks: list[str] = Field(default_factory=lambda: ["DETECTION", "MMR"])
    requested_detection_types: list[str] = Field(default_factory=lambda: ["BOX"])
    mmr_preference: str = "BOX"
    batch_size: int = Field(default=16, ge=1)
    batch_grid_columns: int = Field(default=4, ge=1)
    batch_cell_size_px: int = Field(default=512, ge=64)


class RenderConfig(StrictBaseModel):
    codec: str = "mp4v"
    output_fps: float | None = Field(default=None, gt=0.0)
    encode_backend: Literal["opencv", "auto-nvenc", "ffmpeg-nvenc"] = "opencv"
    ffmpeg_path: str = "ffmpeg"
    nvenc_codec: str = "h264_nvenc"
    nvenc_preset: str = "p4"
    nvenc_cq: int = Field(default=23, ge=0, le=51)
    min_visible_track_observations: int = Field(default=10, ge=1)
    require_crop_eligible_track: bool = False
    box_color: str = "#FFFFFF"
    label_font_scale: float = 1.0
    label_thickness: int = 1
    label_padding_px: int = 4
    label_line_gap_px: int = Field(default=2, ge=0)
    label_max_width_ratio: float = Field(default=0.35, gt=0.0, le=1.0)
    label_min_width_px: int = Field(default=160, ge=1)
    label_gap_px: int = 5
    label_text_color: str = "#FFFFFF"
    label_bg_color: str = "#101820"
    label_bg_alpha: float = Field(default=0.0, ge=0.0, le=1.0)
    label_shadow_enabled: bool = False
    label_shadow_color: str = "#000000"
    label_shadow_alpha: float = Field(default=0.45, ge=0.0, le=1.0)
    label_shadow_offset_px: int = Field(default=1, ge=0)
    label_shadow_thickness_extra: int = Field(default=1, ge=0)
    label_smart_position: bool = True
    label_max_offset_px: int = Field(default=48, ge=0)
    glow_enabled: bool = True
    glow_color: str = "#FFFFFF"
    glow_radius_px: int = Field(default=9, ge=0)
    glow_alpha: float = Field(default=0.55, ge=0.0, le=1.0)
    label_glow_radius_px: int = Field(default=7, ge=0)
    label_glow_alpha: float = Field(default=0.30, ge=0.0, le=1.0)
    corner_thickness: int = 2
    corner_length: int = 32
    unknown_label: str = "UNKNOWN"
    smoothing: RenderSmoothingConfig = Field(default_factory=RenderSmoothingConfig)


class PolygonZoneConfig(StrictBaseModel):
    points: list[list[int]]


class CountLineConfig(StrictBaseModel):
    start: list[int]
    end: list[int]
    direction: str = "BOTH"


class CameraProfile(StrictBaseModel):
    camera_id: str
    polygon: PolygonZoneConfig
    count_line: CountLineConfig | None = None


FULL_FRAME_CAMERA_ID = "__full_frame__"


def build_full_frame_profile(
    width: int,
    height: int,
    camera_id: str = FULL_FRAME_CAMERA_ID,
) -> CameraProfile:
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
    )


class AppConfig(StrictBaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
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
