from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProjectConfig(StrictBaseModel):
    output_root: Path = Path("output")
    camera_profiles_dir: Path = Path("configs/cameras")


class VideoConfig(StrictBaseModel):
    fps: float = Field(default=30.0, gt=0.0)
    fps_tolerance: float = Field(default=0.05, ge=0.0)


class AnalysisConfig(StrictBaseModel):
    fps: float = Field(default=10.0, gt=0.0)
    batch_size: int = Field(default=32, ge=1)
    detector_batch_size: int | None = Field(default=None, ge=1)
    min_track_frames: int = 10
    min_box_width_px: int = Field(
        default=160,
        validation_alias=AliasChoices("min_box_width_px", "min_box_height_px"),
    )
    crop_padding_ratio: float = Field(default=0.08, ge=0.0)
    crop_padding_px: int = Field(default=0, ge=0)
    crop_target_box_range_ratio: float = Field(default=0.70, ge=0.0, le=1.0)
    crop_min_spacing_seconds: float = 0.5
    crop_jpeg_quality: int = 95


class DetectorConfig(StrictBaseModel):
    device: Literal["auto", "cpu", "cuda"] = "auto"
    confidence: float = 0.30
    input_size: int = Field(default=576, ge=64)
    allowed_class_names: list[str] = Field(default_factory=lambda: ["car"])
    pretrain_weights: str | None = None
    include_source_image: bool = False
    optimize_for_inference: bool = True
    inference_dtype: Literal["auto", "float32", "float16"] = "auto"
    compile_for_inference: bool = False


class TrackerConfig(StrictBaseModel):
    lost_track_buffer: int = 12
    max_reassociation_gap_seconds: float | None = Field(default=0.35, ge=0.0)
    track_activation_threshold: float = 0.30
    minimum_consecutive_frames: int = 2
    minimum_iou_threshold_first_assoc: float = 0.25
    minimum_iou_threshold_second_assoc: float = 0.50
    minimum_iou_threshold_unconfirmed_assoc: float = 0.20
    high_conf_det_threshold: float = 0.30
    enable_cmc: bool = False
    cmc_method: Literal["orb", "sift", "sparseOptFlow", "ecc"] = "sparseOptFlow"
    cmc_downscale: int = 2
    instant_first_frame_activation: bool = True
    frame_rate: int = 0
    ignore_edge_touches: bool = True
    edge_margin_px: int = 10
    suppress_duplicate_tracks: bool = True
    duplicate_track_iou_threshold: float = Field(default=0.80, gt=0.0, le=1.0)
    duplicate_track_containment_threshold: float = Field(default=0.95, gt=0.0, le=1.0)
    duplicate_track_min_area_ratio: float = Field(default=0.30, ge=0.0, le=1.0)
    duplicate_track_center_distance_ratio: float = Field(default=0.30, ge=0.0)
    suppress_sequential_duplicate_tracks: bool = True
    sequential_duplicate_max_gap_seconds: float = Field(default=0.35, gt=0.0)
    sequential_duplicate_prediction_error_ratio: float = Field(default=0.25, ge=0.0)
    sequential_duplicate_min_width_ratio: float = Field(default=0.75, gt=0.0, le=1.0)
    sequential_duplicate_min_height_ratio: float = Field(default=0.70, gt=0.0, le=1.0)
    sequential_duplicate_min_handoff_iou: float = Field(default=0.10, ge=0.0, le=1.0)
    sequential_duplicate_require_same_color: bool = True


class RenderSmoothingConfig(StrictBaseModel):
    enabled: bool = True
    observed_box_smoothing: Literal["none", "causal_average", "local_linear"] = (
        "local_linear"
    )
    history_length: int = Field(default=1, ge=1)
    observed_smoothing_window: int = Field(default=5, ge=1)
    observed_smoothing_max_shift_ratio: float = Field(default=0.10, ge=0.0)
    bridge_missing_analysis_frames: bool = True
    max_missing_analysis_gap_frames: int = Field(default=3, ge=0)
    interpolate_source_frames: bool = True
    interpolation_method: Literal["linear"] = "linear"
    max_interpolation_gap_seconds: float | None = Field(default=None, gt=0.0)


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
    show_unclassified_tracks: bool = False
    box_color: str = "#FFFFFF"
    box_enabled: bool = True
    box_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    box_thickness: int = Field(default=2, ge=1)
    label_font_scale: float = 1.0
    label_thickness: int = 1
    label_padding_px: int = 4
    label_gap_px: int = 5
    label_flag_gap_px: int = Field(default=4, ge=0)
    label_text_color: str = "#FFFFFF"
    label_bev_text_color: str = "#00BFFF"
    label_mixed_text_color: str = "#39FF14"
    label_bg_color: str = "#000000"
    label_bg_alpha: float = Field(default=0.35, ge=0.0, le=1.0)
    label_scale_reference_box_width_px: int = Field(default=90, gt=0)
    counter_enabled: bool = True
    counter_position: Literal[
        "top_left", "top_right", "bottom_left", "bottom_right"
    ] = "top_left"
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
    clean_id = camera_id.removesuffix(".mp4")
    return root / config.project.camera_profiles_dir / f"{clean_id}.yaml"


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
