from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProjectConfig(StrictBaseModel):
    output_root: Path
    camera_profiles_dir: Path
    retrieval_cache_dir: Path


class VideoConfig(StrictBaseModel):
    fps: float = Field(gt=0.0)
    fps_tolerance: float = Field(ge=0.0)


class AnalysisConfig(StrictBaseModel):
    fps: float = Field(gt=0.0)
    batch_size: int = Field(ge=1)
    detector_batch_size: int | None = Field(ge=1)
    min_track_frames: int
    min_box_width_px: int = Field(
        validation_alias=AliasChoices("min_box_width_px", "min_box_height_px"),
    )
    crop_padding_ratio: float = Field(ge=0.0)
    crop_padding_px: int = Field(ge=0)
    crop_target_box_range_ratio: float = Field(ge=0.0, le=1.0)
    crop_min_spacing_seconds: float
    crop_jpeg_quality: int


class DetectorConfig(StrictBaseModel):
    device: Literal["auto", "cpu", "cuda"]
    confidence: float = Field(ge=0.0, le=1.0)
    input_size: int = Field(ge=64)
    allowed_class_names: list[str]
    nms_enabled: bool
    nms_iou_threshold: float = Field(gt=0.0, le=1.0)
    nms_class_agnostic: bool
    pretrain_weights: str | None
    include_source_image: bool
    optimize_for_inference: bool
    inference_dtype: Literal["auto", "float32", "float16"]


class ReidConfig(StrictBaseModel):
    """Selective appearance-embedding (ReID) settings.

    Embeddings are computed for a small subset of detections (per-track
    cadence plus rescue candidates) and kept per track in a bounded history;
    they are consumed by the world-space rescue layer to confirm or reject
    identity takeovers.
    """

    enabled: bool
    device: Literal["auto", "cpu", "cuda"]
    embed_every_n_frames: int = Field(ge=1)
    history_length: int = Field(ge=1)
    min_appearance_similarity: float = Field(ge=0.0, le=1.0)
    batch_size: int = Field(ge=1)


class RescueConfig(StrictBaseModel):
    """World-space rescue layer for tracks whose IoU association failed.

    When BoT-SORT spawns a fresh tracklet for a detection that no existing
    track claimed, the rescue layer predicts the missing track's road-plane
    position from its own recent trajectory (constant world velocity, static
    camera) and takes the identity over when the handoff is physically
    plausible. Requires a calibrated homography; without one it stays inert.
    """

    enabled: bool
    max_gap_seconds: float = Field(gt=0.0)
    max_speed_mps: float = Field(gt=0.0)
    max_distance_m: float = Field(gt=0.0)
    lateral_tolerance_m: float = Field(gt=0.0)
    velocity_fit_points: int = Field(ge=2)
    min_direction_speed_mps: float = Field(ge=0.0)


class TrackerConfig(StrictBaseModel):
    lost_track_buffer: int
    max_reassociation_gap_seconds: float | None = Field(ge=0.0)
    track_activation_threshold: float
    minimum_consecutive_frames: int
    minimum_iou_threshold_first_assoc: float
    minimum_iou_threshold_second_assoc: float
    minimum_iou_threshold_unconfirmed_assoc: float
    high_conf_det_threshold: float = Field(ge=0.0, le=1.0)
    enable_cmc: bool
    cmc_method: Literal["orb", "sift", "sparseOptFlow", "ecc"]
    cmc_downscale: int
    instant_first_frame_activation: bool
    frame_rate: int
    ignore_edge_touches: bool
    edge_margin_px: int
    suppress_sequential_duplicate_tracks: bool
    sequential_duplicate_max_gap_seconds: float = Field(gt=0.0)
    sequential_duplicate_prediction_error_ratio: float = Field(ge=0.0)
    sequential_duplicate_min_width_ratio: float = Field(gt=0.0, le=1.0)
    sequential_duplicate_min_height_ratio: float = Field(gt=0.0, le=1.0)
    sequential_duplicate_min_handoff_iou: float = Field(ge=0.0, le=1.0)
    sequential_duplicate_require_same_color: bool
    sequential_duplicate_require_same_generation: bool
    sequential_duplicate_require_same_variation: bool
    world_reassociation_enabled: bool
    world_reassociation_max_gap_seconds: float = Field(gt=0.0)
    world_reassociation_max_speed_mps: float = Field(gt=0.0)
    world_reassociation_max_distance_m: float = Field(gt=0.0)
    sequential_duplicate_max_implied_speed_mps: float | None = Field(gt=0.0)


class RenderSmoothingConfig(StrictBaseModel):
    enabled: bool
    observed_box_smoothing: Literal["none", "causal_average", "local_linear"]
    history_length: int = Field(ge=1)
    observed_smoothing_window: int = Field(ge=1)
    observed_smoothing_max_shift_ratio: float = Field(ge=0.0)
    bridge_missing_analysis_frames: bool
    max_missing_analysis_gap_frames: int = Field(ge=0)
    interpolate_source_frames: bool
    interpolation_method: Literal["linear"]
    max_interpolation_gap_seconds: float | None = Field(gt=0.0)


class MMRConfig(StrictBaseModel):
    api_url: str
    api_key_env: str
    timeout_seconds: float
    accept_model_confidence: float
    mmr_preference: str
    batch_size: int = Field(ge=1)
    batch_grid_columns: int = Field(ge=1)
    batch_cell_size_px: int = Field(ge=64)
    retrieval_mode: Literal["disabled", "shadow", "enforce"]
    retrieval_embedding_distance_threshold: float = Field(ge=0.0, le=2.0)
    retrieval_phash_max_hamming_distance: int = Field(ge=0, le=64)
    retrieval_min_neighbors: int = Field(ge=1)
    retrieval_embedding_api_key_env: str
    retrieval_embedding_model: str
    retrieval_embedding_dimensions: int = Field(ge=1)
    retrieval_calibration_min_same_identity: int = Field(ge=1)
    retrieval_calibration_min_conflicting_identity: int = Field(ge=1)


class RenderConfig(StrictBaseModel):
    codec: str
    output_fps: float | None = Field(gt=0.0)
    encode_backend: Literal["opencv", "ffmpeg", "auto-nvenc", "ffmpeg-nvenc"]
    workers: int = Field(ge=1)
    ffmpeg_path: str
    nvenc_codec: str
    nvenc_preset: str
    nvenc_cq: int = Field(ge=0, le=51)
    min_visible_track_observations: int = Field(ge=1)
    require_crop_eligible_track: bool
    show_unclassified_tracks: bool
    box_color: str
    box_enabled: bool
    box_alpha: float = Field(ge=0.0, le=1.0)
    box_thickness: int = Field(ge=1)
    label_font_scale: float
    label_thickness: int
    label_padding_px: int
    label_gap_px: int
    label_flag_gap_px: int = Field(ge=0)
    label_text_color: str
    label_bev_text_color: str
    label_mixed_text_color: str
    label_bg_color: str
    label_bg_alpha: float = Field(ge=0.0, le=1.0)
    label_scale_reference_box_width_px: int = Field(gt=0)
    counter_enabled: bool
    counter_position: Literal["top_left", "top_right", "bottom_left", "bottom_right"]
    unknown_label: str
    smoothing: RenderSmoothingConfig


class PolygonZoneConfig(StrictBaseModel):
    points: list[list[int]]


class CountLineConfig(StrictBaseModel):
    start: list[int] = Field(min_length=2, max_length=2)
    end: list[int] = Field(min_length=2, max_length=2)
    direction: Literal["A_TO_B", "B_TO_A", "BOTH"] = "BOTH"

    @model_validator(mode="after")
    def validate_nonzero_length(self) -> "CountLineConfig":
        if self.start == self.end:
            raise ValueError("count line start and end must differ")
        return self


class HomographyConfig(StrictBaseModel):
    """Ground-plane calibration mapping pixel coordinates to world meters.

    ``source_points`` are pixel coordinates on the road plane (bottom-center
    anchors of vehicles), ``target_points`` are the matching real-world
    positions in meters. At least 4 non-collinear points are required.
    """

    source_points: list[list[float]] = Field(min_length=4)
    target_points: list[list[float]] = Field(min_length=4)

    @model_validator(mode="after")
    def validate_point_lists(self) -> "HomographyConfig":
        if len(self.source_points) != len(self.target_points):
            raise ValueError(
                "homography.source_points and homography.target_points "
                "must contain the same number of points"
            )
        for name, points in (
            ("source_points", self.source_points),
            ("target_points", self.target_points),
        ):
            for index, point in enumerate(points):
                if len(point) != 2:
                    raise ValueError(
                        f"homography.{name}[{index}] must contain exactly "
                        "2 coordinates (x, y)"
                    )
        return self


class CameraProfile(StrictBaseModel):
    camera_id: str
    polygon: PolygonZoneConfig
    count_line: CountLineConfig | None = None
    homography: HomographyConfig | None = None


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
    project: ProjectConfig
    video: VideoConfig
    analysis: AnalysisConfig
    detector: DetectorConfig
    tracker: TrackerConfig
    reid: ReidConfig
    rescue: RescueConfig
    mmr: MMRConfig
    render: RenderConfig

    @model_validator(mode="after")
    def validate_tracker_confidence_bands(self) -> "AppConfig":
        if self.detector.confidence >= self.tracker.high_conf_det_threshold:
            raise ValueError(
                "detector.confidence must be lower than "
                "tracker.high_conf_det_threshold; equal or inverted thresholds "
                "disable BoT-SORT's low-confidence association stage"
            )
        return self


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


def validate_camera_id(camera_id: str) -> str:
    if (
        not camera_id
        or camera_id in {".", ".."}
        or "/" in camera_id
        or "\\" in camera_id
    ):
        raise ValueError(
            f"Invalid camera id: {camera_id!r}. Camera ids may not be empty, "
            "'.' or '..', or contain path separators."
        )
    return camera_id


def clean_camera_id(camera_id: str) -> str:
    return validate_camera_id(camera_id.removesuffix(".mp4"))


def camera_profile_path(config: AppConfig, camera_id: str, root: Path) -> Path:
    clean_id = clean_camera_id(camera_id)
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
    default_config_path = root / "configs/default.yaml"
    merged = load_yaml(default_config_path)
    if config_path is not None:
        merged = _deep_merge(merged, load_yaml(config_path))
    if overrides:
        merged = _deep_merge(merged, overrides)
    return AppConfig.model_validate(merged)
