from pathlib import Path

import pytest
from pydantic import ValidationError

from config import (
    FULL_FRAME_CAMERA_ID,
    AppConfig,
    CameraProfile,
    CountLineConfig,
    build_full_frame_profile,
    load_app_config,
)


def test_build_full_frame_profile_covers_entire_frame() -> None:
    profile = build_full_frame_profile(width=1920, height=1080)
    assert profile.camera_id == FULL_FRAME_CAMERA_ID
    assert profile.polygon.points == [[0, 0], [1919, 0], [1919, 1079], [0, 1079]]
    assert profile.count_line is None


def test_camera_profile_accepts_polygon_without_count_line() -> None:
    profile = CameraProfile.model_validate(
        {
            "camera_id": "test",
            "polygon": {"points": [[0, 0], [100, 0], [100, 100], [0, 100]]},
        }
    )
    assert profile.count_line is None


def test_edge_touch_filtering_is_enabled_by_default() -> None:
    config = AppConfig()
    assert config.tracker.ignore_edge_touches is True
    assert config.tracker.edge_margin_px == 10


def test_tracker_config_uses_recommended_split_swap_mitigation_defaults() -> None:
    config = AppConfig()

    assert config.tracker.lost_track_buffer == 12
    assert config.tracker.max_reassociation_gap_seconds == 0.35
    assert config.tracker.track_activation_threshold == 0.30
    assert config.tracker.minimum_consecutive_frames == 2
    assert config.tracker.minimum_iou_threshold_first_assoc == 0.25
    assert config.tracker.minimum_iou_threshold_second_assoc == 0.50
    assert config.tracker.minimum_iou_threshold_unconfirmed_assoc == 0.20
    assert config.tracker.high_conf_det_threshold == 0.30
    assert config.tracker.enable_cmc is False
    assert config.tracker.cmc_method == "sparseOptFlow"
    assert config.tracker.cmc_downscale == 2
    assert config.tracker.instant_first_frame_activation is True
    assert config.tracker.suppress_duplicate_tracks is False
    assert config.tracker.duplicate_track_iou_threshold == 0.80
    assert config.tracker.duplicate_track_containment_threshold == 0.95
    assert config.tracker.duplicate_track_min_area_ratio == 0.30
    assert config.tracker.duplicate_track_center_distance_ratio == 0.30


def test_tracker_config_accepts_supported_cmc_methods() -> None:
    for cmc_method in ["sparseOptFlow", "orb", "sift", "ecc"]:
        config = AppConfig.model_validate({"tracker": {"cmc_method": cmc_method}})
        assert config.tracker.cmc_method == cmc_method


def test_tracker_config_accepts_duplicate_suppression_options() -> None:
    config = AppConfig.model_validate(
        {
            "tracker": {
                "suppress_duplicate_tracks": False,
                "duplicate_track_iou_threshold": 0.95,
                "duplicate_track_containment_threshold": 0.99,
                "duplicate_track_min_area_ratio": 0.40,
                "duplicate_track_center_distance_ratio": 0.20,
            }
        }
    )

    assert config.tracker.suppress_duplicate_tracks is False
    assert config.tracker.duplicate_track_iou_threshold == 0.95
    assert config.tracker.duplicate_track_containment_threshold == 0.99
    assert config.tracker.duplicate_track_min_area_ratio == 0.40
    assert config.tracker.duplicate_track_center_distance_ratio == 0.20


def test_tracker_config_accepts_sequential_duplicate_options() -> None:
    config = AppConfig.model_validate(
        {
            "tracker": {
                "suppress_sequential_duplicate_tracks": False,
                "sequential_duplicate_max_gap_seconds": 0.40,
                "sequential_duplicate_prediction_error_ratio": 0.20,
                "sequential_duplicate_min_width_ratio": 0.80,
                "sequential_duplicate_min_height_ratio": 0.75,
                "sequential_duplicate_min_handoff_iou": 0.15,
                "sequential_duplicate_require_same_color": False,
                "sequential_duplicate_require_same_generation": False,
                "sequential_duplicate_require_same_variation": False,
            }
        }
    )

    assert config.tracker.suppress_sequential_duplicate_tracks is False
    assert config.tracker.sequential_duplicate_max_gap_seconds == 0.40
    assert config.tracker.sequential_duplicate_prediction_error_ratio == 0.20
    assert config.tracker.sequential_duplicate_min_width_ratio == 0.80
    assert config.tracker.sequential_duplicate_min_height_ratio == 0.75
    assert config.tracker.sequential_duplicate_min_handoff_iou == 0.15
    assert config.tracker.sequential_duplicate_require_same_color is False
    assert config.tracker.sequential_duplicate_require_same_generation is False
    assert config.tracker.sequential_duplicate_require_same_variation is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sequential_duplicate_max_gap_seconds", 0.0),
        ("sequential_duplicate_prediction_error_ratio", -0.1),
        ("sequential_duplicate_min_width_ratio", 0.0),
        ("sequential_duplicate_min_width_ratio", 1.1),
        ("sequential_duplicate_min_height_ratio", 0.0),
        ("sequential_duplicate_min_height_ratio", 1.1),
        ("sequential_duplicate_min_handoff_iou", -0.1),
        ("sequential_duplicate_min_handoff_iou", 1.1),
    ],
)
def test_tracker_config_rejects_invalid_sequential_duplicate_bounds(
    field: str, value: float
) -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"tracker": {field: value}})


def test_tracker_config_rejects_null_cmc_method() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"tracker": {"cmc_method": None}})


def test_tracker_config_rejects_legacy_tracker_fields() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"tracker": {"reid_half": True}})


def test_tracker_config_rejects_removed_provider_key() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"tracker": {"provider": "botsort"}})


def test_project_config_rejects_removed_device_key() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"project": {"device": "cpu"}})


def test_video_config_accepts_fixed_fps() -> None:
    config = AppConfig.model_validate({"video": {"fps": 30.0}})

    assert config.video.fps == 30.0
    assert config.video.fps_tolerance == 0.05


def test_analysis_config_defaults_to_batched_detection() -> None:
    config = AppConfig()

    assert config.analysis.fps == 10.0
    assert config.analysis.batch_size == 16
    assert config.detector.nms_enabled is True
    assert config.detector.nms_iou_threshold == 0.80
    assert config.detector.nms_class_agnostic is True
    assert config.analysis.detector_batch_size is None
    assert config.analysis.min_box_width_px == 160
    assert not hasattr(config.analysis, "crop_limit_per_track")


def test_analysis_config_rejects_removed_crop_limit_per_track() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"analysis": {"crop_limit_per_track": 1}})


def test_render_config_accepts_visual_defaults() -> None:
    config = load_app_config(Path("configs/default.yaml"))

    assert config.video.fps == 30.0
    assert config.video.fps_tolerance == 0.05
    assert config.analysis.fps == 10.0
    assert config.analysis.batch_size == 16
    assert config.detector.confidence == 0.15
    assert config.detector.input_size == 576
    assert config.detector.nms_enabled is True
    assert config.detector.nms_iou_threshold == 0.80
    assert config.detector.nms_class_agnostic is True
    assert config.detector.pretrain_weights is None
    assert config.detector.include_source_image is False
    assert config.detector.optimize_for_inference is True
    assert config.detector.inference_dtype == "auto"
    assert config.tracker.track_activation_threshold == 0.30
    assert config.tracker.minimum_consecutive_frames == 2
    assert config.tracker.suppress_duplicate_tracks is False
    assert config.tracker.duplicate_track_min_area_ratio == 0.30
    assert config.tracker.duplicate_track_center_distance_ratio == 0.30
    assert config.render.encode_backend == "opencv"
    assert config.render.output_fps is None
    assert config.render.ffmpeg_path == "ffmpeg"
    assert config.render.nvenc_codec == "h264_nvenc"
    assert config.render.nvenc_preset == "p4"
    assert config.render.nvenc_cq == 23
    assert config.render.require_crop_eligible_track is True
    assert config.render.show_unclassified_tracks is False
    assert config.render.box_color == "#FFFFFF"
    assert config.render.box_alpha == 0.5
    assert config.render.box_thickness == 2
    assert config.render.counter_enabled is True
    assert config.render.counter_position == "top_left"
    assert config.render.label_thickness == 1
    assert config.render.label_padding_px == 6
    assert config.render.label_text_color == "#FFFFFF"
    assert config.render.label_bg_color == "#000000"
    assert config.render.smoothing.enabled is True
    assert config.render.smoothing.observed_box_smoothing == "local_linear"
    assert config.render.smoothing.history_length == 1
    assert config.render.smoothing.observed_smoothing_window == 5
    assert config.render.smoothing.observed_smoothing_max_shift_ratio == 0.10
    assert config.render.smoothing.bridge_missing_analysis_frames is True
    assert config.render.smoothing.interpolate_source_frames is True
    assert config.render.smoothing.interpolation_method == "linear"
    assert config.render.smoothing.max_interpolation_gap_seconds == 0.25
    assert config.mmr.batch_cell_size_px == 512
    assert not hasattr(config.mmr, "max_attempts_per_track")


def test_config_rejects_unknown_nested_keys() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"render": {"line_thickness": 1}})


def test_detector_config_accepts_rfdetr_options() -> None:
    config = AppConfig.model_validate(
        {
            "detector": {
                "device": "cpu",
                "input_size": 576,
                "allowed_class_names": ["car", "truck"],
                "nms_enabled": False,
                "nms_iou_threshold": 0.90,
                "nms_class_agnostic": False,
                "pretrain_weights": "weights/rfdetr-medium.pth",
                "include_source_image": True,
                "optimize_for_inference": False,
                "inference_dtype": "float16",
            }
        }
    )

    assert config.detector.device == "cpu"
    assert config.detector.input_size == 576
    assert config.detector.allowed_class_names == ["car", "truck"]
    assert config.detector.nms_enabled is False
    assert config.detector.nms_iou_threshold == 0.90
    assert config.detector.nms_class_agnostic is False
    assert config.detector.pretrain_weights == "weights/rfdetr-medium.pth"
    assert config.detector.include_source_image is True
    assert config.detector.optimize_for_inference is False
    assert config.detector.inference_dtype == "float16"


def test_detector_config_rejects_removed_compile_for_inference() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"detector": {"compile_for_inference": True}})


@pytest.mark.parametrize("threshold", [0.0, 1.01])
def test_detector_config_rejects_invalid_nms_threshold(threshold: float) -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"detector": {"nms_iou_threshold": threshold}})


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_detector_config_rejects_out_of_range_confidence(confidence: float) -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"detector": {"confidence": confidence}})


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_tracker_config_rejects_out_of_range_high_confidence_threshold(
    threshold: float,
) -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"tracker": {"high_conf_det_threshold": threshold}})


@pytest.mark.parametrize("detector_confidence", [0.30, 0.31])
def test_config_requires_detector_floor_below_tracker_high_confidence_threshold(
    detector_confidence: float,
) -> None:
    with pytest.raises(ValidationError, match="low-confidence association stage"):
        AppConfig.model_validate(
            {
                "detector": {"confidence": detector_confidence},
                "tracker": {"high_conf_det_threshold": 0.30},
            }
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"start": [0], "end": [10, 10]},
        {"start": [0, 0, 1], "end": [10, 10]},
        {"start": [0, 0], "end": [0, 0]},
        {"start": [0, 0], "end": [10, 10], "direction": "SIDEWAYS"},
    ],
)
def test_count_line_config_rejects_invalid_geometry_and_direction(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        CountLineConfig.model_validate(payload)


def test_detector_config_accepts_explicit_float32_inference_dtype() -> None:
    config = AppConfig.model_validate({"detector": {"inference_dtype": "float32"}})

    assert config.detector.inference_dtype == "float32"


def test_detector_config_rejects_invalid_inference_dtype() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"detector": {"inference_dtype": "bfloat16"}})


def test_detector_config_rejects_removed_provider_key() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"detector": {"provider": "legacy_local"}})


def test_detector_config_rejects_removed_model_key() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"detector": {"model": "rfdetr-medium"}})


def test_detector_config_rejects_removed_runtime_keys() -> None:
    for key, value in [
        ("weights", "weights/legacy-detector.bin"),
        ("iou", 0.45),
        ("execution_providers", ["CPUExecutionProvider"]),
        ("require_gpu", False),
        ("input_dtype", "auto"),
    ]:
        with pytest.raises(ValidationError):
            AppConfig.model_validate({"detector": {key: value}})


def test_render_config_accepts_nvenc_backend() -> None:
    config = AppConfig.model_validate(
        {
            "render": {
                "encode_backend": "auto-nvenc",
                "ffmpeg_path": "/usr/bin/ffmpeg",
                "nvenc_codec": "hevc_nvenc",
                "nvenc_preset": "p5",
                "nvenc_cq": 19,
            }
        }
    )

    assert config.render.encode_backend == "auto-nvenc"
    assert config.render.ffmpeg_path == "/usr/bin/ffmpeg"
    assert config.render.nvenc_codec == "hevc_nvenc"
    assert config.render.nvenc_preset == "p5"
    assert config.render.nvenc_cq == 19


def test_render_config_rejects_unknown_encode_backend() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"render": {"encode_backend": "cuda-draw"}})


def test_render_config_rejects_invalid_box_alpha() -> None:
    for value in [-0.1, 1.1]:
        with pytest.raises(ValidationError):
            AppConfig.model_validate({"render": {"box_alpha": value}})


def test_render_config_rejects_invalid_label_bg_alpha() -> None:
    for value in [-0.1, 1.1]:
        with pytest.raises(ValidationError):
            AppConfig.model_validate({"render": {"label_bg_alpha": value}})


def test_render_config_rejects_invalid_box_thickness() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"render": {"box_thickness": 0}})


def test_render_config_rejects_invalid_counter_position() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"render": {"counter_position": "center"}})


def test_render_config_rejects_invalid_label_scale_reference_width() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"render": {"label_scale_reference_box_width_px": 0}})


def test_render_config_rejects_negative_label_flag_gap() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"render": {"label_flag_gap_px": -1}})


def test_render_config_rejects_removed_visual_keys() -> None:
    for key, value in [
        ("label" + "_shadow_enabled", True),
        ("corner_thickness", 2),
        ("corner_length", 14),
    ]:
        with pytest.raises(ValidationError):
            AppConfig.model_validate({"render": {key: value}})


def test_render_smoothing_rejects_removed_custom_smoothing_keys() -> None:
    for key, value in [
        ("interpolate", True),
        ("max_gap_seconds", 0.5),
        ("reject_short_excursions", True),
    ]:
        with pytest.raises(ValidationError):
            AppConfig.model_validate({"render": {"smoothing": {key: value}}})


def test_render_smoothing_rejects_unknown_observed_box_smoothing() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate(
            {"render": {"smoothing": {"observed_box_smoothing": "centered"}}}
        )


def test_render_smoothing_rejects_unknown_interpolation_method() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate(
            {"render": {"smoothing": {"interpolation_method": "smoothstep"}}}
        )


def test_render_smoothing_rejects_invalid_observed_smoothing_window() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate(
            {"render": {"smoothing": {"observed_smoothing_window": 0}}}
        )


def test_render_smoothing_rejects_negative_observed_smoothing_shift_ratio() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate(
            {"render": {"smoothing": {"observed_smoothing_max_shift_ratio": -0.1}}}
        )


def test_render_smoothing_rejects_non_positive_max_interpolation_gap() -> None:
    for value in [0, -0.1]:
        with pytest.raises(ValidationError):
            AppConfig.model_validate(
                {"render": {"smoothing": {"max_interpolation_gap_seconds": value}}}
            )
