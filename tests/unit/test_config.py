from pathlib import Path

import pytest
from pydantic import ValidationError

from config import (
    FULL_FRAME_CAMERA_ID,
    AppConfig,
    CameraProfile,
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
    assert config.tracker.edge_margin_px == 0


def test_tracker_config_uses_roboflow_botsort_defaults() -> None:
    config = AppConfig()

    assert config.tracker.lost_track_buffer == 30
    assert config.tracker.track_activation_threshold == 0.35
    assert config.tracker.minimum_consecutive_frames == 2
    assert config.tracker.minimum_iou_threshold_first_assoc == 0.2
    assert config.tracker.minimum_iou_threshold_second_assoc == 0.5
    assert config.tracker.minimum_iou_threshold_unconfirmed_assoc == 0.3
    assert config.tracker.high_conf_det_threshold == 0.35
    assert config.tracker.enable_cmc is False
    assert config.tracker.cmc_method == "sparseOptFlow"
    assert config.tracker.cmc_downscale == 2
    assert config.tracker.instant_first_frame_activation is True


def test_tracker_config_accepts_supported_cmc_methods() -> None:
    for cmc_method in ["sparseOptFlow", "orb", "sift", "ecc"]:
        config = AppConfig.model_validate({"tracker": {"cmc_method": cmc_method}})
        assert config.tracker.cmc_method == cmc_method


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
    assert config.analysis.batch_size == 32
    assert config.analysis.detector_batch_size is None
    assert config.analysis.min_box_height_px == 160
    assert not hasattr(config.analysis, "crop_limit_per_track")


def test_analysis_config_rejects_removed_crop_limit_per_track() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"analysis": {"crop_limit_per_track": 1}})


def test_render_config_accepts_visual_defaults() -> None:
    config = load_app_config(Path("configs/default.yaml"))

    assert config.video.fps == 30.0
    assert config.video.fps_tolerance == 0.05
    assert config.analysis.fps == 10.0
    assert config.analysis.batch_size == 32
    assert config.analysis.detector_batch_size == 32
    assert config.analysis.min_track_frames == 6
    assert config.analysis.min_box_height_px == 160
    assert config.detector.model == "rfdetr-small"
    assert config.detector.confidence == 0.30
    assert config.detector.input_size == 512
    assert config.detector.allowed_class_names == ["car"]
    assert config.detector.pretrain_weights is None
    assert config.detector.include_source_image is False
    assert config.detector.optimize_for_inference is True
    assert config.detector.inference_dtype == "auto"
    assert config.detector.compile_for_inference is False
    assert config.tracker.track_activation_threshold == 0.30
    assert config.tracker.minimum_consecutive_frames == 2
    assert config.tracker.minimum_iou_threshold_first_assoc == 0.08
    assert config.tracker.minimum_iou_threshold_unconfirmed_assoc == 0.10
    assert config.tracker.high_conf_det_threshold == 0.30
    assert config.tracker.edge_margin_px == 10
    assert config.render.encode_backend == "opencv"
    assert config.render.output_fps is None
    assert config.render.ffmpeg_path == "ffmpeg"
    assert config.render.nvenc_codec == "h264_nvenc"
    assert config.render.nvenc_preset == "p4"
    assert config.render.nvenc_cq == 23
    assert config.render.min_visible_track_observations == 6
    assert config.render.require_crop_eligible_track is True
    assert config.render.show_unclassified_tracks is False
    assert config.render.box_color == "#FFFFFF"
    assert config.render.corner_thickness == 4
    assert config.render.corner_length == 20
    assert config.render.label_padding_px == 4
    assert config.render.label_text_color == "#FFFFFF"
    assert config.render.label_bg_color == "#101820"
    assert config.render.unknown_label == "Unknown"
    assert config.render.smoothing.enabled is True
    assert config.render.smoothing.observed_box_smoothing == "none"
    assert config.render.smoothing.history_length == 1
    assert config.render.smoothing.interpolate_source_frames is True
    assert config.render.smoothing.interpolation_method == "linear"
    assert config.render.smoothing.max_interpolation_gap_seconds == 0.25
    assert config.mmr.batch_size == 25
    assert config.mmr.batch_grid_columns == 5
    assert config.mmr.batch_cell_size_px == 512
    assert not hasattr(config.mmr, "max_attempts_per_track")


def test_config_rejects_unknown_nested_keys() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"render": {"line_thickness": 1}})


def test_detector_config_accepts_rfdetr_options() -> None:
    config = AppConfig.model_validate(
        {
            "detector": {
                "model": "rfdetr-small",
                "device": "cpu",
                "input_size": 512,
                "allowed_class_names": ["car", "truck"],
                "pretrain_weights": "weights/rfdetr-small.pth",
                "include_source_image": True,
                "optimize_for_inference": False,
                "inference_dtype": "float16",
                "compile_for_inference": True,
            }
        }
    )

    assert config.detector.model == "rfdetr-small"
    assert config.detector.device == "cpu"
    assert config.detector.input_size == 512
    assert config.detector.allowed_class_names == ["car", "truck"]
    assert config.detector.pretrain_weights == "weights/rfdetr-small.pth"
    assert config.detector.include_source_image is True
    assert config.detector.optimize_for_inference is False
    assert config.detector.inference_dtype == "float16"
    assert config.detector.compile_for_inference is True


def test_detector_config_accepts_explicit_float32_inference_dtype() -> None:
    config = AppConfig.model_validate({"detector": {"inference_dtype": "float32"}})

    assert config.detector.inference_dtype == "float32"


def test_detector_config_rejects_invalid_inference_dtype() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"detector": {"inference_dtype": "bfloat16"}})


def test_detector_config_rejects_removed_provider_key() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"detector": {"provider": "legacy_local"}})


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


def test_render_config_rejects_removed_visual_keys() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"render": {"label" + "_shadow_enabled": True}})


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


def test_render_smoothing_rejects_non_positive_max_interpolation_gap() -> None:
    for value in [0, -0.1]:
        with pytest.raises(ValidationError):
            AppConfig.model_validate(
                {"render": {"smoothing": {"max_interpolation_gap_seconds": value}}}
            )
