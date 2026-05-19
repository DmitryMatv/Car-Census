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

    assert config.tracker.provider == "botsort"
    assert config.tracker.lost_track_buffer == 30
    assert config.tracker.track_activation_threshold == 0.7
    assert config.tracker.minimum_consecutive_frames == 2
    assert config.tracker.minimum_iou_threshold_first_assoc == 0.2
    assert config.tracker.minimum_iou_threshold_second_assoc == 0.5
    assert config.tracker.minimum_iou_threshold_unconfirmed_assoc == 0.3
    assert config.tracker.high_conf_det_threshold == 0.6
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


def test_video_config_accepts_fixed_fps() -> None:
    config = AppConfig.model_validate({"video": {"fps": 30.0}})

    assert config.video.fps == 30.0
    assert config.video.fps_tolerance == 0.05


def test_analysis_config_defaults_to_batched_detection() -> None:
    config = AppConfig()

    assert config.analysis.batch_size == 16


def test_render_config_accepts_visual_defaults() -> None:
    config = load_app_config(Path("configs/default.yaml"))

    assert config.video.fps == 30.0
    assert config.video.fps_tolerance == 0.05
    assert config.analysis.fps == 10.0
    assert config.analysis.batch_size == 32
    assert config.analysis.min_track_frames == 8
    assert config.detector.onnx_execution_providers == ["CPUExecutionProvider"]
    assert config.detector.onnx_require_gpu is False
    assert config.render.encode_backend == "opencv"
    assert config.render.output_fps is None
    assert config.render.ffmpeg_path == "ffmpeg"
    assert config.render.nvenc_codec == "h264_nvenc"
    assert config.render.nvenc_preset == "p4"
    assert config.render.nvenc_cq == 23
    assert config.render.min_visible_track_observations == 8
    assert config.render.require_crop_eligible_track is True
    assert config.render.box_color == "#FFFFFF"
    assert config.render.corner_thickness == 4
    assert config.render.corner_length == 20
    assert config.render.label_padding_px == 4
    assert config.render.label_text_color == "#FFFFFF"
    assert config.render.label_bg_color == "#101820"
    assert config.render.label_bg_alpha == 0.0
    assert config.render.label_shadow_enabled is False
    assert config.render.label_shadow_color == "#000000"
    assert config.render.label_shadow_alpha == 0.45
    assert config.render.label_shadow_offset_px == 1
    assert config.render.label_shadow_thickness_extra == 1
    assert config.render.label_smart_position is True
    assert config.render.label_max_offset_px == 48
    assert config.render.glow_enabled is False
    assert config.render.glow_color == "#FFFFFF"
    assert config.render.glow_radius_px == 9
    assert config.render.label_glow_alpha == 0.0
    assert config.render.label_glow_alpha < config.render.glow_alpha
    assert config.render.unknown_label == "Unknown"
    assert config.render.smoothing.interpolation_method == "hermite"
    assert config.render.smoothing.polynomial_order == 2
    assert config.render.smoothing.reject_short_excursions is True
    assert config.render.smoothing.max_excursion_observations == 2
    assert config.render.smoothing.excursion_center_ratio == 1.25
    assert config.mmr.batch_size == 25
    assert config.mmr.batch_grid_columns == 5
    assert config.mmr.batch_cell_size_px == 512
    assert not hasattr(config.mmr, "max_attempts_per_track")


def test_config_rejects_unknown_nested_keys() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"render": {"line_thickness": 1}})


def test_detector_config_accepts_onnx_execution_provider_options() -> None:
    config = AppConfig.model_validate(
        {
            "detector": {
                "onnx_execution_providers": [
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ],
                "onnx_require_gpu": True,
            }
        }
    )

    assert config.detector.onnx_execution_providers == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert config.detector.onnx_require_gpu is True


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


def test_render_smoothing_accepts_supported_interpolation_methods() -> None:
    for interpolation_method in ["linear", "polynomial", "hermite"]:
        config = AppConfig.model_validate(
            {"render": {"smoothing": {"interpolation_method": interpolation_method}}}
        )
        assert config.render.smoothing.interpolation_method == interpolation_method


def test_render_smoothing_rejects_unknown_interpolation_method() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate(
            {"render": {"smoothing": {"interpolation_method": "spline"}}}
        )
