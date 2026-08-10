from pathlib import Path

import pytest
from pydantic import ValidationError

from config import (
    FULL_FRAME_CAMERA_ID,
    AppConfig,
    CameraProfile,
    CountLineConfig,
    build_effective_config,
    build_full_frame_profile,
    load_app_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def test_canonical_tracker_confidence_contract(default_config) -> None:
    config = default_config

    assert config.detector.confidence == 0.15
    assert config.tracker.high_conf_det_threshold == 0.30
    assert config.detector.confidence < config.tracker.high_conf_det_threshold
    assert config.tracker.suppress_duplicate_tracks is False


def test_tracker_config_accepts_supported_cmc_methods(config_factory) -> None:
    for cmc_method in ["sparseOptFlow", "orb", "sift", "ecc"]:
        config = config_factory({"tracker": {"cmc_method": cmc_method}})
        assert config.tracker.cmc_method == cmc_method


def test_tracker_config_accepts_duplicate_suppression_options(config_factory) -> None:
    config = config_factory(
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


def test_tracker_config_accepts_sequential_duplicate_options(config_factory) -> None:
    config = config_factory(
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
    config_factory, field: str, value: float
) -> None:
    with pytest.raises(ValidationError):
        config_factory({"tracker": {field: value}})


def test_tracker_config_rejects_null_cmc_method(config_factory) -> None:
    with pytest.raises(ValidationError):
        config_factory({"tracker": {"cmc_method": None}})


def test_tracker_config_rejects_legacy_tracker_fields(config_factory) -> None:
    with pytest.raises(ValidationError):
        config_factory({"tracker": {"reid_half": True}})


def test_tracker_config_rejects_removed_provider_key(config_factory) -> None:
    with pytest.raises(ValidationError):
        config_factory({"tracker": {"provider": "botsort"}})


def test_project_config_rejects_removed_device_key(config_factory) -> None:
    with pytest.raises(ValidationError):
        config_factory({"project": {"device": "cpu"}})


def test_video_config_accepts_fixed_fps(config_factory) -> None:
    config = config_factory({"video": {"fps": 30.0}})

    assert config.video.fps == 30.0
    assert config.video.fps_tolerance == 0.05


def test_canonical_analysis_and_mmr_contract(default_config) -> None:
    config = default_config

    assert config.analysis.fps == 10
    assert config.analysis.batch_size == 16
    assert config.analysis.detector_batch_size is None
    assert config.analysis.min_track_frames == 5
    assert config.analysis.crop_min_spacing_seconds == 0.099
    assert config.render.min_visible_track_observations == 5
    assert config.render.smoothing.observed_smoothing_window == 5
    assert config.render.smoothing.max_missing_analysis_gap_frames == 15
    assert config.mmr.batch_size == 1
    assert config.mmr.batch_grid_columns == 1
    assert config.mmr.batch_cell_size_px == 512
    assert not hasattr(config.analysis, "crop_limit_per_track")
    assert not hasattr(config.mmr, "max_attempts_per_track")


def test_analysis_config_rejects_removed_crop_limit_per_track(config_factory) -> None:
    with pytest.raises(ValidationError):
        config_factory({"analysis": {"crop_limit_per_track": 1}})


def test_canonical_default_yaml_is_complete() -> None:
    config = load_app_config(PROJECT_ROOT / "configs/default.yaml")

    assert config.detector.device == "auto"
    assert config.analysis.detector_batch_size is None
    assert config.render.output_fps is None
    assert config.render.encode_backend == "opencv"


def test_app_config_requires_complete_input() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({})


def test_custom_config_is_a_partial_overlay(tmp_path: Path) -> None:
    custom_config = tmp_path / "custom.yaml"
    custom_config.write_text("mmr:\n  batch_size: 4\n", encoding="utf-8")

    config = build_effective_config(PROJECT_ROOT, config_path=custom_config)

    assert config.mmr.batch_size == 4
    assert config.mmr.batch_grid_columns == 1
    assert config.detector.confidence == 0.15


def test_runtime_overrides_win_over_custom_config(tmp_path: Path) -> None:
    custom_config = tmp_path / "custom.yaml"
    custom_config.write_text("detector:\n  device: cpu\n", encoding="utf-8")

    config = build_effective_config(
        PROJECT_ROOT,
        config_path=custom_config,
        overrides={"detector": {"device": "cuda"}},
    )

    assert config.detector.device == "cuda"


def test_custom_config_overlay_rejects_unknown_fields(tmp_path: Path) -> None:
    custom_config = tmp_path / "custom.yaml"
    custom_config.write_text("render:\n  line_thickness: 1\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        build_effective_config(PROJECT_ROOT, config_path=custom_config)


def test_config_factory_returns_fresh_instances(config_factory) -> None:
    first = config_factory(None)
    second = config_factory(None)

    assert first is not second
    first.render.workers = 7
    assert second.render.workers == 1


def test_config_rejects_unknown_nested_keys(config_factory) -> None:
    with pytest.raises(ValidationError):
        config_factory({"render": {"line_thickness": 1}})


def test_detector_config_accepts_rfdetr_options(config_factory) -> None:
    config = config_factory(
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


def test_detector_config_rejects_removed_compile_for_inference(config_factory) -> None:
    with pytest.raises(ValidationError):
        config_factory({"detector": {"compile_for_inference": True}})


@pytest.mark.parametrize("threshold", [0.0, 1.01])
def test_detector_config_rejects_invalid_nms_threshold(
    config_factory, threshold: float
) -> None:
    with pytest.raises(ValidationError):
        config_factory({"detector": {"nms_iou_threshold": threshold}})


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_detector_config_rejects_out_of_range_confidence(
    config_factory, confidence: float
) -> None:
    with pytest.raises(ValidationError):
        config_factory({"detector": {"confidence": confidence}})


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_tracker_config_rejects_out_of_range_high_confidence_threshold(
    config_factory,
    threshold: float,
) -> None:
    with pytest.raises(ValidationError):
        config_factory({"tracker": {"high_conf_det_threshold": threshold}})


@pytest.mark.parametrize("detector_confidence", [0.30, 0.31])
def test_config_requires_detector_floor_below_tracker_high_confidence_threshold(
    config_factory,
    detector_confidence: float,
) -> None:
    with pytest.raises(ValidationError, match="low-confidence association stage"):
        config_factory(
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


def test_detector_config_accepts_explicit_float32_inference_dtype(
    config_factory,
) -> None:
    config = config_factory({"detector": {"inference_dtype": "float32"}})

    assert config.detector.inference_dtype == "float32"


def test_detector_config_rejects_invalid_inference_dtype(config_factory) -> None:
    with pytest.raises(ValidationError):
        config_factory({"detector": {"inference_dtype": "bfloat16"}})


def test_detector_config_rejects_removed_provider_key(config_factory) -> None:
    with pytest.raises(ValidationError):
        config_factory({"detector": {"provider": "legacy_local"}})


def test_detector_config_rejects_removed_model_key(config_factory) -> None:
    with pytest.raises(ValidationError):
        config_factory({"detector": {"model": "rfdetr-medium"}})


def test_detector_config_rejects_removed_runtime_keys(config_factory) -> None:
    for key, value in [
        ("weights", "weights/legacy-detector.bin"),
        ("iou", 0.45),
        ("execution_providers", ["CPUExecutionProvider"]),
        ("require_gpu", False),
        ("input_dtype", "auto"),
    ]:
        with pytest.raises(ValidationError):
            config_factory({"detector": {key: value}})


def test_render_config_accepts_nvenc_backend(config_factory) -> None:
    config = config_factory(
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


def test_render_config_rejects_unknown_encode_backend(config_factory) -> None:
    with pytest.raises(ValidationError):
        config_factory({"render": {"encode_backend": "cuda-draw"}})


def test_render_config_rejects_invalid_box_alpha(config_factory) -> None:
    for value in [-0.1, 1.1]:
        with pytest.raises(ValidationError):
            config_factory({"render": {"box_alpha": value}})


def test_render_config_rejects_invalid_label_bg_alpha(config_factory) -> None:
    for value in [-0.1, 1.1]:
        with pytest.raises(ValidationError):
            config_factory({"render": {"label_bg_alpha": value}})


def test_render_config_rejects_invalid_box_thickness(config_factory) -> None:
    with pytest.raises(ValidationError):
        config_factory({"render": {"box_thickness": 0}})


def test_render_config_rejects_invalid_counter_position(config_factory) -> None:
    with pytest.raises(ValidationError):
        config_factory({"render": {"counter_position": "center"}})


def test_render_config_rejects_invalid_label_scale_reference_width(
    config_factory,
) -> None:
    with pytest.raises(ValidationError):
        config_factory({"render": {"label_scale_reference_box_width_px": 0}})


def test_render_config_rejects_negative_label_flag_gap(config_factory) -> None:
    with pytest.raises(ValidationError):
        config_factory({"render": {"label_flag_gap_px": -1}})


def test_render_config_rejects_removed_visual_keys(config_factory) -> None:
    for key, value in [
        ("label" + "_shadow_enabled", True),
        ("corner_thickness", 2),
        ("corner_length", 14),
    ]:
        with pytest.raises(ValidationError):
            config_factory({"render": {key: value}})


def test_render_smoothing_rejects_removed_custom_smoothing_keys(config_factory) -> None:
    for key, value in [
        ("interpolate", True),
        ("max_gap_seconds", 0.5),
        ("reject_short_excursions", True),
    ]:
        with pytest.raises(ValidationError):
            config_factory({"render": {"smoothing": {key: value}}})


def test_render_smoothing_rejects_unknown_observed_box_smoothing(
    config_factory,
) -> None:
    with pytest.raises(ValidationError):
        config_factory(
            {"render": {"smoothing": {"observed_box_smoothing": "centered"}}}
        )


def test_render_smoothing_rejects_unknown_interpolation_method(config_factory) -> None:
    with pytest.raises(ValidationError):
        config_factory(
            {"render": {"smoothing": {"interpolation_method": "smoothstep"}}}
        )


def test_render_smoothing_rejects_invalid_observed_smoothing_window(
    config_factory,
) -> None:
    with pytest.raises(ValidationError):
        config_factory({"render": {"smoothing": {"observed_smoothing_window": 0}}})


def test_render_smoothing_rejects_negative_observed_smoothing_shift_ratio(
    config_factory,
) -> None:
    with pytest.raises(ValidationError):
        config_factory(
            {"render": {"smoothing": {"observed_smoothing_max_shift_ratio": -0.1}}}
        )


def test_render_smoothing_rejects_non_positive_max_interpolation_gap(
    config_factory,
) -> None:
    for value in [0, -0.1]:
        with pytest.raises(ValidationError):
            config_factory(
                {"render": {"smoothing": {"max_interpolation_gap_seconds": value}}}
            )
