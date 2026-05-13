from pathlib import Path

import pytest
from pydantic import ValidationError

from config import (
    AppConfig,
    CameraProfile,
    FULL_FRAME_CAMERA_ID,
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


def test_video_config_accepts_fixed_fps() -> None:
    config = AppConfig.model_validate({"video": {"fps": 30.0}})

    assert config.video.fps == 30.0
    assert config.video.fps_tolerance == 0.05


def test_render_config_accepts_visual_defaults() -> None:
    config = load_app_config(Path("configs/default.yaml"))

    assert config.video.fps == 30.0
    assert config.video.fps_tolerance == 0.05
    assert config.analysis.fps == 10.0
    assert config.analysis.min_track_frames == 10
    assert config.render.min_visible_track_observations == 10
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
    assert config.render.glow_enabled is True
    assert config.render.glow_color == "#FFFFFF"
    assert config.render.glow_radius_px == 9
    assert config.render.label_glow_alpha == 0.30
    assert config.render.label_glow_alpha < config.render.glow_alpha
    assert config.render.unknown_label == "Unknown"
    assert config.render.smoothing.interpolation_method == "hermite"
    assert config.render.smoothing.polynomial_order == 2
    assert config.render.smoothing.reject_short_excursions is True
    assert config.render.smoothing.max_excursion_observations == 2
    assert config.render.smoothing.excursion_center_ratio == 1.25
    assert not hasattr(config.mmr, "max_attempts_per_track")


def test_config_rejects_unknown_nested_keys() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"render": {"line_thickness": 1}})


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
