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


def test_render_config_accepts_visual_defaults() -> None:
    config = load_app_config(Path("configs/default.yaml"))

    assert config.render.box_color == "#A855F7"
    assert config.render.corner_thickness == 4
    assert config.render.corner_length == 20
    assert config.render.label_padding_px == 4
    assert config.render.label_text_color == "#FFFFFF"
    assert config.render.unknown_label == "Unknown"
    assert config.render.smoothing.interpolation_method == "hermite"
    assert config.render.smoothing.polynomial_order == 2


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
