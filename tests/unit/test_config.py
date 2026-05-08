from car_census.config import (
    AppConfig,
    CameraProfile,
    FULL_FRAME_CAMERA_ID,
    build_full_frame_profile,
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
