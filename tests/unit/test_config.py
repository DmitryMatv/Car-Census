from car_census.config import FULL_FRAME_CAMERA_ID, build_full_frame_profile


def test_build_full_frame_profile_covers_entire_frame() -> None:
    profile = build_full_frame_profile(width=1920, height=1080)
    assert profile.camera_id == FULL_FRAME_CAMERA_ID
    assert profile.polygon.points == [[0, 0], [1919, 0], [1919, 1079], [0, 1079]]
    assert profile.count_line.start == [0, 540]
    assert profile.count_line.end == [1919, 540]
    assert profile.count_line.direction == "BOTH"
