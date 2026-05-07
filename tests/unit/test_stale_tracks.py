from car_census.config import AppConfig
from car_census.pipeline.analyze import (
    MutableTrackState,
    _predict_stale_bbox,
    _stale_track_for_frame,
)
from car_census.types import BBox, TrackedObject


def _state(**overrides) -> MutableTrackState:
    state = MutableTrackState(
        track_id=42,
        first_frame_index=10,
        last_frame_index=12,
        frames_seen=3,
    )
    state.previous_bbox = BBox(x1=10, y1=20, x2=50, y2=80)
    state.last_bbox = BBox(x1=15, y1=22, x2=57, y2=84)
    state.bbox_history = [state.previous_bbox, state.last_bbox]
    state.last_confidence = 0.77
    state.last_class_id = 2
    state.last_class_name = "car"
    state.last_centroid = state.last_bbox.center
    state.last_bottom_center = (
        (state.last_bbox.x1 + state.last_bbox.x2) / 2.0,
        state.last_bbox.y2,
    )
    state.last_inside_roi = True
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def test_predict_stale_bbox_returns_last_bbox_without_previous_bbox() -> None:
    config = AppConfig()
    state = _state(previous_bbox=None)

    assert _predict_stale_bbox(state, 1, config) == state.last_bbox


def test_predict_stale_bbox_uses_full_coordinate_velocity_for_one_gap() -> None:
    config = AppConfig.model_validate(
        {
            "render": {
                "stale_track_velocity_scale": 1.0,
                "stale_track_max_velocity_ratio": 1.0,
            }
        }
    )
    state = _state()

    predicted = _predict_stale_bbox(state, 1, config)

    assert predicted == BBox(x1=20, y1=24, x2=64, y2=88)


def test_predict_stale_bbox_uses_full_coordinate_velocity_for_two_gaps() -> None:
    config = AppConfig.model_validate(
        {
            "render": {
                "stale_track_velocity_scale": 1.0,
                "stale_track_max_velocity_ratio": 1.0,
            }
        }
    )
    state = _state()

    predicted = _predict_stale_bbox(state, 2, config)

    assert predicted == BBox(x1=25, y1=26, x2=71, y2=92)


def test_predict_stale_bbox_can_dampen_velocity() -> None:
    config = AppConfig.model_validate(
        {
            "render": {
                "stale_track_velocity_scale": 0.5,
                "stale_track_max_velocity_ratio": 1.0,
            }
        }
    )
    state = _state()

    predicted = _predict_stale_bbox(state, 1, config)

    assert predicted == BBox(x1=17.5, y1=23, x2=60.5, y2=86)


def test_predict_stale_bbox_caps_large_velocity() -> None:
    config = AppConfig.model_validate(
        {
            "render": {
                "stale_track_velocity_scale": 1.0,
                "stale_track_max_velocity_ratio": 0.25,
            }
        }
    )
    state = _state(
        previous_bbox=BBox(x1=0, y1=20, x2=40, y2=80),
        last_bbox=BBox(x1=100, y1=22, x2=140, y2=84),
    )

    predicted = _predict_stale_bbox(state, 1, config)

    assert predicted == BBox(x1=115, y1=24, x2=155, y2=88)


def test_predict_stale_bbox_uses_median_history_delta() -> None:
    config = AppConfig.model_validate(
        {
            "render": {
                "stale_track_velocity_history": 6,
                "stale_track_velocity_scale": 1.0,
                "stale_track_max_velocity_ratio": 1.0,
            }
        }
    )
    state = _state()
    state.bbox_history = [
        BBox(x1=0, y1=20, x2=40, y2=80),
        BBox(x1=5, y1=20, x2=45, y2=80),
        BBox(x1=10, y1=20, x2=50, y2=80),
        BBox(x1=80, y1=20, x2=120, y2=80),
        BBox(x1=85, y1=20, x2=125, y2=80),
    ]
    state.previous_bbox = state.bbox_history[-2]
    state.last_bbox = state.bbox_history[-1]

    predicted = _predict_stale_bbox(state, 1, config)

    assert predicted == BBox(x1=90, y1=20, x2=130, y2=80)


def test_predict_stale_bbox_falls_back_when_prediction_is_invalid() -> None:
    config = AppConfig()
    state = _state(
        previous_bbox=BBox(x1=0, y1=0, x2=100, y2=100),
        last_bbox=BBox(x1=50, y1=50, x2=60, y2=60),
    )

    assert _predict_stale_bbox(state, 2, config) == state.last_bbox


def test_stale_track_returns_none_without_last_bbox() -> None:
    config = AppConfig()
    state = _state(last_bbox=None)

    assert _stale_track_for_frame(state, 13, 1.3, config) is None


def test_stale_track_returns_none_before_minimum_active_frames() -> None:
    config = AppConfig()
    state = _state(frames_seen=config.render.stale_track_min_active_frames - 1)

    assert _stale_track_for_frame(state, 13, 1.3, config) is None


def test_stale_track_is_predicted_for_one_frame_gap() -> None:
    config = AppConfig.model_validate(
        {
            "render": {
                "stale_track_velocity_scale": 1.0,
                "stale_track_max_velocity_ratio": 1.0,
            }
        }
    )
    state = _state()

    track = _stale_track_for_frame(state, 13, 1.3, config)

    assert track is not None
    assert track.predicted is True
    assert track.stale_frames == 1
    assert track.crossed_line is False
    assert track.bbox == BBox(x1=20, y1=24, x2=64, y2=88)
    assert track.confidence == state.last_confidence
    assert track.class_id == state.last_class_id
    assert track.class_name == state.last_class_name
    assert track.centroid == track.bbox.center
    assert track.bottom_center == (42.0, 88)
    assert track.inside_roi == state.last_inside_roi


def test_stale_track_freezes_bbox_without_previous_bbox() -> None:
    config = AppConfig()
    state = _state(previous_bbox=None, bbox_history=[])

    track = _stale_track_for_frame(state, 13, 1.3, config)

    assert track is not None
    assert track.bbox == state.last_bbox
    assert track.centroid == state.last_bbox.center


def test_stale_track_is_predicted_at_max_gap() -> None:
    config = AppConfig()
    state = _state()

    track = _stale_track_for_frame(
        state,
        state.last_frame_index + config.render.stale_track_frames,
        1.4,
        config,
    )

    assert track is not None
    assert track.predicted is True
    assert track.stale_frames == config.render.stale_track_frames


def test_stale_track_returns_none_after_max_gap() -> None:
    config = AppConfig()
    state = _state()

    track = _stale_track_for_frame(
        state,
        state.last_frame_index + config.render.stale_track_frames + 1,
        1.5,
        config,
    )

    assert track is None


def test_tracked_object_defaults_support_existing_json() -> None:
    track = TrackedObject.model_validate(
        {
            "track_id": 1,
            "frame_index": 2,
            "timestamp_seconds": 0.2,
            "bbox": {"x1": 1, "y1": 2, "x2": 3, "y2": 4},
            "confidence": 0.9,
            "centroid": [2, 3],
            "bottom_center": [2, 4],
            "inside_roi": True,
        }
    )

    assert track.predicted is False
    assert track.stale_frames == 0
