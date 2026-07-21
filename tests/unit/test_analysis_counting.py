from __future__ import annotations

import warnings

import numpy as np
import pytest
import supervision as sv

from config import CameraProfile, PolygonZoneConfig
from models import BBox
from pipeline.analysis_counting import TrackCounter
from pipeline.analysis_track_state import MutableTrackState
from pipeline.analysis_tracking import FrameTrackingInput, TrackObservation


def _profile(direction: str = "BOTH") -> CameraProfile:
    return CameraProfile.model_validate(
        {
            "camera_id": "test",
            "polygon": {
                "points": [[0, 0], [200, 0], [200, 100], [0, 100]],
            },
            "count_line": {
                "start": [0, 50],
                "end": [100, 50],
                "direction": direction,
            },
        }
    )


def _observation(track_id: int, center_x: float, bottom_y: float) -> TrackObservation:
    bbox = BBox(
        x1=center_x - 5,
        y1=bottom_y - 10,
        x2=center_x + 5,
        y2=bottom_y,
    )
    return TrackObservation(
        index=0,
        track_id=track_id,
        bbox=bbox,
        confidence=0.9,
        class_id=2,
        class_name="car",
        bottom_center=bbox.bottom_center,
    )


def _frame_input(frame_index: int) -> FrameTrackingInput:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    return FrameTrackingInput(
        frame_index=frame_index,
        timestamp_seconds=frame_index / 10,
        frame=frame,
        roi_frame=frame,
        roi_offset=(0, 0),
        detections=sv.Detections.empty(),
    )


def test_line_zone_preserves_a_to_b_and_b_to_a_direction_names() -> None:
    counter = TrackCounter(_profile())

    assert (
        counter.crossing_directions(
            [_observation(1, center_x=25, bottom_y=40), _observation(2, 75, 60)]
        )
        == {}
    )
    assert counter.crossing_directions(
        [_observation(1, center_x=25, bottom_y=60), _observation(2, 75, 40)]
    ) == {1: "A_TO_B", 2: "B_TO_A"}


def test_line_zone_ignores_crossing_of_finite_line_extension() -> None:
    counter = TrackCounter(_profile())

    counter.crossing_directions([_observation(1, center_x=150, bottom_y=40)])

    assert (
        counter.crossing_directions([_observation(1, center_x=150, bottom_y=60)]) == {}
    )


def test_line_zone_uses_only_bottom_center_anchor() -> None:
    counter = TrackCounter(_profile())

    counter.crossing_directions([_observation(1, center_x=50, bottom_y=55)])

    assert (
        counter.crossing_directions([_observation(1, center_x=50, bottom_y=55)]) == {}
    )


def test_line_zone_handles_empty_observation_frames() -> None:
    assert TrackCounter(_profile()).crossing_directions([]) == {}


def test_line_zone_compatibility_guard_does_not_hide_unrelated_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counter = TrackCounter(_profile())

    def warning_trigger(
        detections: sv.Detections,
    ) -> tuple[np.ndarray, np.ndarray]:
        warnings.warn("unrelated deprecation", DeprecationWarning, stacklevel=1)
        empty = np.zeros(len(detections), dtype=bool)
        return empty, empty

    assert counter._line_zone is not None
    monkeypatch.setattr(counter._line_zone, "trigger", warning_trigger)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(DeprecationWarning, match="unrelated deprecation"):
            counter.crossing_directions([_observation(1, center_x=25, bottom_y=40)])


def test_track_counter_accepts_one_matching_crossing_per_track() -> None:
    counter = TrackCounter(_profile("A_TO_B"))
    state = MutableTrackState(track_id=1, first_frame_index=0, last_frame_index=0)
    first = _observation(1, center_x=50, bottom_y=40)
    second = _observation(1, center_x=50, bottom_y=60)
    third = _observation(1, center_x=50, bottom_y=40)

    counter.crossing_directions([first])
    second_direction = counter.crossing_directions([second]).get(1)
    accepted = counter.update(
        state,
        second,
        _frame_input(1),
        inside_roi=True,
        crossed_direction=second_direction,
    )
    third_direction = counter.crossing_directions([third]).get(1)
    repeated = counter.update(
        state,
        third,
        _frame_input(2),
        inside_roi=True,
        crossed_direction=third_direction,
    )

    assert accepted.crossed_line is True
    assert accepted.counted_event is not None
    assert accepted.counted_event.direction == "A_TO_B"
    assert repeated.crossed_line is False
    assert repeated.counted_event is None


def test_track_counter_rejects_wrong_direction_and_outside_roi() -> None:
    wrong_direction_counter = TrackCounter(_profile("B_TO_A"))
    outside_counter = TrackCounter(_profile("BOTH"))
    first = _observation(1, center_x=50, bottom_y=40)
    second = _observation(1, center_x=50, bottom_y=60)
    wrong_direction_counter.crossing_directions([first])
    outside_counter.crossing_directions([first])

    wrong_direction = wrong_direction_counter.crossing_directions([second]).get(1)
    outside_direction = outside_counter.crossing_directions([second]).get(1)
    wrong_state = MutableTrackState(track_id=1, first_frame_index=0, last_frame_index=0)
    outside_state = MutableTrackState(
        track_id=1, first_frame_index=0, last_frame_index=0
    )

    assert (
        wrong_direction_counter.update(
            wrong_state,
            second,
            _frame_input(1),
            inside_roi=True,
            crossed_direction=wrong_direction,
        ).counted_event
        is None
    )
    assert (
        outside_counter.update(
            outside_state,
            second,
            _frame_input(1),
            inside_roi=False,
            crossed_direction=outside_direction,
        ).counted_event
        is None
    )


def test_track_counter_without_line_preserves_polygon_counting() -> None:
    profile = CameraProfile(
        camera_id="test",
        polygon=PolygonZoneConfig(points=[[0, 0], [100, 0], [100, 100], [0, 100]]),
        count_line=None,
    )
    counter = TrackCounter(profile)
    state = MutableTrackState(track_id=1, first_frame_index=0, last_frame_index=0)

    result = counter.update(
        state,
        _observation(1, center_x=50, bottom_y=60),
        _frame_input(0),
        inside_roi=True,
        crossed_direction=None,
    )

    assert state.counted is True
    assert result.counted_event is None
    assert result.crossed_line is False
