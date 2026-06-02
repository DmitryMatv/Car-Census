from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from config import CameraProfile
from models import CountEvent
from roi.geometry import line_crossing_direction

if TYPE_CHECKING:
    from pipeline.analysis_track_state import MutableTrackState
    from pipeline.analysis_tracking import FrameTrackingInput, TrackObservation


@dataclass(frozen=True, slots=True)
class CountUpdate:
    counted_event: CountEvent | None
    crossed_line: bool


class TrackCounter:
    def __init__(self, profile: CameraProfile) -> None:
        self._count_line = profile.count_line
        self._line_start: tuple[float, float] | None = None
        self._line_end: tuple[float, float] | None = None
        if self._count_line is not None:
            self._line_start = (
                float(self._count_line.start[0]),
                float(self._count_line.start[1]),
            )
            self._line_end = (
                float(self._count_line.end[0]),
                float(self._count_line.end[1]),
            )

    def update(
        self,
        state: MutableTrackState,
        observation: TrackObservation,
        frame_input: FrameTrackingInput,
        inside_roi: bool,
    ) -> CountUpdate:
        crossed_direction = None
        if self._count_line is None:
            if inside_roi and not state.counted:
                state.counted = True
        elif state.previous_bottom_center is not None:
            assert self._line_start is not None
            assert self._line_end is not None
            crossed_direction = line_crossing_direction(
                previous_point=state.previous_bottom_center,
                current_point=observation.bottom_center,
                line_start=self._line_start,
                line_end=self._line_end,
            )

        if (
            self._count_line is not None
            and crossed_direction is not None
            and (
                self._count_line.direction == "BOTH"
                or crossed_direction == self._count_line.direction
            )
            and not state.counted
            and inside_roi
        ):
            state.counted = True
            state.count_event = CountEvent(
                track_id=observation.track_id,
                frame_index=frame_input.frame_index,
                timestamp_seconds=frame_input.timestamp_seconds,
                direction=crossed_direction,
            )
            return CountUpdate(counted_event=state.count_event, crossed_line=True)

        return CountUpdate(counted_event=None, crossed_line=False)
