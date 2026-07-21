from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import supervision as sv

from config import CameraProfile
from models import CountEvent

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
        self._line_zone: sv.LineZone | None = None
        if self._count_line is not None:
            self._line_zone = sv.LineZone(
                start=sv.Point(
                    x=self._count_line.start[0],
                    y=self._count_line.start[1],
                ),
                end=sv.Point(
                    x=self._count_line.end[0],
                    y=self._count_line.end[1],
                ),
                triggering_anchors=[sv.Position.BOTTOM_CENTER],
                minimum_crossing_threshold=1,
            )

    def crossing_directions(
        self, observations: Sequence[TrackObservation]
    ) -> dict[int, str]:
        if self._line_zone is None or not observations:
            return {}

        detections = sv.Detections(
            xyxy=np.asarray(
                [
                    [
                        observation.bbox.x1,
                        observation.bbox.y1,
                        observation.bbox.x2,
                        observation.bbox.y2,
                    ]
                    for observation in observations
                ],
                dtype=np.float32,
            ),
            tracker_id=np.asarray(
                [observation.track_id for observation in observations],
                dtype=np.int32,
            ),
        )
        crossed_in, crossed_out = self._line_zone.trigger(detections)
        directions: dict[int, str] = {}
        for index, observation in enumerate(observations):
            if bool(crossed_out[index]):
                directions[observation.track_id] = "A_TO_B"
            elif bool(crossed_in[index]):
                directions[observation.track_id] = "B_TO_A"
        return directions

    def update(
        self,
        state: MutableTrackState,
        observation: TrackObservation,
        frame_input: FrameTrackingInput,
        inside_roi: bool,
        crossed_direction: str | None,
    ) -> CountUpdate:
        if self._count_line is None:
            if inside_roi and not state.counted:
                state.counted = True
            return CountUpdate(counted_event=None, crossed_line=False)

        if (
            crossed_direction is not None
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
