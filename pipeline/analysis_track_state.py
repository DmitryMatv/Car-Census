from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from models import CountEvent, CropCandidate

if TYPE_CHECKING:
    from pipeline.analysis_tracking import FrameTrackingInput, TrackObservation


@dataclass(slots=True)
class MutableTrackState:
    track_id: int
    first_frame_index: int
    last_frame_index: int
    vehicle_index: int | None = None
    frames_seen: int = 0
    min_box_width_px: float | None = None
    max_box_width_px: float = 0.0
    min_box_height_px: float | None = None
    max_box_height_px: float = 0.0
    counted: bool = False
    count_event: CountEvent | None = None
    candidates: list[CropCandidate] = field(default_factory=list)
    last_candidate_time: float | None = None


class TrackStateStore:
    def __init__(self) -> None:
        self._track_states: dict[int, MutableTrackState] = {}

    @property
    def track_states(self) -> dict[int, MutableTrackState]:
        return self._track_states

    def get(self, track_id: int) -> MutableTrackState | None:
        return self._track_states.get(track_id)

    def get_or_create(
        self, observation: TrackObservation, frame_input: FrameTrackingInput
    ) -> MutableTrackState:
        state = self._track_states.get(observation.track_id)
        if state is not None:
            return state

        state = MutableTrackState(
            track_id=observation.track_id,
            first_frame_index=frame_input.frame_index,
            last_frame_index=frame_input.frame_index,
        )
        self._track_states[observation.track_id] = state
        return state

    def update_metrics(
        self,
        state: MutableTrackState,
        observation: TrackObservation,
        frame_index: int,
    ) -> None:
        state.frames_seen += 1
        state.last_frame_index = frame_index
        state.min_box_width_px = (
            observation.bbox.width
            if state.min_box_width_px is None
            else min(state.min_box_width_px, observation.bbox.width)
        )
        state.max_box_width_px = max(state.max_box_width_px, observation.bbox.width)
        state.min_box_height_px = (
            observation.bbox.height
            if state.min_box_height_px is None
            else min(state.min_box_height_px, observation.bbox.height)
        )
        state.max_box_height_px = max(state.max_box_height_px, observation.bbox.height)

    def sorted_active_states(self) -> list[MutableTrackState]:
        return sorted(
            self._track_states.values(),
            key=lambda item: (item.first_frame_index, item.track_id),
        )
