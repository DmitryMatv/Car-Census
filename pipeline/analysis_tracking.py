from __future__ import annotations

from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field

import cv2
import supervision as sv

from config import AppConfig, CameraProfile
from models import (
    BBox,
    CountEvent,
    CropCandidate,
    Detection,
    FrameRecord,
    TrackedObject,
)
from pipeline.analysis_crops import CropCandidateSelector
from pipeline.analysis_diagnostics import AnalysisDiagnostics
from pipeline.analysis_edges import EdgeSuppression
from roi.geometry import line_crossing_direction, point_in_polygon


@dataclass(slots=True)
class MutableTrackState:
    track_id: int
    first_frame_index: int
    last_frame_index: int
    vehicle_index: int | None = None
    frames_seen: int = 0
    min_box_height_px: float | None = None
    max_box_height_px: float = 0.0
    previous_bottom_center: tuple[float, float] | None = None
    counted: bool = False
    count_event: CountEvent | None = None
    candidates: list[CropCandidate] = field(default_factory=list)
    last_candidate_time: float | None = None


@dataclass(frozen=True, slots=True)
class FrameTrackingInput:
    frame_index: int
    timestamp_seconds: float
    frame: cv2.typing.MatLike
    roi_frame: cv2.typing.MatLike
    roi_offset: tuple[int, int]
    detections: list[Detection]


@dataclass(slots=True)
class TrackUpdateResult:
    frame_record: FrameRecord
    counted_events: list[CountEvent] = field(default_factory=list)


class TrackStateUpdater:
    def __init__(
        self,
        *,
        config: AppConfig,
        profile: CameraProfile,
        crop_selector: CropCandidateSelector,
        edge_suppression: EdgeSuppression,
        diagnostics: AnalysisDiagnostics,
    ) -> None:
        self.config = config
        self.profile = profile
        self.crop_selector = crop_selector
        self.edge_suppression = edge_suppression
        self.diagnostics = diagnostics
        self._track_states: dict[int, MutableTrackState] = {}
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

    @property
    def track_states(self) -> dict[int, MutableTrackState]:
        return self._track_states

    def process_tracker_outputs(
        self,
        *,
        tracked: sv.Detections,
        frame_input: FrameTrackingInput,
        edge_detection_bboxes: SequenceABC[BBox],
    ) -> TrackUpdateResult:
        self.diagnostics.tracker_outputs += len(tracked.xyxy)
        frame_tracks: list[TrackedObject] = []
        counted_events: list[CountEvent] = []
        tracker_ids = (
            tracked.tracker_id.tolist() if tracked.tracker_id is not None else []
        )
        class_ids = (
            tracked.class_id.tolist()
            if tracked.class_id is not None
            else [-1] * len(tracked.xyxy)
        )
        class_names = tracked.data.get("class_name", []) if tracked.data else []
        confidences = (
            tracked.confidence.tolist()
            if tracked.confidence is not None
            else [0.0] * len(tracked.xyxy)
        )
        self.diagnostics.tracker_confidence_values.extend(
            float(value) for value in confidences
        )

        for index, xyxy in enumerate(tracked.xyxy.tolist()):
            track_id = int(tracker_ids[index]) if index < len(tracker_ids) else -1
            bbox = BBox(x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3])
            touches_suppression_edge = (
                self.edge_suppression.should_skip_track_observation(
                    bbox=bbox,
                    edge_detection_bboxes=edge_detection_bboxes,
                    frame_shape=frame_input.frame.shape,
                    roi_shape=frame_input.roi_frame.shape,
                    roi_offset=frame_input.roi_offset,
                )
            )
            if track_id < 0:
                if touches_suppression_edge:
                    self.diagnostics.edge_observations_skipped += 1
                continue
            state = self._track_states.get(track_id)
            if touches_suppression_edge:
                self.diagnostics.edge_observations_skipped += 1
                self.diagnostics.tracks_discarded_edge_contact += 1
                continue

            bottom_center = bbox.bottom_center
            self.diagnostics.tracker_box_height_values.append(bbox.height)
            inside_roi = point_in_polygon(bottom_center, self.profile.polygon.points)
            confidence = float(confidences[index])
            class_id = int(class_ids[index]) if index < len(class_ids) else None
            class_name = str(class_names[index]) if index < len(class_names) else None

            if state is None:
                state = MutableTrackState(
                    track_id=track_id,
                    first_frame_index=frame_input.frame_index,
                    last_frame_index=frame_input.frame_index,
                )
                self._track_states[track_id] = state

            crossed_direction = None
            if self._count_line is None:
                if inside_roi and not state.counted:
                    state.counted = True
            elif state.previous_bottom_center is not None:
                assert self._line_start is not None
                assert self._line_end is not None
                crossed_direction = line_crossing_direction(
                    previous_point=state.previous_bottom_center,
                    current_point=bottom_center,
                    line_start=self._line_start,
                    line_end=self._line_end,
                )

            crossed_line = False
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
                    track_id=track_id,
                    frame_index=frame_input.frame_index,
                    timestamp_seconds=frame_input.timestamp_seconds,
                    direction=crossed_direction,
                )
                counted_events.append(state.count_event)
                crossed_line = True

            state.frames_seen += 1
            state.last_frame_index = frame_input.frame_index
            state.min_box_height_px = (
                bbox.height
                if state.min_box_height_px is None
                else min(state.min_box_height_px, bbox.height)
            )
            state.max_box_height_px = max(state.max_box_height_px, bbox.height)
            state.previous_bottom_center = bottom_center

            self.crop_selector.maybe_save_candidate(
                track_state=state,
                frame=frame_input.frame,
                bbox=bbox,
                frame_index=frame_input.frame_index,
                timestamp_seconds=frame_input.timestamp_seconds,
            )

            render_bbox = self.crop_selector.render_bbox_for_track(
                bbox, frame_input.frame.shape
            )
            tracked_object = TrackedObject(
                track_id=track_id,
                vehicle_index=None,
                frame_index=frame_input.frame_index,
                timestamp_seconds=frame_input.timestamp_seconds,
                bbox=render_bbox,
                confidence=confidence,
                class_id=class_id,
                class_name=class_name,
                centroid=render_bbox.center,
                bottom_center=render_bbox.bottom_center,
                inside_roi=inside_roi,
                counted=state.counted,
                crossed_line=crossed_line,
            )
            frame_tracks.append(tracked_object)

        return TrackUpdateResult(
            frame_record=FrameRecord(
                frame_index=frame_input.frame_index,
                timestamp_seconds=frame_input.timestamp_seconds,
                tracks=frame_tracks,
            ),
            counted_events=counted_events,
        )

    def sorted_track_states(self) -> list[MutableTrackState]:
        return sorted(
            self._track_states.values(),
            key=lambda item: (item.first_frame_index, item.track_id),
        )
