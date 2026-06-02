from __future__ import annotations

from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field

import cv2
import supervision as sv

from config import AppConfig, CameraProfile
from models import (
    BBox,
    CountEvent,
    FrameRecord,
    TrackedObject,
)
from pipeline.analysis_counting import TrackCounter
from pipeline.analysis_crops import CropCandidateSelector
from pipeline.analysis_diagnostics import AnalysisDiagnostics
from pipeline.analysis_duplicates import DuplicateTrackSuppressor
from pipeline.analysis_edges import EdgeSuppression
from pipeline.analysis_track_state import MutableTrackState, TrackStateStore
from roi.geometry import point_in_polygon


@dataclass(frozen=True, slots=True)
class FrameTrackingInput:
    frame_index: int
    timestamp_seconds: float
    frame: cv2.typing.MatLike
    roi_frame: cv2.typing.MatLike
    roi_offset: tuple[int, int]
    detections: sv.Detections


@dataclass(slots=True)
class TrackUpdateResult:
    frame_record: FrameRecord
    counted_events: list[CountEvent] = field(default_factory=list)
    duplicate_track_ids_dropped: set[int] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class TrackObservation:
    index: int
    track_id: int
    bbox: BBox
    confidence: float
    class_id: int | None
    class_name: str | None
    bottom_center: tuple[float, float]


def _tracker_confidences(tracked: sv.Detections) -> list[float]:
    if tracked.confidence is None:
        return [0.0] * len(tracked.xyxy)
    return [float(value) for value in tracked.confidence.tolist()]


def _iter_track_observations(tracked: sv.Detections) -> list[TrackObservation]:
    tracker_ids = tracked.tracker_id.tolist() if tracked.tracker_id is not None else []
    class_ids = tracked.class_id.tolist() if tracked.class_id is not None else []
    class_names = tracked.data.get("class_name", []) if tracked.data else []
    confidences = _tracker_confidences(tracked)

    observations: list[TrackObservation] = []
    for index, xyxy in enumerate(tracked.xyxy.tolist()):
        track_id = int(tracker_ids[index]) if index < len(tracker_ids) else -1
        bbox = BBox(x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3])
        observations.append(
            TrackObservation(
                index=index,
                track_id=track_id,
                bbox=bbox,
                confidence=confidences[index],
                class_id=int(class_ids[index]) if index < len(class_ids) else None,
                class_name=(
                    str(class_names[index]) if index < len(class_names) else None
                ),
                bottom_center=bbox.bottom_center,
            )
        )
    return observations


class TrackerObservationReader:
    def __init__(self, diagnostics: AnalysisDiagnostics) -> None:
        self._diagnostics = diagnostics

    def read(self, tracked: sv.Detections) -> list[TrackObservation]:
        self._diagnostics.tracker_outputs += len(tracked.xyxy)
        confidences = _tracker_confidences(tracked)
        self._diagnostics.tracker_confidence_histogram.extend(
            value for value in confidences
        )
        return _iter_track_observations(tracked)

    def record_box_width(self, observation: TrackObservation) -> None:
        self._diagnostics.tracker_box_width_histogram.observe(observation.bbox.width)


class EdgeObservationFilter:
    def __init__(
        self,
        *,
        edge_suppression: EdgeSuppression,
        diagnostics: AnalysisDiagnostics,
    ) -> None:
        self._edge_suppression = edge_suppression
        self._diagnostics = diagnostics

    def filter(
        self,
        observations: list[TrackObservation],
        frame_input: FrameTrackingInput,
        edge_detection_bboxes: SequenceABC[BBox],
    ) -> list[TrackObservation]:
        filtered: list[TrackObservation] = []
        for observation in observations:
            touches_suppression_edge = self._should_skip_observation(
                observation, frame_input, edge_detection_bboxes
            )
            if observation.track_id < 0:
                if touches_suppression_edge:
                    self._diagnostics.edge_observations_skipped += 1
                continue
            if touches_suppression_edge:
                self._diagnostics.edge_observations_skipped += 1
                self._diagnostics.tracks_discarded_edge_contact += 1
                continue
            filtered.append(observation)
        return filtered

    def _should_skip_observation(
        self,
        observation: TrackObservation,
        frame_input: FrameTrackingInput,
        edge_detection_bboxes: SequenceABC[BBox],
    ) -> bool:
        return self._edge_suppression.should_skip_track_observation(
            bbox=observation.bbox,
            edge_detection_bboxes=edge_detection_bboxes,
            frame_shape=frame_input.frame.shape,
            roi_shape=frame_input.roi_frame.shape,
            roi_offset=frame_input.roi_offset,
        )


class TrackedObjectBuilder:
    def __init__(
        self,
        *,
        profile: CameraProfile,
        crop_selector: CropCandidateSelector,
    ) -> None:
        self._profile = profile
        self._crop_selector = crop_selector

    def inside_roi(self, observation: TrackObservation) -> bool:
        return point_in_polygon(observation.bottom_center, self._profile.polygon.points)

    def update_crop_state(
        self,
        state: MutableTrackState,
        observation: TrackObservation,
        frame_input: FrameTrackingInput,
    ) -> None:
        self._crop_selector.maybe_save_candidate(
            track_state=state,
            frame=frame_input.frame,
            bbox=observation.bbox,
            frame_index=frame_input.frame_index,
            timestamp_seconds=frame_input.timestamp_seconds,
        )

    def build(
        self,
        state: MutableTrackState,
        observation: TrackObservation,
        frame_input: FrameTrackingInput,
        inside_roi: bool,
        crossed_line: bool,
    ) -> TrackedObject:
        render_bbox = self._crop_selector.render_bbox_for_track(
            observation.bbox, frame_input.frame.shape
        )
        return TrackedObject(
            track_id=observation.track_id,
            vehicle_index=None,
            frame_index=frame_input.frame_index,
            timestamp_seconds=frame_input.timestamp_seconds,
            bbox=render_bbox,
            confidence=observation.confidence,
            class_id=observation.class_id,
            class_name=observation.class_name,
            centroid=render_bbox.center,
            bottom_center=render_bbox.bottom_center,
            inside_roi=inside_roi,
            counted=state.counted,
            crossed_line=crossed_line,
        )


@dataclass(frozen=True, slots=True)
class TrackStateUpdaterComponents:
    observation_reader: TrackerObservationReader
    edge_filter: EdgeObservationFilter
    duplicate_suppressor: DuplicateTrackSuppressor
    track_store: TrackStateStore
    track_counter: TrackCounter
    tracked_object_builder: TrackedObjectBuilder


def _empty_tracked_objects() -> list[TrackedObject]:
    return []


def _empty_count_events() -> list[CountEvent]:
    return []


class TrackStateUpdater:
    def __init__(
        self,
        *,
        components: TrackStateUpdaterComponents,
    ) -> None:
        self._components = components

    @property
    def track_states(self) -> dict[int, MutableTrackState]:
        return self._components.track_store.track_states

    def process_tracker_outputs(
        self,
        *,
        tracked: sv.Detections,
        frame_input: FrameTrackingInput,
        edge_detection_bboxes: SequenceABC[BBox],
    ) -> TrackUpdateResult:
        frame_tracks = _empty_tracked_objects()
        counted_events = _empty_count_events()

        observations = self._components.edge_filter.filter(
            self._components.observation_reader.read(tracked),
            frame_input,
            edge_detection_bboxes,
        )
        duplicate_result = self._components.duplicate_suppressor.suppress(observations)

        for observation in duplicate_result.observations:
            self._components.observation_reader.record_box_width(observation)
            inside_roi = self._components.tracked_object_builder.inside_roi(observation)
            state = self._components.track_store.get_or_create(observation, frame_input)
            count_update = self._components.track_counter.update(
                state, observation, frame_input, inside_roi
            )
            if count_update.counted_event is not None:
                counted_events.append(count_update.counted_event)

            self._components.track_store.update_metrics(
                state, observation, frame_input.frame_index
            )
            self._components.tracked_object_builder.update_crop_state(
                state, observation, frame_input
            )
            frame_tracks.append(
                self._components.tracked_object_builder.build(
                    state,
                    observation,
                    frame_input,
                    inside_roi,
                    count_update.crossed_line,
                )
            )

        return TrackUpdateResult(
            frame_record=FrameRecord(
                frame_index=frame_input.frame_index,
                timestamp_seconds=frame_input.timestamp_seconds,
                tracks=frame_tracks,
            ),
            counted_events=counted_events,
            duplicate_track_ids_dropped=duplicate_result.dropped_track_ids,
        )

    def sorted_track_states(self) -> list[MutableTrackState]:
        return self._components.track_store.sorted_active_states()


def build_track_state_updater(
    *,
    config: AppConfig,
    profile: CameraProfile,
    crop_selector: CropCandidateSelector,
    edge_suppression: EdgeSuppression,
    diagnostics: AnalysisDiagnostics,
) -> TrackStateUpdater:
    track_store = TrackStateStore()
    return TrackStateUpdater(
        components=TrackStateUpdaterComponents(
            observation_reader=TrackerObservationReader(diagnostics),
            edge_filter=EdgeObservationFilter(
                edge_suppression=edge_suppression,
                diagnostics=diagnostics,
            ),
            duplicate_suppressor=DuplicateTrackSuppressor(
                tracker_config=config.tracker,
                track_store=track_store,
                crops_dir=crop_selector.store.crops_dir,
                diagnostics=diagnostics,
            ),
            track_store=track_store,
            track_counter=TrackCounter(profile),
            tracked_object_builder=TrackedObjectBuilder(
                profile=profile,
                crop_selector=crop_selector,
            ),
        ),
    )
