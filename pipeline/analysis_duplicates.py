from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from config import TrackerConfig
from pipeline.analysis_diagnostics import AnalysisDiagnostics
from pipeline.analysis_track_state import TrackStateStore
from pipeline.vehicles import discard_track_artifacts

if TYPE_CHECKING:
    from pipeline.analysis_tracking import TrackObservation


@dataclass(frozen=True, slots=True)
class DuplicateSuppressionResult:
    observations: list[TrackObservation]
    dropped_track_ids: set[int]


class DuplicateTrackSuppressor:
    def __init__(
        self,
        *,
        tracker_config: TrackerConfig,
        track_store: TrackStateStore,
        crops_dir: Path,
        diagnostics: AnalysisDiagnostics,
    ) -> None:
        self._tracker_config = tracker_config
        self._track_store = track_store
        self._crops_dir = crops_dir
        self._diagnostics = diagnostics
        self._suppressed_track_ids: set[int] = set()

    def suppress(
        self, observations: list[TrackObservation]
    ) -> DuplicateSuppressionResult:
        if not self._tracker_config.suppress_duplicate_tracks:
            return DuplicateSuppressionResult(
                observations=observations,
                dropped_track_ids=set(),
            )

        suppressed_this_frame = self._already_suppressed_ids(observations)
        newly_suppressed = self._new_duplicate_ids_to_suppress(
            observations, suppressed_this_frame
        )
        suppressed_this_frame.update(newly_suppressed)

        if not suppressed_this_frame:
            return DuplicateSuppressionResult(
                observations=observations,
                dropped_track_ids=set(),
            )

        self._diagnostics.duplicate_track_observations_suppressed += len(
            suppressed_this_frame
        )
        return DuplicateSuppressionResult(
            observations=[
                observation
                for observation in observations
                if observation.track_id not in suppressed_this_frame
            ],
            dropped_track_ids=suppressed_this_frame,
        )

    def _already_suppressed_ids(self, observations: list[TrackObservation]) -> set[int]:
        return {
            observation.track_id
            for observation in observations
            if observation.track_id in self._suppressed_track_ids
        }

    def _new_duplicate_ids_to_suppress(
        self,
        observations: list[TrackObservation],
        suppressed_this_frame: set[int],
    ) -> set[int]:
        newly_suppressed: set[int] = set()
        for left, right in self._iter_candidate_pairs(
            observations, suppressed_this_frame
        ):
            loser = self._select_loser(left, right)
            if loser is None:
                continue

            self._suppress_track(loser.track_id)
            newly_suppressed.add(loser.track_id)
            suppressed_this_frame.add(loser.track_id)
        return newly_suppressed

    def _iter_candidate_pairs(
        self,
        observations: list[TrackObservation],
        suppressed_ids: set[int],
    ) -> Iterator[tuple[TrackObservation, TrackObservation]]:
        for left_index, left in enumerate(observations):
            if left.track_id in suppressed_ids:
                continue
            for right in observations[left_index + 1 :]:
                if right.track_id in suppressed_ids:
                    continue
                if self._is_duplicate_pair(left, right):
                    yield left, right
                    if left.track_id in suppressed_ids:
                        break

    def _is_duplicate_pair(
        self, left: TrackObservation, right: TrackObservation
    ) -> bool:
        if left.track_id == right.track_id or left.track_id < 0 or right.track_id < 0:
            return False

        if left.bbox.iou(right.bbox) >= (
            self._tracker_config.duplicate_track_iou_threshold
        ):
            return True

        coverage = left.bbox.smaller_coverage(right.bbox)
        if coverage < self._tracker_config.duplicate_track_containment_threshold:
            return False
        if (
            left.bbox.area_ratio(right.bbox)
            < self._tracker_config.duplicate_track_min_area_ratio
        ):
            return False
        larger_area = max(left.bbox.area, right.bbox.area)
        max_center_distance = (
            self._tracker_config.duplicate_track_center_distance_ratio
            * math.sqrt(larger_area)
        )
        return left.bbox.center_distance(right.bbox) <= max_center_distance

    def _select_loser(
        self, left: TrackObservation, right: TrackObservation
    ) -> TrackObservation | None:
        left_score = self._survivor_score(left)
        right_score = self._survivor_score(right)
        loser = right if left_score >= right_score else left
        state = self._track_store.get(loser.track_id)
        if state is not None and state.count_event is not None:
            self._diagnostics.duplicate_track_suppression_blocked_counted += 1
            return None
        return loser

    def _survivor_score(
        self, observation: TrackObservation
    ) -> tuple[int, int, int, int, float, float, float, int]:
        state = self._track_store.get(observation.track_id)
        return (
            1 if state is not None else 0,
            1 if state is not None and state.counted else 0,
            state.frames_seen if state is not None else 0,
            1 if state is not None and state.candidates else 0,
            state.max_box_width_px if state is not None else 0.0,
            observation.confidence,
            observation.bbox.area,
            -observation.track_id,
        )

    def _suppress_track(self, track_id: int) -> None:
        state = self._track_store.get(track_id)
        if state is not None:
            discard_track_artifacts(state, self._crops_dir)
            state.candidates = []
            state.suppressed_duplicate = True
        if track_id not in self._suppressed_track_ids:
            self._suppressed_track_ids.add(track_id)
            self._diagnostics.duplicate_track_ids_dropped += 1
