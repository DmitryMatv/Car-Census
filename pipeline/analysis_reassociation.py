from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from config import TrackerConfig
from pipeline.analysis_diagnostics import AnalysisDiagnostics

if TYPE_CHECKING:
    from pipeline.analysis_tracking import TrackObservation


@dataclass(frozen=True, slots=True)
class StaleReassociationResult:
    observations: list[TrackObservation]
    dropped_track_ids: set[int]


class StaleReassociationRejector:
    def __init__(
        self,
        *,
        tracker_config: TrackerConfig,
        diagnostics: AnalysisDiagnostics,
    ) -> None:
        self._max_gap_seconds = tracker_config.max_reassociation_gap_seconds
        self._diagnostics = diagnostics
        self._last_observation_time_by_track: dict[int, float] = {}
        self._retired_track_ids: set[int] = set()

    def reject(
        self,
        observations: list[TrackObservation],
        timestamp_seconds: float,
    ) -> StaleReassociationResult:
        if self._max_gap_seconds is None:
            return StaleReassociationResult(
                observations=observations,
                dropped_track_ids=set(),
            )

        accepted: list[TrackObservation] = []
        dropped_track_ids: set[int] = set()
        for observation in observations:
            track_id = observation.track_id
            if track_id < 0:
                accepted.append(observation)
                continue

            if track_id in self._retired_track_ids:
                self._diagnostics.stale_reassociation_observations_suppressed += 1
                dropped_track_ids.add(track_id)
                continue

            previous_timestamp = self._last_observation_time_by_track.get(track_id)
            if (
                previous_timestamp is not None
                and timestamp_seconds - previous_timestamp > self._max_gap_seconds
            ):
                self._retired_track_ids.add(track_id)
                self._diagnostics.stale_reassociation_observations_suppressed += 1
                self._diagnostics.stale_reassociation_track_ids_dropped += 1
                dropped_track_ids.add(track_id)
                continue

            self._last_observation_time_by_track[track_id] = timestamp_seconds
            accepted.append(observation)

        return StaleReassociationResult(
            observations=accepted,
            dropped_track_ids=dropped_track_ids,
        )
