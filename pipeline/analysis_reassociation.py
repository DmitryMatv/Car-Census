from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from config import TrackerConfig
from pipeline.analysis_diagnostics import AnalysisDiagnostics
from roi.transform import ViewTransformer

if TYPE_CHECKING:
    from pipeline.analysis_tracking import TrackObservation


@dataclass(frozen=True, slots=True)
class StaleReassociationResult:
    observations: list[TrackObservation]
    dropped_track_ids: set[int]


class StaleReassociationRejector:
    """Retires tracker IDs that return after an implausibly long absence.

    A returning observation is accepted while its gap stays within
    ``max_reassociation_gap_seconds``. Beyond that, the ID is retired unless
    a calibrated :class:`ViewTransformer` proves the move physically
    plausible in world space: the displacement between the last seen road
    position and the returning one must satisfy

        distance_m <= min(world_reassociation_max_distance_m,
                          world_reassociation_max_speed_mps * gap_seconds)

    and the gap itself must stay within ``world_reassociation_max_gap_seconds``.
    Without a homography the legacy strict behavior applies unchanged.
    """

    def __init__(
        self,
        *,
        tracker_config: TrackerConfig,
        diagnostics: AnalysisDiagnostics,
        view_transformer: ViewTransformer | None = None,
    ) -> None:
        self._max_gap_seconds = tracker_config.max_reassociation_gap_seconds
        self._world_enabled = (
            tracker_config.world_reassociation_enabled
            and view_transformer is not None
        )
        self._world_max_gap_seconds = (
            tracker_config.world_reassociation_max_gap_seconds
        )
        self._world_max_speed_mps = (
            tracker_config.world_reassociation_max_speed_mps
        )
        self._world_max_distance_m = (
            tracker_config.world_reassociation_max_distance_m
        )
        self._view_transformer = view_transformer
        self._diagnostics = diagnostics
        self._last_observation_time_by_track: dict[int, float] = {}
        self._world_position_by_track: dict[int, tuple[float, float]] = {}
        self._retired_track_ids: set[int] = set()
        self._gate_rescued_track_ids: set[int] = set()

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
            if previous_timestamp is not None:
                gap_seconds = timestamp_seconds - previous_timestamp
                if gap_seconds > self._max_gap_seconds:
                    if self._is_world_plausible(observation, gap_seconds):
                        self._record_gate_acceptance(track_id)
                    else:
                        self._retire(
                            track_id,
                            dropped_track_ids,
                        )
                        continue

            self._accept(track_id, observation, timestamp_seconds)
            accepted.append(observation)

        return StaleReassociationResult(
            observations=accepted,
            dropped_track_ids=dropped_track_ids,
        )

    def _is_world_plausible(
        self,
        observation: TrackObservation,
        gap_seconds: float,
    ) -> bool:
        view_transformer = self._view_transformer
        if (
            not self._world_enabled
            or view_transformer is None
            or gap_seconds > self._world_max_gap_seconds
        ):
            return False
        last_position = self._world_position_by_track.get(observation.track_id)
        if last_position is None:
            return False
        distance_m = view_transformer.distance_between(
            last_position, observation.bottom_center
        )
        if not math.isfinite(distance_m):
            return False
        speed_limit_m = self._world_max_speed_mps * gap_seconds
        allowed_distance_m = min(self._world_max_distance_m, speed_limit_m)
        return distance_m <= allowed_distance_m

    def _record_gate_acceptance(self, track_id: int) -> None:
        self._diagnostics.world_reassociation_observations_accepted += 1
        if track_id not in self._gate_rescued_track_ids:
            self._gate_rescued_track_ids.add(track_id)
            self._diagnostics.world_reassociation_tracks_retained += 1

    def _retire(
        self,
        track_id: int,
        dropped_track_ids: set[int],
    ) -> None:
        self._retired_track_ids.add(track_id)
        self._diagnostics.stale_reassociation_observations_suppressed += 1
        self._diagnostics.stale_reassociation_track_ids_dropped += 1
        dropped_track_ids.add(track_id)

    def _accept(
        self,
        track_id: int,
        observation: TrackObservation,
        timestamp_seconds: float,
    ) -> None:
        self._last_observation_time_by_track[track_id] = timestamp_seconds
        view_transformer = self._view_transformer
        if self._world_enabled and view_transformer is not None:
            self._world_position_by_track[track_id] = (
                view_transformer.transform_point(observation.bottom_center)
            )
