"""Offline track linking.

Merges track fragments of the same physical vehicle into one canonical
``vehicle_index`` after analysis has finished. The failures it targets,
quantified on reference runs of the static motorway camera:

- Birth flicker: a vehicle's first 1-2 samples spawn throwaway tracks
  before BoT-SORT's Kalman velocity converges (chains such as
  61 -> 62 -> 63 -> 64 where 61-63 hold a single observation each).
- Mid-life splits: a missed detection or two in the middle of a pass
  continues the vehicle under a new track id.

The stage never mutates raw artifacts. It writes:

- ``analysis/linked_tracks.jsonl``: derived track summaries with the
  canonical ``vehicle_index`` assigned per merge group. Consumers read
  tracks through ``RunStore.tracks_effective``, which prefers this file
  when present.
- ``analysis/links.json``: audit trail with per-merge geometry evidence.

Matching is geometry-only, in ground-plane meters via the calibrated
homography: mutual-best endpoint pairing under gap, displacement, lateral,
implied-speed, and motion-direction gates, then transitive union. Appearance
confirmation for longer gaps is future work, so ``max_gap_seconds`` must
stay well under the local same-lane following time; in platoon traffic the
next car reaches a predecessor's last position at the same road speed and
geometry alone cannot separate the two.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Sequence

from config import AppConfig, CameraProfile, LinkingConfig
from models import FrameRecord, TrackSummary
from roi.transform import ViewTransformer
from storage.json_artifacts import write_json
from storage.run_artifacts import JsonlModelFile
from storage.run_store import RunStore

logger = logging.getLogger(__name__)

_STATUS_LINKED = "linked"
_STATUS_NO_HOMOGRAPHY = "skipped_no_homography"
_NO_HOMOGRAPHY_NOTE = (
    "Camera profile has no calibrated homography; linking needs ground-plane "
    "geometry and stayed inert."
)


@dataclass(frozen=True, slots=True)
class _Observation:
    frame_index: int
    timestamp_seconds: float
    bottom_center: tuple[float, float]


@dataclass(frozen=True, slots=True)
class _TrackEndpoints:
    track_id: int
    vehicle_index: int | None
    first_frame_index: int
    last_frame_index: int
    first_timestamp_seconds: float
    last_timestamp_seconds: float
    first_world: tuple[float, float]
    last_world: tuple[float, float]
    # World velocity (m/s) over the predecessor's last observations; used by
    # the direction-continuity gate.
    recent_velocity_mps: tuple[float, float] | None


@dataclass(frozen=True, slots=True)
class _CandidatePair:
    predecessor: _TrackEndpoints
    successor: _TrackEndpoints
    gap_seconds: float
    distance_m: float
    lateral_offset_m: float
    implied_speed_mps: float

    @property
    def sort_key(self) -> tuple[float, float, float]:
        return (self.gap_seconds, self.lateral_offset_m, self.distance_m)

    def to_payload(self) -> dict[str, object]:
        return {
            "predecessor_track_id": self.predecessor.track_id,
            "successor_track_id": self.successor.track_id,
            "gap_seconds": round(self.gap_seconds, 4),
            "distance_m": round(self.distance_m, 3),
            "implied_speed_mps": round(self.implied_speed_mps, 3),
            "lateral_offset_m": round(self.lateral_offset_m, 3),
        }


@dataclass(frozen=True, slots=True)
class _MergeGroup:
    canonical_vehicle_index: int | None
    member_track_ids: tuple[int, ...]
    evidence: tuple[_CandidatePair, ...]


@dataclass(frozen=True, slots=True)
class LinkResult:
    status: str
    input_track_count: int
    input_vehicle_count: int
    output_vehicle_count: int
    merge_group_count: int
    merged_track_count: int
    ambiguous_pair_count: int


def _collect_observations(
    records: Sequence[FrameRecord],
) -> dict[int, list[_Observation]]:
    observations: dict[int, list[_Observation]] = {}
    for record in records:
        for track in record.tracks:
            observations.setdefault(track.track_id, []).append(
                _Observation(
                    frame_index=track.frame_index,
                    timestamp_seconds=track.timestamp_seconds,
                    bottom_center=track.bottom_center,
                )
            )
    for track_observations in observations.values():
        track_observations.sort(key=lambda item: item.frame_index)
    return observations


def _recent_velocity_mps(
    track_observations: list[_Observation],
) -> tuple[float, float] | None:
    """World velocity over the last (up to) three observations."""
    window = track_observations[-3:]
    if len(window) < 2:
        return None
    span = window[-1].timestamp_seconds - window[0].timestamp_seconds
    if span <= 0.0:
        return None
    first = window[0].bottom_center
    last = window[-1].bottom_center
    return (
        (last[0] - first[0]) / span,
        (last[1] - first[1]) / span,
    )


def _build_endpoints(
    summaries: Sequence[TrackSummary],
    observations: dict[int, list[_Observation]],
    transformer: ViewTransformer,
) -> list[_TrackEndpoints]:
    endpoints: list[_TrackEndpoints] = []
    for summary in summaries:
        track_observations = observations.get(summary.track_id)
        if not track_observations:
            continue
        first = track_observations[0]
        last = track_observations[-1]
        endpoints.append(
            _TrackEndpoints(
                track_id=summary.track_id,
                vehicle_index=summary.vehicle_index,
                first_frame_index=first.frame_index,
                last_frame_index=last.frame_index,
                first_timestamp_seconds=first.timestamp_seconds,
                last_timestamp_seconds=last.timestamp_seconds,
                first_world=transformer.transform_point(first.bottom_center),
                last_world=transformer.transform_point(last.bottom_center),
                recent_velocity_mps=_recent_velocity_mps(track_observations),
            )
        )
    return endpoints


def _passes_gates(
    pair: _CandidatePair,
    config: LinkingConfig,
) -> bool:
    if not (0.0 < pair.gap_seconds <= config.max_gap_seconds):
        return False
    if pair.distance_m > config.max_endpoint_distance_m:
        return False
    if pair.lateral_offset_m > config.max_lateral_offset_m:
        return False
    if not (
        config.min_implied_speed_mps
        <= pair.implied_speed_mps
        <= config.max_implied_speed_mps
    ):
        return False
    if not config.enforce_direction_continuity:
        return True
    displacement = (
        pair.successor.first_world[0] - pair.predecessor.last_world[0],
        pair.successor.first_world[1] - pair.predecessor.last_world[1],
    )
    velocity = pair.predecessor.recent_velocity_mps
    if velocity is None:
        return True
    displacement_magnitude = math.hypot(*displacement)
    speed = math.hypot(*velocity)
    if (
        displacement_magnitude < config.direction_min_displacement_m
        or speed < config.direction_min_speed_mps
    ):
        # Heading is unreliable at this resolution; do not reject on it.
        return True
    return (displacement[0] * velocity[0] + displacement[1] * velocity[1]) > 0.0


def _candidate_pairs(
    endpoints: Sequence[_TrackEndpoints],
    config: LinkingConfig,
) -> list[_CandidatePair]:
    pairs: list[_CandidatePair] = []
    for index, predecessor in enumerate(endpoints):
        for successor in endpoints[index + 1 :]:
            gap = successor.first_timestamp_seconds - predecessor.last_timestamp_seconds
            if gap <= 0.0:
                continue
            displacement = (
                successor.first_world[0] - predecessor.last_world[0],
                successor.first_world[1] - predecessor.last_world[1],
            )
            distance = math.hypot(*displacement)
            if not math.isfinite(distance) or distance <= 0.0:
                continue
            pair = _CandidatePair(
                predecessor=predecessor,
                successor=successor,
                gap_seconds=gap,
                distance_m=distance,
                lateral_offset_m=abs(displacement[0]),
                implied_speed_mps=distance / gap,
            )
            if _passes_gates(pair, config):
                pairs.append(pair)
    return pairs


def _resolve_mutual_best(
    pairs: Sequence[_CandidatePair],
) -> tuple[list[_CandidatePair], list[_CandidatePair]]:
    """Keep a pair only when it is the best remaining option for both of its
    endpoints. Returns (accepted, ambiguous); ambiguous pairs passed the
    geometry gates but lost the mutual-best contest, and are recorded in the
    audit instead of being merged."""
    best_by_predecessor: dict[int, _CandidatePair] = {}
    best_by_successor: dict[int, _CandidatePair] = {}
    for pair in pairs:
        predecessor_id = pair.predecessor.track_id
        successor_id = pair.successor.track_id
        predecessor_best = best_by_predecessor.get(predecessor_id)
        if predecessor_best is None or pair.sort_key < predecessor_best.sort_key:
            best_by_predecessor[predecessor_id] = pair
        successor_best = best_by_successor.get(successor_id)
        if successor_best is None or pair.sort_key < successor_best.sort_key:
            best_by_successor[successor_id] = pair

    accepted: list[_CandidatePair] = []
    ambiguous: list[_CandidatePair] = []
    for pair in pairs:
        if (
            best_by_predecessor[pair.predecessor.track_id] is pair
            and best_by_successor[pair.successor.track_id] is pair
        ):
            accepted.append(pair)
        else:
            ambiguous.append(pair)
    accepted.sort(key=lambda pair: pair.sort_key)
    return accepted, ambiguous


class _UnionFind:
    def __init__(self, track_ids: Sequence[int]) -> None:
        self._parent: dict[int, int] = {track_id: track_id for track_id in track_ids}

    def find(self, track_id: int) -> int:
        parent = self._parent[track_id]
        if parent != track_id:
            self._parent[track_id] = self.find(parent)
        return self._parent[track_id]

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self._parent[second_root] = first_root


def _build_merge_groups(
    endpoints: Sequence[_TrackEndpoints],
    accepted_pairs: Sequence[_CandidatePair],
) -> list[_MergeGroup]:
    union_find = _UnionFind([endpoint.track_id for endpoint in endpoints])
    for pair in accepted_pairs:
        union_find.union(pair.predecessor.track_id, pair.successor.track_id)

    members_by_root: dict[int, list[_TrackEndpoints]] = {}
    for endpoint in endpoints:
        members_by_root.setdefault(union_find.find(endpoint.track_id), []).append(
            endpoint
        )

    evidence_by_root: dict[int, list[_CandidatePair]] = {}
    for pair in accepted_pairs:
        root = union_find.find(pair.predecessor.track_id)
        evidence_by_root.setdefault(root, []).append(pair)

    groups: list[_MergeGroup] = []
    for members in members_by_root.values():
        if len(members) < 2:
            continue
        ordered = sorted(
            members,
            key=lambda endpoint: (endpoint.first_frame_index, endpoint.track_id),
        )
        component_root = union_find.find(ordered[0].track_id)
        canonical: int | None = None
        for endpoint in ordered:
            if endpoint.vehicle_index is not None:
                canonical = endpoint.vehicle_index
                break
        groups.append(
            _MergeGroup(
                canonical_vehicle_index=canonical,
                member_track_ids=tuple(endpoint.track_id for endpoint in ordered),
                evidence=tuple(evidence_by_root.get(component_root, ())),
            )
        )
    groups.sort(key=lambda group: group.member_track_ids[0])
    return groups


def _vehicle_count(summaries: Sequence[TrackSummary]) -> int:
    return len(
        {
            summary.vehicle_index
            for summary in summaries
            if summary.vehicle_index is not None
        }
    )


def link_analysis_tracks(
    config: AppConfig,
    profile: CameraProfile,
    run_store: RunStore,
) -> LinkResult | None:
    if not config.linking.enabled:
        return None

    summaries = run_store.tracks.read_all()
    records = run_store.frames.read_all(smoothed=False)
    observations = _collect_observations(records)

    transformer: ViewTransformer | None = None
    if profile.homography is not None:
        transformer = ViewTransformer(
            source_points=profile.homography.source_points,
            target_points=profile.homography.target_points,
        )
    if transformer is None:
        write_json(
            run_store.links_path,
            {
                "status": _STATUS_NO_HOMOGRAPHY,
                "note": _NO_HOMOGRAPHY_NOTE,
                "config": config.linking.model_dump(mode="json"),
                "input_track_count": len(summaries),
                "merge_groups": [],
                "rejected_ambiguous_pairs": [],
            },
        )
        return LinkResult(
            status=_STATUS_NO_HOMOGRAPHY,
            input_track_count=len(summaries),
            input_vehicle_count=_vehicle_count(summaries),
            output_vehicle_count=_vehicle_count(summaries),
            merge_group_count=0,
            merged_track_count=0,
            ambiguous_pair_count=0,
        )

    endpoints = _build_endpoints(summaries, observations, transformer)
    candidates = _candidate_pairs(endpoints, config.linking)
    accepted, ambiguous = _resolve_mutual_best(candidates)
    merge_groups = _build_merge_groups(endpoints, accepted)

    canonical_by_track: dict[int, int] = {}
    for group in merge_groups:
        if group.canonical_vehicle_index is None:
            continue
        for track_id in group.member_track_ids:
            canonical_by_track[track_id] = group.canonical_vehicle_index

    linked_summaries = [
        (
            summary.model_copy(
                update={"vehicle_index": canonical_by_track[summary.track_id]}
            )
            if summary.track_id in canonical_by_track
            else summary
        )
        for summary in summaries
    ]
    JsonlModelFile(run_store.linked_tracks_path, TrackSummary).write_all(
        linked_summaries
    )

    write_json(
        run_store.links_path,
        {
            "status": _STATUS_LINKED,
            "config": config.linking.model_dump(mode="json"),
            "input_track_count": len(summaries),
            "input_vehicle_index_count": _vehicle_count(summaries),
            "output_vehicle_index_count": _vehicle_count(linked_summaries),
            "merge_groups": [
                {
                    "canonical_vehicle_index": group.canonical_vehicle_index,
                    "member_track_ids": list(group.member_track_ids),
                    "evidence": [pair.to_payload() for pair in group.evidence],
                }
                for group in merge_groups
            ],
            "rejected_ambiguous_pairs": [pair.to_payload() for pair in ambiguous],
        },
    )

    result = LinkResult(
        status=_STATUS_LINKED,
        input_track_count=len(summaries),
        input_vehicle_count=_vehicle_count(summaries),
        output_vehicle_count=_vehicle_count(linked_summaries),
        merge_group_count=len(merge_groups),
        merged_track_count=len(canonical_by_track),
        ambiguous_pair_count=len(ambiguous),
    )
    logger.info(
        "Track linking: %d merge group(s), %d track(s) re-identified, "
        "%d -> %d vehicle identities (%d ambiguous pair(s) left unmerged)",
        result.merge_group_count,
        result.merged_track_count,
        result.input_vehicle_count,
        result.output_vehicle_count,
        result.ambiguous_pair_count,
    )
    return result
