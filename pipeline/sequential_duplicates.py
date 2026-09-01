from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import AppConfig, CameraProfile
from models import BBox, FrameRecord, MMRResult, TrackedObject, TrackSummary
from roi.transform import ViewTransformer, build_view_transformer
from storage.json_artifacts import write_json
from storage.run_store import RunStore


@dataclass(frozen=True)
class _TrackEndpoints:
    track_id: int
    vehicle_index: int
    first_time: float
    first_bbox: BBox
    previous_time: float | None
    previous_bbox: BBox | None
    last_time: float
    last_bbox: BBox


class _DisjointSet:
    def __init__(self, vehicle_indices: set[int]) -> None:
        self._parent = {index: index for index in vehicle_indices}

    def find(self, item: int) -> int:
        parent = self._parent[item]
        if parent != item:
            parent = self.find(parent)
            self._parent[item] = parent
        return parent

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        canonical = min(left_root, right_root)
        duplicate = max(left_root, right_root)
        self._parent[duplicate] = canonical

    def canonical_map(self) -> dict[int, int]:
        return {item: self.find(item) for item in self._parent}


def _identity_tuple(
    label: MMRResult,
    *,
    require_same_color: bool,
    require_same_generation: bool,
    require_same_variation: bool,
) -> tuple[str, ...]:
    values = [
        label.category,
        label.make,
        label.model,
    ]
    if require_same_generation:
        values.append(label.generation)
    if require_same_variation:
        values.append(label.variation)
    if require_same_color:
        values.append(label.color)
    return tuple("" if value is None else value.strip().casefold() for value in values)


def _labels_match(
    left: MMRResult,
    right: MMRResult,
    *,
    require_same_color: bool,
    require_same_generation: bool,
    require_same_variation: bool,
) -> bool:
    return (
        left.accepted
        and right.accepted
        and _identity_tuple(
            left,
            require_same_color=require_same_color,
            require_same_generation=require_same_generation,
            require_same_variation=require_same_variation,
        )
        == _identity_tuple(
            right,
            require_same_color=require_same_color,
            require_same_generation=require_same_generation,
            require_same_variation=require_same_variation,
        )
    )


def _box_from_center(
    center_x: float, center_y: float, width: float, height: float
) -> BBox:
    return BBox(
        x1=center_x - width / 2.0,
        y1=center_y - height / 2.0,
        x2=center_x + width / 2.0,
        y2=center_y + height / 2.0,
    )


def _predict_bbox(endpoint: _TrackEndpoints, target_time: float) -> BBox:
    if endpoint.previous_time is None or endpoint.previous_bbox is None:
        return endpoint.last_bbox
    elapsed = endpoint.last_time - endpoint.previous_time
    if elapsed <= 0.0:
        return endpoint.last_bbox

    scale = (target_time - endpoint.last_time) / elapsed
    previous_center_x, previous_center_y = endpoint.previous_bbox.center
    last_center_x, last_center_y = endpoint.last_bbox.center
    center_x = last_center_x + (last_center_x - previous_center_x) * scale
    center_y = last_center_y + (last_center_y - previous_center_y) * scale
    width = (
        endpoint.last_bbox.width
        + (endpoint.last_bbox.width - endpoint.previous_bbox.width) * scale
    )
    height = (
        endpoint.last_bbox.height
        + (endpoint.last_bbox.height - endpoint.previous_bbox.height) * scale
    )
    return _box_from_center(
        center_x,
        center_y,
        max(0.0, width),
        max(0.0, height),
    )


def _size_ratio(left: float, right: float) -> float:
    larger = max(left, right)
    if larger <= 0.0:
        return 0.0
    return min(left, right) / larger


def _prediction_metrics(
    left: _TrackEndpoints, right: _TrackEndpoints
) -> dict[str, float]:
    predicted = _predict_bbox(left, right.first_time)
    predicted_center_x, predicted_center_y = predicted.center
    first_center_x, first_center_y = right.first_bbox.center
    center_error = (
        (predicted_center_x - first_center_x) ** 2
        + (predicted_center_y - first_center_y) ** 2
    ) ** 0.5
    max_dimension = max(
        predicted.width,
        predicted.height,
        right.first_bbox.width,
        right.first_bbox.height,
    )
    return {
        "center_error": center_error,
        "max_dimension": max_dimension,
        "prediction_error_ratio": center_error / max_dimension
        if max_dimension > 0.0
        else float("inf"),
        "width_ratio": _size_ratio(predicted.width, right.first_bbox.width),
        "height_ratio": _size_ratio(predicted.height, right.first_bbox.height),
        "handoff_iou": predicted.iou(right.first_bbox),
    }


def _track_endpoints(records: list[FrameRecord]) -> dict[int, _TrackEndpoints]:
    observations: dict[int, list[tuple[float, BBox, int | None]]] = {}
    for record in records:
        for track in record.tracks:
            observations.setdefault(track.track_id, []).append(
                (track.timestamp_seconds, track.bbox, track.vehicle_index)
            )

    endpoints: dict[int, _TrackEndpoints] = {}
    for track_id, track_observations in observations.items():
        ordered = sorted(track_observations, key=lambda item: item[0])
        vehicle_indices = [item[2] for item in ordered if item[2] is not None]
        if not vehicle_indices:
            continue
        previous = ordered[-2] if len(ordered) >= 2 else None
        first_time, first_bbox, _first_vehicle_index = ordered[0]
        last_time, last_bbox, last_vehicle_index = ordered[-1]
        vehicle_index = (
            last_vehicle_index if last_vehicle_index is not None else vehicle_indices[0]
        )
        endpoints[track_id] = _TrackEndpoints(
            track_id=track_id,
            vehicle_index=vehicle_index,
            first_time=first_time,
            first_bbox=first_bbox,
            previous_time=previous[0] if previous is not None else None,
            previous_bbox=previous[1] if previous is not None else None,
            last_time=last_time,
            last_bbox=last_bbox,
        )
    return endpoints


def _world_handoff_metrics(
    view_transformer: ViewTransformer | None,
    left: _TrackEndpoints,
    right: _TrackEndpoints,
    gap_seconds: float,
) -> dict[str, float | None]:
    if view_transformer is None or gap_seconds <= 0.0:
        return {
            "world_handoff_distance_m": None,
            "implied_speed_mps": None,
        }
    distance_m = view_transformer.distance_between(
        left.last_bbox.bottom_center, right.first_bbox.bottom_center
    )
    if not math.isfinite(distance_m):
        return {
            "world_handoff_distance_m": None,
            "implied_speed_mps": None,
        }
    return {
        "world_handoff_distance_m": distance_m,
        "implied_speed_mps": distance_m / gap_seconds,
    }


def _passes_geometry(
    config: AppConfig,
    left: _TrackEndpoints,
    right: _TrackEndpoints,
    view_transformer: ViewTransformer | None = None,
) -> tuple[bool, dict[str, float | None]]:
    pixel_metrics = _prediction_metrics(left, right)
    tracker = config.tracker
    gap_seconds = right.first_time - left.last_time
    world_metrics = _world_handoff_metrics(view_transformer, left, right, gap_seconds)
    metrics: dict[str, float | None] = {**pixel_metrics, **world_metrics}
    implied_speed_mps = world_metrics["implied_speed_mps"]
    max_implied_speed = tracker.sequential_duplicate_max_implied_speed_mps
    passes_world_gate = (
        implied_speed_mps is None
        or max_implied_speed is None
        or implied_speed_mps <= max_implied_speed
    )
    return (
        passes_world_gate
        and pixel_metrics["prediction_error_ratio"]
        <= tracker.sequential_duplicate_prediction_error_ratio
        and pixel_metrics["width_ratio"] >= tracker.sequential_duplicate_min_width_ratio
        and pixel_metrics["height_ratio"]
        >= tracker.sequential_duplicate_min_height_ratio
        and pixel_metrics["handoff_iou"]
        >= tracker.sequential_duplicate_min_handoff_iou,
        metrics,
    )


def _rewrite_labels(
    labels: dict[int, MMRResult], canonical_by_vehicle: dict[int, int]
) -> dict[int, MMRResult]:
    rewritten: dict[int, MMRResult] = {}
    for track_id, label in labels.items():
        if label.vehicle_index is None:
            rewritten[track_id] = label
            continue
        canonical = canonical_by_vehicle.get(label.vehicle_index, label.vehicle_index)
        rewritten[track_id] = label.model_copy(
            update={
                "vehicle_index": canonical,
                "api_classification_index": canonical,
            }
        )
    return rewritten


def _rewrite_frames(
    records: list[FrameRecord], canonical_by_vehicle: dict[int, int]
) -> list[FrameRecord]:
    rewritten: list[FrameRecord] = []
    for record in records:
        tracks = []
        for track in record.tracks:
            vehicle_index = track.vehicle_index
            if vehicle_index is not None:
                vehicle_index = canonical_by_vehicle.get(vehicle_index, vehicle_index)
            tracks.append(track.model_copy(update={"vehicle_index": vehicle_index}))
        rewritten.append(record.model_copy(update={"tracks": tracks}))
    return rewritten


def _rewrite_summaries(
    summaries: list[TrackSummary], canonical_by_vehicle: dict[int, int]
) -> list[TrackSummary]:
    rewritten: list[TrackSummary] = []
    for summary in summaries:
        vehicle_index = summary.vehicle_index
        if vehicle_index is not None:
            vehicle_index = canonical_by_vehicle.get(vehicle_index, vehicle_index)
        rewritten.append(summary.model_copy(update={"vehicle_index": vehicle_index}))
    return rewritten


def _interpolate_bridge_bbox(
    last_bbox: BBox,
    first_bbox: BBox,
    last_time: float,
    first_time: float,
    target_time: float,
) -> BBox:
    total = first_time - last_time
    if total <= 0.0:
        return last_bbox
    ratio = (target_time - last_time) / total
    ratio = max(0.0, min(1.0, ratio))
    return BBox(
        x1=last_bbox.x1 + (first_bbox.x1 - last_bbox.x1) * ratio,
        y1=last_bbox.y1 + (first_bbox.y1 - last_bbox.y1) * ratio,
        x2=last_bbox.x2 + (first_bbox.x2 - last_bbox.x2) * ratio,
        y2=last_bbox.y2 + (first_bbox.y2 - last_bbox.y2) * ratio,
    )


def _positions_by_track_id(records: list[FrameRecord]) -> dict[int, list[int]]:
    positions: dict[int, list[int]] = {}
    for position, record in enumerate(records):
        for track in record.tracks:
            positions.setdefault(track.track_id, []).append(position)
    return positions


def _bridge_window(
    positions: dict[int, list[int]], left_track_id: int, right_track_id: int
) -> tuple[int, int] | None:
    left_positions = positions.get(left_track_id)
    right_positions = positions.get(right_track_id)
    if not left_positions or not right_positions:
        return None
    left_last_position = left_positions[-1]
    right_first_position = right_positions[0]
    if left_last_position >= right_first_position:
        return None
    return left_last_position, right_first_position


def _bridge_observation(
    record: FrameRecord,
    left_ep: _TrackEndpoints,
    right_ep: _TrackEndpoints,
    source: TrackedObject,
    vehicle_index: int,
) -> TrackedObject:
    bbox = _interpolate_bridge_bbox(
        left_ep.last_bbox,
        right_ep.first_bbox,
        left_ep.last_time,
        right_ep.first_time,
        record.timestamp_seconds,
    )
    return TrackedObject(
        track_id=left_ep.track_id,
        vehicle_index=vehicle_index,
        frame_index=record.frame_index,
        timestamp_seconds=record.timestamp_seconds,
        bbox=bbox,
        confidence=source.confidence,
        class_id=source.class_id,
        class_name=source.class_name,
        centroid=bbox.center,
        bottom_center=bbox.bottom_center,
        inside_roi=source.inside_roi,
        counted=False,
        crossed_line=False,
    )


def _inject_bridge_for_pair(
    records: list[FrameRecord],
    pair: dict[str, Any],
    endpoints: dict[int, _TrackEndpoints],
    canonical_by_vehicle: dict[int, int],
    positions: dict[int, list[int]],
) -> bool:
    left_ep = endpoints.get(pair["from_track_id"])
    right_ep = endpoints.get(pair["to_track_id"])
    if left_ep is None or right_ep is None:
        return False
    window = _bridge_window(positions, left_ep.track_id, right_ep.track_id)
    if window is None:
        return False
    left_last_position, right_first_position = window
    source = next(
        (
            track
            for track in records[left_last_position].tracks
            if track.track_id == left_ep.track_id
        ),
        None,
    )
    if source is None:
        return False
    vehicle_index = canonical_by_vehicle.get(
        left_ep.vehicle_index, left_ep.vehicle_index
    )
    injected = False
    for position in range(left_last_position + 1, right_first_position + 1):
        record = records[position]
        if any(track.track_id == left_ep.track_id for track in record.tracks):
            continue
        record.tracks.append(
            _bridge_observation(record, left_ep, right_ep, source, vehicle_index)
        )
        injected = True
    return injected


def _inject_bridge_observations(
    records: list[FrameRecord],
    merged_pairs: list[dict[str, Any]],
    endpoints: dict[int, _TrackEndpoints],
    canonical_by_vehicle: dict[int, int],
) -> list[FrameRecord]:
    if not merged_pairs:
        return records

    positions = _positions_by_track_id(records)
    for pair in merged_pairs:
        _inject_bridge_for_pair(
            records, pair, endpoints, canonical_by_vehicle, positions
        )
    return records


def _thresholds(config: AppConfig) -> dict[str, object]:
    tracker = config.tracker
    return {
        "max_gap_seconds": tracker.sequential_duplicate_max_gap_seconds,
        "prediction_error_ratio": tracker.sequential_duplicate_prediction_error_ratio,
        "min_width_ratio": tracker.sequential_duplicate_min_width_ratio,
        "min_height_ratio": tracker.sequential_duplicate_min_height_ratio,
        "min_handoff_iou": tracker.sequential_duplicate_min_handoff_iou,
        "max_implied_speed_mps": (tracker.sequential_duplicate_max_implied_speed_mps),
        "require_same_color": tracker.sequential_duplicate_require_same_color,
        "require_same_generation": tracker.sequential_duplicate_require_same_generation,
        "require_same_variation": tracker.sequential_duplicate_require_same_variation,
    }


def _labeled_vehicle_indices(labels: dict[int, MMRResult]) -> set[int]:
    return {
        label.vehicle_index
        for label in labels.values()
        if label.vehicle_index is not None
    }


def _ordered_labeled_endpoints(
    labels: dict[int, MMRResult], endpoints: dict[int, _TrackEndpoints]
) -> list[_TrackEndpoints]:
    labeled = [
        endpoint for track_id, endpoint in endpoints.items() if track_id in labels
    ]
    return sorted(labeled, key=lambda item: (item.first_time, item.track_id))


def _find_merged_pairs(
    config: AppConfig,
    *,
    labels: dict[int, MMRResult],
    ordered_endpoints: list[_TrackEndpoints],
    view_transformer: ViewTransformer | None,
    disjoint_set: _DisjointSet,
) -> list[dict[str, Any]]:
    tracker = config.tracker
    endpoint_first_times = [endpoint.first_time for endpoint in ordered_endpoints]
    merged_pairs: list[dict[str, Any]] = []
    for left in ordered_endpoints:
        left_label = labels[left.track_id]
        left_vehicle_index = left_label.vehicle_index
        if left_vehicle_index is None:
            continue
        search_start = bisect_right(endpoint_first_times, left.last_time)
        for right in ordered_endpoints[search_start:]:
            gap_seconds = right.first_time - left.last_time
            if gap_seconds > tracker.sequential_duplicate_max_gap_seconds:
                break
            right_label = labels[right.track_id]
            right_vehicle_index = right_label.vehicle_index
            if right_vehicle_index is None:
                continue
            if not _labels_match(
                left_label,
                right_label,
                require_same_color=tracker.sequential_duplicate_require_same_color,
                require_same_generation=(
                    tracker.sequential_duplicate_require_same_generation
                ),
                require_same_variation=(
                    tracker.sequential_duplicate_require_same_variation
                ),
            ):
                continue
            passes_geometry, metrics = _passes_geometry(
                config, left, right, view_transformer
            )
            if not passes_geometry:
                continue
            disjoint_set.union(left_vehicle_index, right_vehicle_index)
            merged_pairs.append(
                {
                    "from_track_id": left.track_id,
                    "to_track_id": right.track_id,
                    "from_vehicle_index": left.vehicle_index,
                    "to_vehicle_index": right.vehicle_index,
                    "gap_seconds": gap_seconds,
                    **metrics,
                }
            )
    return merged_pairs


def _changed_vehicle_indices(canonical_by_vehicle: dict[int, int]) -> dict[int, int]:
    return {
        vehicle_index: canonical
        for vehicle_index, canonical in canonical_by_vehicle.items()
        if vehicle_index != canonical
    }


def _write_audit(audit_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    write_json(audit_path, payload)
    return payload


def deduplicate_classified_tracks(
    config: AppConfig,
    run_store: RunStore,
    profile: CameraProfile | None = None,
) -> dict[str, Any]:
    audit_path = run_store.analysis_dir / "sequential_duplicates.json"
    thresholds = _thresholds(config)
    view_transformer = (
        build_view_transformer(config, profile) if profile is not None else None
    )
    if not config.tracker.suppress_sequential_duplicate_tracks:
        return _write_audit(
            audit_path,
            {
                "enabled": False,
                "thresholds": thresholds,
                "world_gate_active": view_transformer is not None,
                "merged_pairs": [],
            },
        )

    labels = run_store.labels.read()
    records = run_store.frames.read_all(smoothed=False)
    summaries = run_store.tracks.read_all()
    endpoints = _track_endpoints(records)
    disjoint_set = _DisjointSet(_labeled_vehicle_indices(labels))
    merged_pairs = _find_merged_pairs(
        config,
        labels=labels,
        ordered_endpoints=_ordered_labeled_endpoints(labels, endpoints),
        view_transformer=view_transformer,
        disjoint_set=disjoint_set,
    )

    canonical_by_vehicle = disjoint_set.canonical_map()
    changed_vehicle_indices = _changed_vehicle_indices(canonical_by_vehicle)
    if changed_vehicle_indices:
        run_store.labels.write(_rewrite_labels(labels, canonical_by_vehicle))
        records = _rewrite_frames(records, canonical_by_vehicle)
        run_store.tracks.write_all(_rewrite_summaries(summaries, canonical_by_vehicle))
    if merged_pairs:
        records = _inject_bridge_observations(
            records, merged_pairs, endpoints, canonical_by_vehicle
        )
    if changed_vehicle_indices or merged_pairs:
        run_store.frames.write_all(records)

    return _write_audit(
        audit_path,
        {
            "enabled": True,
            "thresholds": thresholds,
            "world_gate_active": view_transformer is not None,
            "merged_pairs": merged_pairs,
            "canonical_vehicle_indices": {
                str(vehicle_index): canonical
                for vehicle_index, canonical in sorted(changed_vehicle_indices.items())
            },
            "merged_vehicle_count": len(changed_vehicle_indices),
        },
    )
