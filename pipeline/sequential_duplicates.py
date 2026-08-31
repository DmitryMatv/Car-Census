from __future__ import annotations

import math
from dataclasses import dataclass
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
    world_metrics = _world_handoff_metrics(
        view_transformer, left, right, gap_seconds
    )
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
        and pixel_metrics["width_ratio"]
        >= tracker.sequential_duplicate_min_width_ratio
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


def _inject_bridge_observations(
    records: list[FrameRecord],
    merged_pairs: list[dict[str, Any]],
    endpoints: dict[int, _TrackEndpoints],
    canonical_by_vehicle: dict[int, int],
) -> list[FrameRecord]:
    if not merged_pairs:
        return records

    for pair in merged_pairs:
        left_ep = endpoints.get(pair["from_track_id"])
        right_ep = endpoints.get(pair["to_track_id"])
        if left_ep is None or right_ep is None:
            continue

        left_last_pos: int | None = None
        right_first_pos: int | None = None
        for pos, record in enumerate(records):
            has_left = any(t.track_id == left_ep.track_id for t in record.tracks)
            has_right = any(t.track_id == right_ep.track_id for t in record.tracks)
            if has_left:
                left_last_pos = pos
            if has_right and right_first_pos is None:
                right_first_pos = pos

        if left_last_pos is None or right_first_pos is None:
            continue
        if left_last_pos >= right_first_pos:
            continue

        left_record = records[left_last_pos]
        left_track_obs = None
        for t in left_record.tracks:
            if t.track_id == left_ep.track_id:
                left_track_obs = t
                break
        if left_track_obs is None:
            continue

        vehicle_index = canonical_by_vehicle.get(
            left_ep.vehicle_index, left_ep.vehicle_index
        )
        track_id = left_ep.track_id
        confidence = left_track_obs.confidence
        class_id = left_track_obs.class_id
        class_name = left_track_obs.class_name
        inside_roi = left_track_obs.inside_roi

        for pos in range(left_last_pos + 1, right_first_pos + 1):
            record = records[pos]
            already_has_track = any(t.track_id == track_id for t in record.tracks)
            if already_has_track:
                continue

            bbox = _interpolate_bridge_bbox(
                left_ep.last_bbox,
                right_ep.first_bbox,
                left_ep.last_time,
                right_ep.first_time,
                record.timestamp_seconds,
            )
            synthetic = TrackedObject(
                track_id=track_id,
                vehicle_index=vehicle_index,
                frame_index=record.frame_index,
                timestamp_seconds=record.timestamp_seconds,
                bbox=bbox,
                confidence=confidence,
                class_id=class_id,
                class_name=class_name,
                centroid=bbox.center,
                bottom_center=bbox.bottom_center,
                inside_roi=inside_roi,
                counted=False,
                crossed_line=False,
            )
            record.tracks.append(synthetic)

    return records


def _thresholds(config: AppConfig) -> dict[str, object]:
    tracker = config.tracker
    return {
        "max_gap_seconds": tracker.sequential_duplicate_max_gap_seconds,
        "prediction_error_ratio": tracker.sequential_duplicate_prediction_error_ratio,
        "min_width_ratio": tracker.sequential_duplicate_min_width_ratio,
        "min_height_ratio": tracker.sequential_duplicate_min_height_ratio,
        "min_handoff_iou": tracker.sequential_duplicate_min_handoff_iou,
        "max_implied_speed_mps": (
            tracker.sequential_duplicate_max_implied_speed_mps
        ),
        "require_same_color": tracker.sequential_duplicate_require_same_color,
        "require_same_generation": tracker.sequential_duplicate_require_same_generation,
        "require_same_variation": tracker.sequential_duplicate_require_same_variation,
    }


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
        payload = {
            "enabled": False,
            "thresholds": thresholds,
            "world_gate_active": view_transformer is not None,
            "merged_pairs": [],
        }
        write_json(audit_path, payload)
        return payload

    labels = run_store.labels.read()
    records = run_store.frames.read_all(smoothed=False)
    summaries = run_store.tracks.read_all()
    endpoints = _track_endpoints(records)
    labeled_endpoints = [
        endpoint for track_id, endpoint in endpoints.items() if track_id in labels
    ]
    vehicle_indices = {
        label.vehicle_index
        for label in labels.values()
        if label.vehicle_index is not None
    }
    disjoint_set = _DisjointSet(vehicle_indices)
    merged_pairs: list[dict[str, Any]] = []

    ordered_endpoints = sorted(
        labeled_endpoints, key=lambda item: (item.first_time, item.track_id)
    )
    for left in ordered_endpoints:
        left_label = labels[left.track_id]
        for right in ordered_endpoints:
            if right.first_time <= left.last_time:
                continue
            gap_seconds = right.first_time - left.last_time
            if gap_seconds > config.tracker.sequential_duplicate_max_gap_seconds:
                break
            right_label = labels[right.track_id]
            if not _labels_match(
                left_label,
                right_label,
                require_same_color=config.tracker.sequential_duplicate_require_same_color,
                require_same_generation=config.tracker.sequential_duplicate_require_same_generation,
                require_same_variation=config.tracker.sequential_duplicate_require_same_variation,
            ):
                continue
            passes_geometry, metrics = _passes_geometry(
                config, left, right, view_transformer
            )
            if not passes_geometry:
                continue
            if left_label.vehicle_index is None or right_label.vehicle_index is None:
                continue
            disjoint_set.union(left_label.vehicle_index, right_label.vehicle_index)
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

    canonical_by_vehicle = disjoint_set.canonical_map()
    changed_vehicle_indices = {
        vehicle_index: canonical
        for vehicle_index, canonical in canonical_by_vehicle.items()
        if vehicle_index != canonical
    }
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

    payload = {
        "enabled": True,
        "thresholds": thresholds,
        "world_gate_active": view_transformer is not None,
        "merged_pairs": merged_pairs,
        "canonical_vehicle_indices": {
            str(vehicle_index): canonical
            for vehicle_index, canonical in sorted(changed_vehicle_indices.items())
        },
        "merged_vehicle_count": len(changed_vehicle_indices),
    }
    write_json(audit_path, payload)
    return payload
