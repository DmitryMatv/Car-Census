from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import supervision as sv

from config import AppConfig, CameraProfile
from models import BBox, FrameRecord, TrackedObject
from roi.geometry import point_in_polygon
from storage.run_store import RunStore


@dataclass(slots=True)
class _TrackRenderPoint:
    frame_index: int
    timestamp_seconds: float
    bbox: BBox
    confidence: float
    source: TrackedObject


@dataclass(slots=True)
class _ObservedTrackPoint:
    record_position: int
    frame_index: int
    track: TrackedObject


@dataclass(frozen=True, slots=True)
class _MissingAnalysisGap:
    track_id: int
    previous_position: int
    current_position: int


def _tracks_to_detections(tracks: Sequence[TrackedObject]) -> sv.Detections:
    if not tracks:
        return sv.Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.empty((0,), dtype=np.float32),
            class_id=np.empty((0,), dtype=np.int32),
            tracker_id=np.empty((0,), dtype=np.int32),
        )

    return sv.Detections(
        xyxy=np.array(
            [
                [track.bbox.x1, track.bbox.y1, track.bbox.x2, track.bbox.y2]
                for track in tracks
            ],
            dtype=np.float32,
        ),
        confidence=np.array([track.confidence for track in tracks], dtype=np.float32),
        class_id=np.array(
            [track.class_id if track.class_id is not None else -1 for track in tracks],
            dtype=np.int32,
        ),
        tracker_id=np.array([track.track_id for track in tracks], dtype=np.int32),
    )


def _bbox_from_xyxy(xyxy: np.ndarray) -> BBox:
    x1, y1, x2, y2 = [float(value) for value in xyxy]
    return BBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _bbox_to_center_size(bbox: BBox) -> tuple[float, float, float, float]:
    center_x, center_y = bbox.center
    return center_x, center_y, bbox.width, bbox.height


def _center_size_to_bbox(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
) -> BBox:
    width = max(1.0, width)
    height = max(1.0, height)
    half_width = width / 2.0
    half_height = height / 2.0
    return BBox(
        x1=center_x - half_width,
        y1=center_y - half_height,
        x2=center_x + half_width,
        y2=center_y + half_height,
    )


def _clamp_float(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _smoothed_tracks_for_record(
    record: FrameRecord,
    smoothed_detections: sv.Detections,
    profile: CameraProfile,
) -> list[TrackedObject]:
    # DetectionsSmoother keeps recent history per track; render only tracks that
    # are still present in the current analysis record.
    current_by_track_id = {track.track_id: track for track in record.tracks}
    if smoothed_detections.tracker_id is None:
        return []

    smoothed_tracks: list[TrackedObject] = []
    for detection_index, tracker_id_value in enumerate(smoothed_detections.tracker_id):
        track_id = int(tracker_id_value)
        source = current_by_track_id.get(track_id)
        if source is None:
            continue

        bbox = _bbox_from_xyxy(smoothed_detections.xyxy[detection_index])
        bottom_center = bbox.bottom_center
        confidence = (
            float(smoothed_detections.confidence[detection_index])
            if smoothed_detections.confidence is not None
            else source.confidence
        )
        smoothed_tracks.append(
            source.model_copy(
                update={
                    "bbox": bbox,
                    "confidence": confidence,
                    "centroid": bbox.center,
                    "bottom_center": bottom_center,
                    "inside_roi": point_in_polygon(
                        bottom_center, profile.polygon.points
                    ),
                    "crossed_line": source.crossed_line,
                }
            )
        )

    return sorted(smoothed_tracks, key=lambda track: track.track_id)


def _track_points_by_id(record: FrameRecord) -> dict[int, _TrackRenderPoint]:
    return {
        track.track_id: _TrackRenderPoint(
            frame_index=record.frame_index,
            timestamp_seconds=record.timestamp_seconds,
            bbox=track.bbox,
            confidence=track.confidence,
            source=track,
        )
        for track in record.tracks
    }


def _interpolate_float(previous: float, current: float, ratio: float) -> float:
    return previous + ((current - previous) * ratio)


def _interpolate_bbox(previous: BBox, current: BBox, ratio: float) -> BBox:
    return BBox(
        x1=_interpolate_float(previous.x1, current.x1, ratio),
        y1=_interpolate_float(previous.y1, current.y1, ratio),
        x2=_interpolate_float(previous.x2, current.x2, ratio),
        y2=_interpolate_float(previous.y2, current.y2, ratio),
    )


def _interpolation_ratio(ratio: float, config: AppConfig) -> float:
    ratio = max(0.0, min(1.0, ratio))
    if config.render.smoothing.interpolation_method == "linear":
        return ratio
    raise ValueError(
        f"Unsupported interpolation method: "
        f"{config.render.smoothing.interpolation_method}"
    )


def _interpolate_track(
    previous: _TrackRenderPoint,
    current: _TrackRenderPoint,
    frame_index: int,
    timestamp_seconds: float,
    profile: CameraProfile,
    config: AppConfig,
) -> TrackedObject:
    total = current.timestamp_seconds - previous.timestamp_seconds
    ratio = (
        0.0 if total <= 0 else (timestamp_seconds - previous.timestamp_seconds) / total
    )
    ratio = _interpolation_ratio(ratio, config)
    bbox = _interpolate_bbox(previous.bbox, current.bbox, ratio)
    bottom_center = bbox.bottom_center
    source = previous.source
    # Interpolated frames are render-only. Keep count state available for
    # display, but leave the crossing event anchored to the observed frame.
    return source.model_copy(
        update={
            "vehicle_index": source.vehicle_index
            if source.vehicle_index is not None
            else current.source.vehicle_index,
            "frame_index": frame_index,
            "timestamp_seconds": timestamp_seconds,
            "bbox": bbox,
            "confidence": _interpolate_float(
                previous.confidence, current.confidence, ratio
            ),
            "class_id": source.class_id
            if source.class_id is not None
            else current.source.class_id,
            "class_name": source.class_name
            if source.class_name is not None
            else current.source.class_name,
            "centroid": bbox.center,
            "bottom_center": bottom_center,
            "inside_roi": point_in_polygon(bottom_center, profile.polygon.points),
            "counted": source.counted or current.source.counted,
            "crossed_line": False,
        }
    )


def _effective_max_interpolation_gap_seconds(config: AppConfig) -> float:
    configured = config.render.smoothing.max_interpolation_gap_seconds
    if configured is not None:
        return configured
    # Allow roughly one expected analysis interval with tolerance, but avoid
    # bridging longer detector/tracker absences by default.
    return 1.5 / min(config.analysis.fps, config.video.fps)


def _empty_clear_record(previous: FrameRecord, config: AppConfig) -> FrameRecord:
    frame_index = previous.frame_index + 1
    return FrameRecord(
        frame_index=frame_index,
        timestamp_seconds=frame_index / config.video.fps,
        tracks=[],
    )


def _interpolated_records_between(
    previous: FrameRecord,
    current: FrameRecord,
    config: AppConfig,
    profile: CameraProfile,
) -> Iterator[FrameRecord]:
    if current.frame_index <= previous.frame_index + 1:
        return

    gap_seconds = current.timestamp_seconds - previous.timestamp_seconds
    if gap_seconds <= 0:
        return
    if gap_seconds > _effective_max_interpolation_gap_seconds(config):
        yield _empty_clear_record(previous, config)
        return

    previous_points = _track_points_by_id(previous)
    current_points = _track_points_by_id(current)
    interpolated_track_ids = sorted(previous_points.keys() & current_points.keys())
    if not interpolated_track_ids:
        yield _empty_clear_record(previous, config)
        return

    for frame_index in range(previous.frame_index + 1, current.frame_index):
        timestamp_seconds = frame_index / config.video.fps
        yield FrameRecord(
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            tracks=[
                _interpolate_track(
                    previous=previous_points[track_id],
                    current=current_points[track_id],
                    frame_index=frame_index,
                    timestamp_seconds=timestamp_seconds,
                    profile=profile,
                    config=config,
                )
                for track_id in interpolated_track_ids
            ],
        )


def iter_interpolated_source_frame_records(
    records: Iterable[FrameRecord],
    config: AppConfig,
    profile: CameraProfile,
) -> Iterator[FrameRecord]:
    iterator = iter(records)
    previous = next(iterator, None)
    if previous is None:
        return

    yield previous
    for current in iterator:
        yield from _interpolated_records_between(
            previous=previous,
            current=current,
            config=config,
            profile=profile,
        )
        yield current
        previous = current


def iter_causal_average_analysis_records(
    records: Iterable[FrameRecord],
    config: AppConfig,
    profile: CameraProfile,
) -> Iterator[FrameRecord]:
    smoother = sv.DetectionsSmoother(length=config.render.smoothing.history_length)
    for record in records:
        detections = _tracks_to_detections(record.tracks)
        smoothed_detections = smoother.update_with_detections(detections)
        yield FrameRecord(
            frame_index=record.frame_index,
            timestamp_seconds=record.timestamp_seconds,
            tracks=_smoothed_tracks_for_record(
                record=record,
                smoothed_detections=smoothed_detections,
                profile=profile,
            ),
        )


def _window_around_index(
    points: list[_ObservedTrackPoint],
    target_index: int,
    window_size: int,
) -> list[_ObservedTrackPoint]:
    window_size = min(window_size, len(points))
    before = window_size // 2
    start = target_index - before
    start = max(0, min(start, len(points) - window_size))
    return points[start : start + window_size]


def _fit_linear_at_frame(
    points: list[_ObservedTrackPoint],
    frame_index: int,
    values: list[float],
    fallback: float,
) -> float:
    if len(points) < 2:
        return fallback

    base_frame_index = points[0].frame_index
    x_values = np.array(
        [point.frame_index - base_frame_index for point in points],
        dtype=np.float64,
    )
    if np.unique(x_values).size < 2:
        return fallback

    y_values = np.array(values, dtype=np.float64)
    try:
        slope, intercept = np.polyfit(x_values, y_values, deg=1)
    except np.linalg.LinAlgError:
        return fallback
    return float((slope * (frame_index - base_frame_index)) + intercept)


def _smoothed_local_linear_bbox(
    point: _ObservedTrackPoint,
    window: list[_ObservedTrackPoint],
    config: AppConfig,
) -> BBox:
    raw_center_x, raw_center_y, raw_width, raw_height = _bbox_to_center_size(
        point.track.bbox
    )
    if len(window) < 2:
        return point.track.bbox

    center_x_values: list[float] = []
    center_y_values: list[float] = []
    width_values: list[float] = []
    height_values: list[float] = []
    for window_point in window:
        center_x, center_y, width, height = _bbox_to_center_size(
            window_point.track.bbox
        )
        center_x_values.append(center_x)
        center_y_values.append(center_y)
        width_values.append(width)
        height_values.append(height)

    fitted_center_x = _fit_linear_at_frame(
        window, point.frame_index, center_x_values, raw_center_x
    )
    fitted_center_y = _fit_linear_at_frame(
        window, point.frame_index, center_y_values, raw_center_y
    )
    fitted_width = _fit_linear_at_frame(
        window, point.frame_index, width_values, raw_width
    )
    fitted_height = _fit_linear_at_frame(
        window, point.frame_index, height_values, raw_height
    )

    shift_ratio = config.render.smoothing.observed_smoothing_max_shift_ratio
    max_center_x_delta = raw_width * shift_ratio
    max_center_y_delta = raw_height * shift_ratio
    max_width_delta = raw_width * shift_ratio
    max_height_delta = raw_height * shift_ratio

    center_x = _clamp_float(
        fitted_center_x,
        raw_center_x - max_center_x_delta,
        raw_center_x + max_center_x_delta,
    )
    center_y = _clamp_float(
        fitted_center_y,
        raw_center_y - max_center_y_delta,
        raw_center_y + max_center_y_delta,
    )
    width = _clamp_float(
        fitted_width,
        raw_width - max_width_delta,
        raw_width + max_width_delta,
    )
    height = _clamp_float(
        fitted_height,
        raw_height - max_height_delta,
        raw_height + max_height_delta,
    )
    return _center_size_to_bbox(center_x, center_y, width, height)


def _segment_observed_points(
    points: list[_ObservedTrackPoint],
    config: AppConfig,
) -> Iterator[list[_ObservedTrackPoint]]:
    if not points:
        return

    segment = [points[0]]
    max_gap = config.render.smoothing.max_missing_analysis_gap_frames
    for previous, current in zip(points, points[1:], strict=False):
        missing_count = current.record_position - previous.record_position - 1
        if missing_count > max_gap:
            yield segment
            segment = [current]
        else:
            segment.append(current)
    yield segment


def _local_linear_track_updates(
    points: list[_ObservedTrackPoint],
    config: AppConfig,
    profile: CameraProfile,
) -> dict[int, TrackedObject]:
    updates: dict[int, TrackedObject] = {}
    window_size = config.render.smoothing.observed_smoothing_window
    for segment in _segment_observed_points(points, config):
        for index, point in enumerate(segment):
            window = _window_around_index(segment, index, window_size)
            bbox = _smoothed_local_linear_bbox(point, window, config)
            bottom_center = bbox.bottom_center
            updates[point.record_position] = point.track.model_copy(
                update={
                    "bbox": bbox,
                    "centroid": bbox.center,
                    "bottom_center": bottom_center,
                    "inside_roi": point_in_polygon(
                        bottom_center, profile.polygon.points
                    ),
                }
            )
    return updates


def iter_local_linear_analysis_records(
    records: Iterable[FrameRecord],
    config: AppConfig,
    profile: CameraProfile,
) -> Iterator[FrameRecord]:
    materialized = list(records)
    if not materialized:
        return

    points_by_track: dict[int, list[_ObservedTrackPoint]] = {}
    for record_position, record in enumerate(materialized):
        for track in record.tracks:
            points_by_track.setdefault(track.track_id, []).append(
                _ObservedTrackPoint(
                    record_position=record_position,
                    frame_index=record.frame_index,
                    track=track,
                )
            )

    updates_by_track: dict[int, dict[int, TrackedObject]] = {
        track_id: _local_linear_track_updates(points, config, profile)
        for track_id, points in points_by_track.items()
    }

    for record_position, record in enumerate(materialized):
        tracks = [
            updates_by_track.get(track.track_id, {}).get(record_position, track)
            for track in record.tracks
        ]
        yield record.model_copy(
            update={"tracks": sorted(tracks, key=lambda track: track.track_id)}
        )


def iter_observed_render_records(
    records: Iterable[FrameRecord],
    config: AppConfig,
    profile: CameraProfile,
) -> Iterator[FrameRecord]:
    mode = config.render.smoothing.observed_box_smoothing
    if mode == "none":
        yield from records
        return

    if mode == "causal_average":
        yield from iter_causal_average_analysis_records(records, config, profile)
        return

    if mode == "local_linear":
        yield from iter_local_linear_analysis_records(records, config, profile)
        return

    raise ValueError(f"Unsupported observed box smoothing mode: {mode}")


def _track_positions_by_id(
    records: Sequence[FrameRecord],
) -> dict[int, list[int]]:
    positions_by_track: dict[int, list[int]] = {}
    for record_position, record in enumerate(records):
        for track in record.tracks:
            positions_by_track.setdefault(track.track_id, []).append(record_position)
    return positions_by_track


def _tracks_by_record(records: Sequence[FrameRecord]) -> list[list[TrackedObject]]:
    return [list(record.tracks) for record in records]


def _track_ids_by_record(
    tracks_by_record: Sequence[Sequence[TrackedObject]],
) -> list[set[int]]:
    return [{track.track_id for track in tracks} for tracks in tracks_by_record]


def _iter_missing_analysis_gaps(
    observed_positions_by_track: dict[int, list[int]],
    max_missing_frames: int,
) -> Iterator[_MissingAnalysisGap]:
    for track_id, observed_positions in observed_positions_by_track.items():
        for previous_position, current_position in zip(
            observed_positions, observed_positions[1:], strict=False
        ):
            missing_count = current_position - previous_position - 1
            if 1 <= missing_count <= max_missing_frames:
                yield _MissingAnalysisGap(
                    track_id=track_id,
                    previous_position=previous_position,
                    current_position=current_position,
                )


def _record_track_points_by_id(
    records: Sequence[FrameRecord],
) -> list[dict[int, _TrackRenderPoint]]:
    return [_track_points_by_id(record) for record in records]


def _fill_missing_analysis_gap(
    gap: _MissingAnalysisGap,
    records: Sequence[FrameRecord],
    tracks_by_record: list[list[TrackedObject]],
    track_ids_by_record: list[set[int]],
    points_by_record: Sequence[dict[int, _TrackRenderPoint]],
    config: AppConfig,
    profile: CameraProfile,
) -> None:
    previous_point = points_by_record[gap.previous_position].get(gap.track_id)
    current_point = points_by_record[gap.current_position].get(gap.track_id)
    if previous_point is None or current_point is None:
        return

    for record_position in range(gap.previous_position + 1, gap.current_position):
        if gap.track_id in track_ids_by_record[record_position]:
            continue

        target_record = records[record_position]
        tracks_by_record[record_position].append(
            _interpolate_track(
                previous=previous_point,
                current=current_point,
                frame_index=target_record.frame_index,
                timestamp_seconds=target_record.timestamp_seconds,
                profile=profile,
                config=config,
            )
        )
        track_ids_by_record[record_position].add(gap.track_id)


def _sorted_gap_filled_records(
    records: Sequence[FrameRecord],
    tracks_by_record: Sequence[Sequence[TrackedObject]],
) -> Iterator[FrameRecord]:
    for record, tracks in zip(records, tracks_by_record, strict=True):
        yield record.model_copy(
            update={"tracks": sorted(tracks, key=lambda track: track.track_id)}
        )


def iter_missing_analysis_gap_filled_records(
    records: Iterable[FrameRecord],
    config: AppConfig,
    profile: CameraProfile,
) -> Iterator[FrameRecord]:
    materialized = list(records)
    if not materialized:
        return

    tracks_by_record = _tracks_by_record(materialized)
    track_ids_by_record = _track_ids_by_record(tracks_by_record)
    points_by_record = _record_track_points_by_id(materialized)
    for gap in _iter_missing_analysis_gaps(
        _track_positions_by_id(materialized),
        config.render.smoothing.max_missing_analysis_gap_frames,
    ):
        _fill_missing_analysis_gap(
            gap=gap,
            records=materialized,
            tracks_by_record=tracks_by_record,
            track_ids_by_record=track_ids_by_record,
            points_by_record=points_by_record,
            config=config,
            profile=profile,
        )
    yield from _sorted_gap_filled_records(materialized, tracks_by_record)


def smooth_render_tracks(
    config: AppConfig,
    profile: CameraProfile,
    run_store: RunStore,
) -> Path:
    if not config.render.smoothing.enabled:
        return run_store.frames_path

    render_records: Iterable[FrameRecord] = iter_observed_render_records(
        run_store.frames.iter(smoothed=False), config, profile
    )
    if config.render.smoothing.bridge_missing_analysis_frames:
        render_records = iter_missing_analysis_gap_filled_records(
            render_records, config, profile
        )
    if config.render.smoothing.interpolate_source_frames:
        render_records = iter_interpolated_source_frame_records(
            render_records, config, profile
        )

    with run_store.frames.open_writer(smoothed=True) as writer:
        for record in render_records:
            writer.write(record)
    return run_store.render_frames_path
