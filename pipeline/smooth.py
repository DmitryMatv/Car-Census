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


def iter_observed_render_records(
    records: Iterable[FrameRecord],
    config: AppConfig,
    profile: CameraProfile,
) -> Iterator[FrameRecord]:
    if config.render.smoothing.observed_box_smoothing == "none":
        yield from records
        return

    yield from iter_causal_average_analysis_records(records, config, profile)


def iter_missing_analysis_gap_filled_records(
    records: Iterable[FrameRecord],
    config: AppConfig,
    profile: CameraProfile,
) -> Iterator[FrameRecord]:
    materialized = list(records)
    if not materialized:
        return

    observed_positions_by_track: dict[int, list[int]] = {}
    for record_position, record in enumerate(materialized):
        for track in record.tracks:
            observed_positions_by_track.setdefault(track.track_id, []).append(
                record_position
            )

    tracks_by_record = [list(record.tracks) for record in materialized]
    for track_id, observed_positions in observed_positions_by_track.items():
        for previous_position, current_position in zip(
            observed_positions, observed_positions[1:], strict=False
        ):
            missing_count = current_position - previous_position - 1
            if (
                missing_count < 1
                or missing_count
                > config.render.smoothing.max_missing_analysis_gap_frames
            ):
                continue

            previous_record = materialized[previous_position]
            current_record = materialized[current_position]
            previous_point = _track_points_by_id(previous_record).get(track_id)
            current_point = _track_points_by_id(current_record).get(track_id)
            if previous_point is None or current_point is None:
                continue
            for record_position in range(previous_position + 1, current_position):
                if any(
                    track.track_id == track_id
                    for track in tracks_by_record[record_position]
                ):
                    continue
                target_record = materialized[record_position]
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

    for record, tracks in zip(materialized, tracks_by_record, strict=True):
        yield record.model_copy(
            update={"tracks": sorted(tracks, key=lambda track: track.track_id)}
        )


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
