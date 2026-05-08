from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import orjson

from car_census.config import AppConfig, CameraProfile
from car_census.roi.geometry import point_in_polygon
from car_census.storage.run_store import RunStore
from car_census.types import BBox, FrameRecord, TrackedObject


@dataclass(slots=True)
class _TrackPoint:
    frame_index: int
    timestamp_seconds: float
    vector: np.ndarray
    source: TrackedObject
    interpolated: bool = False


def _iter_frame_records(path: Path) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    if not path.exists():
        return records
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                records.append(FrameRecord.model_validate(orjson.loads(line)))
    return records


def _write_frame_records(path: Path, records: list[FrameRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for record in records:
            handle.write(orjson.dumps(record.model_dump(mode="json")))
            handle.write(b"\n")


def _expand_records_to_source_frames(
    records: list[FrameRecord],
    source_fps: float,
) -> list[FrameRecord]:
    if not records:
        return []
    if source_fps <= 0:
        return records

    records_by_index = {record.frame_index: record for record in records}
    first_frame_index = records[0].frame_index
    last_frame_index = records[-1].frame_index
    expanded: list[FrameRecord] = []
    for frame_index in range(first_frame_index, last_frame_index + 1):
        record = records_by_index.get(frame_index)
        if record is not None:
            expanded.append(record)
            continue
        expanded.append(
            FrameRecord(
                frame_index=frame_index,
                timestamp_seconds=frame_index / source_fps,
                tracks=[],
            )
        )
    return expanded


def _box_to_vector(bbox: BBox) -> np.ndarray:
    center_x, center_y = bbox.center
    return np.array([center_x, center_y, bbox.width, bbox.height], dtype=np.float64)


def _vector_to_box(vector: np.ndarray) -> BBox | None:
    center_x, center_y, width, height = [float(value) for value in vector]
    if width <= 0 or height <= 0:
        return None
    return BBox(
        x1=center_x - width / 2.0,
        y1=center_y - height / 2.0,
        x2=center_x + width / 2.0,
        y2=center_y + height / 2.0,
    )


def _interpolate_point(
    previous: _TrackPoint,
    current: _TrackPoint,
    frame: FrameRecord,
) -> _TrackPoint:
    total = current.timestamp_seconds - previous.timestamp_seconds
    if total <= 0:
        ratio = 0.5
    else:
        ratio = (frame.timestamp_seconds - previous.timestamp_seconds) / total
    ratio = max(0.0, min(1.0, ratio))
    vector = previous.vector + ((current.vector - previous.vector) * ratio)
    source = previous.source.model_copy(
        update={
            "frame_index": frame.frame_index,
            "timestamp_seconds": frame.timestamp_seconds,
            "confidence": min(previous.source.confidence, current.source.confidence),
            "crossed_line": False,
            "counted": previous.source.counted or current.source.counted,
        }
    )
    return _TrackPoint(
        frame_index=frame.frame_index,
        timestamp_seconds=frame.timestamp_seconds,
        vector=vector,
        source=source,
        interpolated=True,
    )


def _fill_short_gaps(
    points: list[_TrackPoint],
    records: list[FrameRecord],
    max_gap_seconds: float,
) -> list[_TrackPoint]:
    if len(points) < 2:
        return points

    records_by_index = {record.frame_index: record for record in records}
    frame_order = [record.frame_index for record in records]
    frame_position = {
        frame_index: index for index, frame_index in enumerate(frame_order)
    }

    filled: list[_TrackPoint] = []
    for previous, current in zip(points, points[1:]):
        filled.append(previous)
        gap_seconds = current.timestamp_seconds - previous.timestamp_seconds
        previous_position = frame_position.get(previous.frame_index)
        current_position = frame_position.get(current.frame_index)
        if (
            previous_position is None
            or current_position is None
            or current_position <= previous_position + 1
            or gap_seconds > max_gap_seconds
        ):
            continue
        for frame_index in frame_order[previous_position + 1 : current_position]:
            frame = records_by_index[frame_index]
            filled.append(_interpolate_point(previous, current, frame))
    filled.append(points[-1])
    return filled


def _split_contiguous_segments(
    points: list[_TrackPoint],
    records: list[FrameRecord],
) -> list[list[_TrackPoint]]:
    if not points:
        return []

    frame_position = {record.frame_index: index for index, record in enumerate(records)}
    segments: list[list[_TrackPoint]] = [[points[0]]]
    for point in points[1:]:
        previous = segments[-1][-1]
        previous_position = frame_position.get(previous.frame_index)
        current_position = frame_position.get(point.frame_index)
        if (
            previous_position is not None
            and current_position is not None
            and current_position == previous_position + 1
        ):
            segments[-1].append(point)
        else:
            segments.append([point])
    return segments


def _local_polynomial_smooth(
    values: np.ndarray,
    timestamps: np.ndarray,
    window_size: int,
    polynomial_order: int,
) -> np.ndarray:
    if len(values) < 3 or window_size < 3:
        return values.copy()

    window_size = min(window_size, len(values))
    if window_size % 2 == 0:
        window_size -= 1
    if window_size < 3:
        return values.copy()

    half_window = window_size // 2
    smoothed = values.copy()
    for index in range(len(values)):
        start = max(0, index - half_window)
        end = min(len(values), index + half_window + 1)
        if end - start < window_size:
            if start == 0:
                end = min(len(values), window_size)
            elif end == len(values):
                start = max(0, len(values) - window_size)

        local_timestamps = timestamps[start:end] - timestamps[index]
        local_order = min(polynomial_order, len(local_timestamps) - 1)
        if local_order < 1:
            continue
        for column in range(values.shape[1]):
            coefficients = np.polyfit(
                local_timestamps, values[start:end, column], local_order
            )
            smoothed[index, column] = np.polyval(coefficients, 0.0)
    return smoothed


def _clamp_vector(
    candidate: np.ndarray, reference: np.ndarray, config: AppConfig
) -> np.ndarray:
    result = candidate.copy()
    max_offset = (
        max(float(reference[2]), float(reference[3]))
        * config.render.smoothing.max_center_offset_ratio
    )
    center_delta = result[0:2] - reference[0:2]
    center_distance = float(np.linalg.norm(center_delta))
    if max_offset >= 0 and center_distance > max_offset and center_distance > 0:
        result[0:2] = reference[0:2] + center_delta * (max_offset / center_distance)

    max_size_delta_ratio = config.render.smoothing.max_size_delta_ratio
    for column in [2, 3]:
        reference_size = float(reference[column])
        max_delta = reference_size * max_size_delta_ratio
        result[column] = min(
            reference_size + max_delta,
            max(reference_size - max_delta, float(result[column])),
        )
    return result


def _window_size(config: AppConfig, analysis_fps: float) -> int:
    if analysis_fps <= 0:
        analysis_fps = 30.0
    size = int(round(config.render.smoothing.window_seconds * analysis_fps))
    size = max(3, size)
    if size % 2 == 0:
        size += 1
    return size


def _smooth_segment(
    segment: list[_TrackPoint],
    config: AppConfig,
    analysis_fps: float,
) -> list[_TrackPoint]:
    if len(segment) < 3:
        return segment

    values = np.stack([point.vector for point in segment])
    timestamps = np.array(
        [point.timestamp_seconds for point in segment], dtype=np.float64
    )
    smoothed = _local_polynomial_smooth(
        values=values,
        timestamps=timestamps,
        window_size=_window_size(config, analysis_fps),
        polynomial_order=config.render.smoothing.polynomial_order,
    )

    output: list[_TrackPoint] = []
    for point, candidate in zip(segment, smoothed):
        clamped = _clamp_vector(candidate, point.vector, config)
        if _vector_to_box(clamped) is None:
            clamped = point.vector
        output.append(
            _TrackPoint(
                frame_index=point.frame_index,
                timestamp_seconds=point.timestamp_seconds,
                vector=clamped,
                source=point.source,
                interpolated=point.interpolated,
            )
        )
    return output


def _render_track_from_point(
    point: _TrackPoint, profile: CameraProfile
) -> TrackedObject:
    bbox = _vector_to_box(point.vector) or point.source.bbox
    bottom_center = ((bbox.x1 + bbox.x2) / 2.0, bbox.y2)
    return point.source.model_copy(
        update={
            "frame_index": point.frame_index,
            "timestamp_seconds": point.timestamp_seconds,
            "bbox": bbox,
            "centroid": bbox.center,
            "bottom_center": bottom_center,
            "inside_roi": point_in_polygon(bottom_center, profile.polygon.points),
            "crossed_line": False if point.interpolated else point.source.crossed_line,
        }
    )


def smooth_render_tracks(
    config: AppConfig,
    profile: CameraProfile,
    run_store: RunStore,
) -> Path:
    analysis_records = _iter_frame_records(run_store.frames_path)
    if not config.render.smoothing.enabled:
        return run_store.frames_path
    if not analysis_records:
        _write_frame_records(run_store.render_frames_path, [])
        return run_store.render_frames_path

    manifest = run_store.read_manifest()
    records = (
        _expand_records_to_source_frames(analysis_records, manifest.source_fps)
        if config.render.smoothing.interpolate
        else analysis_records
    )
    tracks_by_id: dict[int, list[_TrackPoint]] = {}
    output_by_frame: dict[int, list[TrackedObject]] = {
        record.frame_index: list(record.tracks) for record in records
    }

    for record in analysis_records:
        for track in record.tracks:
            tracks_by_id.setdefault(track.track_id, []).append(
                _TrackPoint(
                    frame_index=record.frame_index,
                    timestamp_seconds=record.timestamp_seconds,
                    vector=_box_to_vector(track.bbox),
                    source=track,
                )
            )

    for track_id, points in tracks_by_id.items():
        points.sort(key=lambda point: point.timestamp_seconds)
        if len(points) < config.render.smoothing.min_observations:
            continue

        for frame_tracks in output_by_frame.values():
            frame_tracks[:] = [
                track for track in frame_tracks if track.track_id != track_id
            ]

        filled = (
            _fill_short_gaps(
                points=points,
                records=records,
                max_gap_seconds=config.render.smoothing.max_gap_seconds,
            )
            if config.render.smoothing.interpolate
            else points
        )
        smoothing_fps = (
            manifest.source_fps if manifest.source_fps > 0 else manifest.analysis_fps
        )
        for segment in _split_contiguous_segments(filled, records):
            segment_points = (
                _smooth_segment(segment, config, smoothing_fps)
                if config.render.smoothing.smooth_keyframes
                else segment
            )
            for point in segment_points:
                output_by_frame[point.frame_index].append(
                    _render_track_from_point(point, profile)
                )

    smoothed_records = [
        FrameRecord(
            frame_index=record.frame_index,
            timestamp_seconds=record.timestamp_seconds,
            tracks=sorted(
                output_by_frame[record.frame_index],
                key=lambda track: track.track_id,
            ),
        )
        for record in records
    ]
    _write_frame_records(run_store.render_frames_path, smoothed_records)
    return run_store.render_frames_path
