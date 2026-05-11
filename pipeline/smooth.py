from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import orjson

from config import AppConfig, CameraProfile
from roi.geometry import point_in_polygon
from storage.run_store import RunStore
from models import BBox, FrameRecord, TrackedObject


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


def _linear_interpolate_vector(
    previous: _TrackPoint,
    current: _TrackPoint,
    timestamp_seconds: float,
) -> np.ndarray:
    total = current.timestamp_seconds - previous.timestamp_seconds
    if total <= 0:
        ratio = 0.5
    else:
        ratio = (timestamp_seconds - previous.timestamp_seconds) / total
    ratio = max(0.0, min(1.0, ratio))
    return previous.vector + ((current.vector - previous.vector) * ratio)


def _interpolated_source(
    previous: _TrackPoint,
    current: _TrackPoint,
    frame: FrameRecord,
) -> TrackedObject:
    return previous.source.model_copy(
        update={
            "frame_index": frame.frame_index,
            "timestamp_seconds": frame.timestamp_seconds,
            "confidence": min(previous.source.confidence, current.source.confidence),
            "crossed_line": False,
            "counted": previous.source.counted or current.source.counted,
        }
    )


def _fill_linear_gaps(
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


def _select_polynomial_keyframes(
    points: list[_TrackPoint],
    previous_index: int,
    target_timestamp: float,
    requested_order: int,
) -> list[_TrackPoint]:
    required_count = requested_order + 1
    selected_indices = {previous_index, previous_index + 1}
    before_index = previous_index - 1
    after_index = previous_index + 2

    while len(selected_indices) < required_count and (
        before_index >= 0 or after_index < len(points)
    ):
        before_distance = (
            abs(points[before_index].timestamp_seconds - target_timestamp)
            if before_index >= 0
            else None
        )
        after_distance = (
            abs(points[after_index].timestamp_seconds - target_timestamp)
            if after_index < len(points)
            else None
        )

        if before_distance is None:
            selected_indices.add(after_index)
            after_index += 1
        elif after_distance is None:
            selected_indices.add(before_index)
            before_index -= 1
        elif before_distance <= after_distance:
            selected_indices.add(before_index)
            before_index -= 1
        else:
            selected_indices.add(after_index)
            after_index += 1

    return [points[index] for index in sorted(selected_indices)]


def _polynomial_interpolate_vector(
    keyframes: list[_TrackPoint],
    target_timestamp: float,
    requested_order: int,
) -> np.ndarray | None:
    local_order = min(requested_order, len(keyframes) - 1)
    if local_order < 1:
        return None

    timestamps = np.array(
        [point.timestamp_seconds for point in keyframes], dtype=np.float64
    )
    local_timestamps = timestamps - target_timestamp
    values = np.stack([point.vector for point in keyframes])
    vector = np.empty(values.shape[1], dtype=np.float64)
    for column in range(values.shape[1]):
        coefficients = np.polyfit(local_timestamps, values[:, column], local_order)
        vector[column] = np.polyval(coefficients, 0.0)
    if not np.all(np.isfinite(vector)):
        return None
    return vector


def _fill_polynomial_gaps(
    points: list[_TrackPoint],
    records: list[FrameRecord],
    config: AppConfig,
) -> list[_TrackPoint]:
    if len(points) < 2:
        return points

    records_by_index = {record.frame_index: record for record in records}
    frame_order = [record.frame_index for record in records]
    frame_position = {
        frame_index: index for index, frame_index in enumerate(frame_order)
    }

    filled: list[_TrackPoint] = []
    requested_order = config.render.smoothing.polynomial_order
    for point_index, (previous, current) in enumerate(zip(points, points[1:])):
        filled.append(previous)
        gap_seconds = current.timestamp_seconds - previous.timestamp_seconds
        previous_position = frame_position.get(previous.frame_index)
        current_position = frame_position.get(current.frame_index)
        if (
            previous_position is None
            or current_position is None
            or current_position <= previous_position + 1
            or gap_seconds > config.render.smoothing.max_gap_seconds
        ):
            continue

        for frame_index in frame_order[previous_position + 1 : current_position]:
            frame = records_by_index[frame_index]
            linear_vector = _linear_interpolate_vector(
                previous, current, frame.timestamp_seconds
            )
            keyframes = _select_polynomial_keyframes(
                points, point_index, frame.timestamp_seconds, requested_order
            )
            candidate = _polynomial_interpolate_vector(
                keyframes, frame.timestamp_seconds, requested_order
            )
            if candidate is None:
                candidate = linear_vector
            clamped = _clamp_vector(candidate, linear_vector, config)
            if _vector_to_box(clamped) is None:
                clamped = linear_vector
            filled.append(
                _TrackPoint(
                    frame_index=frame.frame_index,
                    timestamp_seconds=frame.timestamp_seconds,
                    vector=clamped,
                    source=_interpolated_source(previous, current, frame),
                    interpolated=True,
                )
            )
    filled.append(points[-1])
    return filled


def _endpoint_hermite_slope(
    first_delta: float, second_delta: float, first_h: float, second_h: float
) -> float:
    slope = ((2 * first_h + second_h) * first_delta - first_h * second_delta) / (
        first_h + second_h
    )
    if np.sign(slope) != np.sign(first_delta):
        return 0.0
    if np.sign(first_delta) != np.sign(second_delta) and abs(slope) > abs(
        3.0 * first_delta
    ):
        return 3.0 * first_delta
    return float(slope)


def _monotone_hermite_slopes(
    timestamps: np.ndarray,
    values: np.ndarray,
) -> np.ndarray | None:
    if len(timestamps) != len(values) or len(timestamps) < 2:
        return None

    h = np.diff(timestamps)
    if np.any(h <= 0):
        return None

    delta = np.diff(values, axis=0) / h[:, np.newaxis]
    if not np.all(np.isfinite(delta)):
        return None

    slopes = np.empty_like(values, dtype=np.float64)
    if len(timestamps) == 2:
        slopes[0] = delta[0]
        slopes[1] = delta[0]
        return slopes

    for column in range(values.shape[1]):
        column_delta = delta[:, column]
        slopes[0, column] = _endpoint_hermite_slope(
            column_delta[0], column_delta[1], h[0], h[1]
        )
        for index in range(1, len(timestamps) - 1):
            previous_delta = column_delta[index - 1]
            current_delta = column_delta[index]
            if previous_delta * current_delta <= 0:
                slopes[index, column] = 0.0
                continue
            w1 = 2.0 * h[index] + h[index - 1]
            w2 = h[index] + 2.0 * h[index - 1]
            slopes[index, column] = (w1 + w2) / (
                (w1 / previous_delta) + (w2 / current_delta)
            )
        slopes[-1, column] = _endpoint_hermite_slope(
            column_delta[-1], column_delta[-2], h[-1], h[-2]
        )

    if not np.all(np.isfinite(slopes)):
        return None
    return slopes


def _hermite_interpolate_vector(
    previous: _TrackPoint,
    current: _TrackPoint,
    previous_slope: np.ndarray,
    current_slope: np.ndarray,
    timestamp_seconds: float,
) -> np.ndarray | None:
    segment_duration = current.timestamp_seconds - previous.timestamp_seconds
    if segment_duration <= 0:
        return None

    s = (timestamp_seconds - previous.timestamp_seconds) / segment_duration
    s = max(0.0, min(1.0, float(s)))
    s2 = s * s
    s3 = s2 * s
    h00 = (2.0 * s3) - (3.0 * s2) + 1.0
    h10 = s3 - (2.0 * s2) + s
    h01 = (-2.0 * s3) + (3.0 * s2)
    h11 = s3 - s2
    vector = (
        h00 * previous.vector
        + h10 * segment_duration * previous_slope
        + h01 * current.vector
        + h11 * segment_duration * current_slope
    )
    if not np.all(np.isfinite(vector)):
        return None
    return vector


def _fill_hermite_gaps(
    points: list[_TrackPoint],
    records: list[FrameRecord],
    config: AppConfig,
) -> list[_TrackPoint]:
    if len(points) < 2:
        return points

    timestamps = np.array(
        [point.timestamp_seconds for point in points], dtype=np.float64
    )
    values = np.stack([point.vector for point in points])
    slopes = _monotone_hermite_slopes(timestamps, values)

    records_by_index = {record.frame_index: record for record in records}
    frame_order = [record.frame_index for record in records]
    frame_position = {
        frame_index: index for index, frame_index in enumerate(frame_order)
    }

    filled: list[_TrackPoint] = []
    for point_index, (previous, current) in enumerate(zip(points, points[1:])):
        filled.append(previous)
        gap_seconds = current.timestamp_seconds - previous.timestamp_seconds
        previous_position = frame_position.get(previous.frame_index)
        current_position = frame_position.get(current.frame_index)
        if (
            previous_position is None
            or current_position is None
            or current_position <= previous_position + 1
            or gap_seconds > config.render.smoothing.max_gap_seconds
        ):
            continue

        for frame_index in frame_order[previous_position + 1 : current_position]:
            frame = records_by_index[frame_index]
            linear_vector = _linear_interpolate_vector(
                previous, current, frame.timestamp_seconds
            )
            candidate = (
                _hermite_interpolate_vector(
                    previous=previous,
                    current=current,
                    previous_slope=slopes[point_index],
                    current_slope=slopes[point_index + 1],
                    timestamp_seconds=frame.timestamp_seconds,
                )
                if slopes is not None
                else None
            )
            if candidate is None:
                candidate = linear_vector
            clamped = _clamp_vector(candidate, linear_vector, config)
            if _vector_to_box(clamped) is None:
                clamped = linear_vector
            filled.append(
                _TrackPoint(
                    frame_index=frame.frame_index,
                    timestamp_seconds=frame.timestamp_seconds,
                    vector=clamped,
                    source=_interpolated_source(previous, current, frame),
                    interpolated=True,
                )
            )
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
    if not np.all(np.isfinite(candidate)):
        return reference.copy()
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

        if not config.render.smoothing.interpolate:
            filled = points
        elif config.render.smoothing.interpolation_method == "linear":
            filled = _fill_linear_gaps(
                points=points,
                records=records,
                max_gap_seconds=config.render.smoothing.max_gap_seconds,
            )
        elif config.render.smoothing.interpolation_method == "polynomial":
            filled = _fill_polynomial_gaps(
                points=points,
                records=records,
                config=config,
            )
        else:
            filled = _fill_hermite_gaps(
                points=points,
                records=records,
                config=config,
            )
        smoothing_fps = (
            manifest.source_fps if manifest.source_fps > 0 else manifest.analysis_fps
        )
        should_smooth_segment = (
            config.render.smoothing.smooth_keyframes
            and config.render.smoothing.interpolation_method == "linear"
        )
        for segment in _split_contiguous_segments(filled, records):
            segment_points = (
                _smooth_segment(segment, config, smoothing_fps)
                if should_smooth_segment
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
