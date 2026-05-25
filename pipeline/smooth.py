from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from config import AppConfig, CameraProfile
from models import BBox, FrameRecord, RunManifest, TrackedObject
from roi.geometry import point_in_polygon
from storage.run_store import RunStore


@dataclass(slots=True)
class _TrackPoint:
    frame_index: int
    timestamp_seconds: float
    vector: np.ndarray
    source: TrackedObject
    interpolated: bool = False


@dataclass(slots=True)
class _FrameGrid:
    by_index: dict[int, FrameRecord]
    order: list[int]
    position: dict[int, int]


def _build_frame_grid(records: list[FrameRecord]) -> _FrameGrid:
    order = [record.frame_index for record in records]
    return _FrameGrid(
        by_index={record.frame_index: record for record in records},
        order=order,
        position={frame_index: index for index, frame_index in enumerate(order)},
    )


def _expand_records_to_source_frames(
    records: list[FrameRecord],
    source_fps: float,
    final_frame_index: int | None = None,
) -> list[FrameRecord]:
    if not records:
        return []
    if source_fps <= 0:
        return records

    records_by_index = {record.frame_index: record for record in records}
    first_frame_index = records[0].frame_index
    last_frame_index = max(records[-1].frame_index, final_frame_index or 0)
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


def _manifest_final_frame_index(manifest: RunManifest) -> int | None:
    if manifest.frame_count > 0:
        return manifest.frame_count - 1
    return None


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


def _linear_extrapolate_vector(
    previous: _TrackPoint,
    current: _TrackPoint,
    timestamp_seconds: float,
) -> np.ndarray | None:
    total = current.timestamp_seconds - previous.timestamp_seconds
    if total <= 0:
        return None
    return current.vector + (
        (current.vector - previous.vector)
        * ((timestamp_seconds - current.timestamp_seconds) / total)
    )


def _generated_gap_source(
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


def _gap_frame_indices(
    previous: _TrackPoint,
    current: _TrackPoint,
    grid: _FrameGrid,
    max_gap_seconds: float,
) -> list[int]:
    gap_seconds = current.timestamp_seconds - previous.timestamp_seconds
    previous_position = grid.position.get(previous.frame_index)
    current_position = grid.position.get(current.frame_index)
    if (
        previous_position is None
        or current_position is None
        or current_position <= previous_position + 1
        or gap_seconds > max_gap_seconds
    ):
        return []
    return grid.order[previous_position + 1 : current_position]


def _fill_linear_gaps(
    points: list[_TrackPoint],
    grid: _FrameGrid,
    max_gap_seconds: float,
) -> list[_TrackPoint]:
    if len(points) < 2:
        return points

    filled: list[_TrackPoint] = []
    for previous, current in zip(points, points[1:]):
        filled.append(previous)
        for frame_index in _gap_frame_indices(previous, current, grid, max_gap_seconds):
            frame = grid.by_index[frame_index]
            filled.append(
                _TrackPoint(
                    frame_index=frame.frame_index,
                    timestamp_seconds=frame.timestamp_seconds,
                    vector=_linear_interpolate_vector(
                        previous, current, frame.timestamp_seconds
                    ),
                    source=_generated_gap_source(previous, current, frame),
                    interpolated=True,
                )
            )
    filled.append(points[-1])
    return filled


def _fill_pchip_gaps(
    points: list[_TrackPoint],
    grid: _FrameGrid,
    config: AppConfig,
) -> list[_TrackPoint]:
    if len(points) < 2:
        return points
    if len(points) == 2:
        return _fill_linear_gaps(
            points=points,
            grid=grid,
            max_gap_seconds=config.render.smoothing.max_gap_seconds,
        )

    timestamps = np.array(
        [point.timestamp_seconds for point in points], dtype=np.float64
    )

    # PCHIP requires strictly increasing timestamps; fall back to linear if not satisfied
    if np.any(np.diff(timestamps) <= 0):
        return _fill_linear_gaps(
            points=points,
            grid=grid,
            max_gap_seconds=config.render.smoothing.max_gap_seconds,
        )
    values = np.stack([point.vector for point in points])
    interpolator = PchipInterpolator(timestamps, values, axis=0, extrapolate=False)

    filled: list[_TrackPoint] = []
    for previous, current in zip(points, points[1:]):
        filled.append(previous)
        for frame_index in _gap_frame_indices(
            previous,
            current,
            grid,
            config.render.smoothing.max_gap_seconds,
        ):
            frame = grid.by_index[frame_index]
            linear_vector = _linear_interpolate_vector(
                previous, current, frame.timestamp_seconds
            )
            candidate = interpolator(frame.timestamp_seconds)
            if not np.all(np.isfinite(candidate)):
                candidate = linear_vector
            clamped = _clamp_vector(candidate, linear_vector, config)
            if _vector_to_box(clamped) is None:
                clamped = linear_vector
            filled.append(
                _TrackPoint(
                    frame_index=frame.frame_index,
                    timestamp_seconds=frame.timestamp_seconds,
                    vector=clamped,
                    source=_generated_gap_source(previous, current, frame),
                    interpolated=True,
                )
            )
    filled.append(points[-1])
    return filled


def _extrapolate_visible_tail(
    points: list[_TrackPoint],
    grid: _FrameGrid,
    config: AppConfig,
    final_analysis_frame_index: int,
) -> list[_TrackPoint]:
    if len(points) < 2 or points[-1].frame_index != final_analysis_frame_index:
        return points

    last_position = grid.position.get(points[-1].frame_index)
    if last_position is None or last_position >= len(grid.order) - 1:
        return points

    previous = points[-2]
    current = points[-1]
    extrapolated = list(points)
    for frame_index in grid.order[last_position + 1 :]:
        frame = grid.by_index[frame_index]
        gap_seconds = frame.timestamp_seconds - current.timestamp_seconds
        if gap_seconds > config.render.smoothing.max_gap_seconds:
            break
        candidate = _linear_extrapolate_vector(
            previous, current, frame.timestamp_seconds
        )
        if candidate is None:
            break
        reference = extrapolated[-1].vector
        clamped = _clamp_vector(candidate, reference, config)
        if _vector_to_box(clamped) is None:
            break
        extrapolated.append(
            _TrackPoint(
                frame_index=frame.frame_index,
                timestamp_seconds=frame.timestamp_seconds,
                vector=clamped,
                source=current.source.model_copy(
                    update={
                        "frame_index": frame.frame_index,
                        "timestamp_seconds": frame.timestamp_seconds,
                        "crossed_line": False,
                    }
                ),
                interpolated=True,
            )
        )
    return extrapolated


def _split_temporal_segments(
    points: list[_TrackPoint],
    grid: _FrameGrid,
    max_gap_seconds: float,
) -> list[list[_TrackPoint]]:
    if not points:
        return []

    segments: list[list[_TrackPoint]] = [[points[0]]]
    for point in points[1:]:
        previous = segments[-1][-1]
        previous_position = grid.position.get(previous.frame_index)
        current_position = grid.position.get(point.frame_index)
        gap_seconds = point.timestamp_seconds - previous.timestamp_seconds
        if (
            previous_position is not None
            and current_position is not None
            and current_position > previous_position
            and 0 <= gap_seconds <= max_gap_seconds
        ):
            segments[-1].append(point)
        else:
            segments.append([point])
    return segments


def _is_short_excursion(
    segment: list[_TrackPoint],
    start_index: int,
    run_length: int,
    config: AppConfig,
) -> bool:
    before = segment[start_index - 1]
    after = segment[start_index + run_length]
    anchor_distance = float(np.linalg.norm(after.vector[0:2] - before.vector[0:2]))
    anchor_size = max(
        float(before.vector[2]),
        float(before.vector[3]),
        float(after.vector[2]),
        float(after.vector[3]),
    )
    max_anchor_distance = (
        config.render.smoothing.excursion_center_ratio
        * max(anchor_size, 1.0)
        * (run_length + 1)
    )
    if anchor_distance > max_anchor_distance:
        return False

    for point in segment[start_index : start_index + run_length]:
        expected = _linear_interpolate_vector(before, after, point.timestamp_seconds)
        distance = float(np.linalg.norm(point.vector[0:2] - expected[0:2]))
        threshold = config.render.smoothing.excursion_center_ratio * max(
            float(expected[2]), float(expected[3]), 1.0
        )
        if distance <= threshold:
            return False
    return True


def _reject_short_excursions(
    points: list[_TrackPoint],
    grid: _FrameGrid,
    config: AppConfig,
) -> list[_TrackPoint]:
    smoothing = config.render.smoothing
    if not smoothing.reject_short_excursions or len(points) < 3:
        return points

    output: list[_TrackPoint] = []
    max_run_length = smoothing.max_excursion_observations
    for segment in _split_temporal_segments(points, grid, smoothing.max_gap_seconds):
        if len(segment) < 3:
            output.extend(segment)
            continue

        corrected = list(segment)
        index = 1
        while index < len(segment) - 1:
            remaining = len(segment) - index - 1
            candidate_lengths = range(min(max_run_length, remaining), 0, -1)
            accepted_length = next(
                (
                    run_length
                    for run_length in candidate_lengths
                    if _is_short_excursion(segment, index, run_length, config)
                ),
                None,
            )
            if accepted_length is None:
                index += 1
                continue

            before = segment[index - 1]
            after = segment[index + accepted_length]
            for offset, point in enumerate(segment[index : index + accepted_length]):
                expected = _linear_interpolate_vector(
                    before, after, point.timestamp_seconds
                )
                corrected[index + offset] = _TrackPoint(
                    frame_index=point.frame_index,
                    timestamp_seconds=point.timestamp_seconds,
                    vector=expected,
                    source=point.source,
                    interpolated=True,
                )
            index += accepted_length

        output.extend(corrected)
    return output


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


def _render_track_from_point(
    point: _TrackPoint, profile: CameraProfile
) -> TrackedObject:
    bbox = _vector_to_box(point.vector) or point.source.bbox
    bottom_center = bbox.bottom_center
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
    analysis_records = run_store.frames.read_all(smoothed=False)
    if not config.render.smoothing.enabled:
        return run_store.frames_path
    if not analysis_records:
        run_store.frames.write_all([], smoothed=True)
        return run_store.render_frames_path

    manifest = run_store.manifest.read()
    final_frame_index = _manifest_final_frame_index(manifest)
    records = (
        _expand_records_to_source_frames(
            analysis_records, config.video.fps, final_frame_index
        )
        if config.render.smoothing.interpolate
        else analysis_records
    )
    grid = _build_frame_grid(records)
    final_analysis_frame_index = analysis_records[-1].frame_index
    tracks_by_id: dict[int, list[_TrackPoint]] = {}

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

    eligible_track_ids = {
        track_id
        for track_id, points in tracks_by_id.items()
        if len(points) >= config.render.smoothing.min_observations
    }
    output_by_frame: dict[int, list[TrackedObject]] = {
        record.frame_index: [] for record in records
    }
    for record in analysis_records:
        for track in record.tracks:
            if track.track_id not in eligible_track_ids:
                output_by_frame[record.frame_index].append(track)

    for track_id, points in tracks_by_id.items():
        points.sort(key=lambda point: point.timestamp_seconds)
        if track_id not in eligible_track_ids:
            continue
        points = _reject_short_excursions(points, grid, config)

        if not config.render.smoothing.interpolate:
            filled = points
        elif config.render.smoothing.interpolation_method == "linear":
            filled = _fill_linear_gaps(
                points=points,
                grid=grid,
                max_gap_seconds=config.render.smoothing.max_gap_seconds,
            )
        else:
            filled = _fill_pchip_gaps(
                points=points,
                grid=grid,
                config=config,
            )
        if config.render.smoothing.interpolate:
            filled = _extrapolate_visible_tail(
                points=filled,
                grid=grid,
                config=config,
                final_analysis_frame_index=final_analysis_frame_index,
            )
        for point in filled:
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
    run_store.frames.write_all(smoothed_records, smoothed=True)
    return run_store.render_frames_path
