from __future__ import annotations

import logging
from collections.abc import Mapping
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import cv2

from config import AppConfig, CameraProfile
from detectors.base import Detector
from detectors.factory import create_detector
from models import (
    BBox,
    CountEvent,
    CropCandidate,
    Detection,
    FrameRecord,
    RunManifest,
    TrackedObject,
)
from pipeline.vehicles import (
    discard_track_artifacts,
    finalize_vehicle_identities,
    staged_track_crop_dir,
    track_summary_from_state,
)
from roi.geometry import (
    bbox_touches_frame_edge,
    bbox_touches_polygon_edge,
    bbox_touches_rect_edge,
    clip_bbox_to_frame,
    crop_to_polygon,
    line_crossing_direction,
    map_bbox_to_global,
    point_in_polygon,
)
from storage.run_store import RunStore
from tracking_adapters.botsort import BotSortAdapter
from utils.image_quality import laplacian_sharpness
from utils.video import iter_sampled_frames, read_video_metadata, validate_video_fps

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MutableTrackState:
    track_id: int
    first_frame_index: int
    last_frame_index: int
    vehicle_index: int | None = None
    frames_seen: int = 0
    min_box_height_px: float | None = None
    max_box_height_px: float = 0.0
    previous_bottom_center: tuple[float, float] | None = None
    counted: bool = False
    count_event: CountEvent | None = None
    candidates: list[CropCandidate] = field(default_factory=list)
    last_candidate_time: float | None = None


@dataclass(slots=True)
class _SampledFrame:
    frame_index: int
    timestamp_seconds: float
    frame: cv2.typing.MatLike
    roi_frame: cv2.typing.MatLike
    offset: tuple[int, int]


@dataclass(slots=True)
class AnalysisDiagnostics:
    total_sampled_frames: int = 0
    detections_passed_to_tracker: int = 0
    tracker_outputs: int = 0
    tracks_discarded_edge_contact: int = 0
    edge_observations_skipped: int = 0
    tracks_discarded_min_track_frames: int = 0
    tracks_without_crop_candidates: int = 0
    tracks_without_crop_due_to_height: int = 0
    tracks_without_crop_due_to_short_lifetime: int = 0
    tracks_hidden_from_render_crop_eligibility: int = 0
    tracker_confidence_values: list[float] = field(default_factory=list)
    tracker_box_height_values: list[float] = field(default_factory=list)


def _histogram(
    values: SequenceABC[float], bins: SequenceABC[float]
) -> list[dict[str, Any]]:
    counts = [0 for _ in range(max(0, len(bins) - 1))]
    for value in values:
        for index, (lower, upper) in enumerate(zip(bins, bins[1:], strict=True)):
            if lower <= value < upper or (index == len(counts) - 1 and value == upper):
                counts[index] += 1
                break
    return [
        {
            "min": lower,
            "max": upper if upper != float("inf") else None,
            "count": count,
        }
        for lower, upper, count in zip(bins[:-1], bins[1:], counts, strict=True)
    ]


def _detector_diagnostics(detector: Detector) -> dict[str, Any]:
    snapshot = getattr(detector, "detection_diagnostics", None)
    if not callable(snapshot):
        return {}
    raw = snapshot()
    return raw if isinstance(raw, dict) else {}


def _diagnostic_count(
    detector_counts: Mapping[str, object], key: str, fallback: int
) -> int:
    value = detector_counts.get(key)
    return int(value) if isinstance(value, int | float) else fallback


def _diagnostic_float_values(value: object) -> list[float]:
    if not isinstance(value, list):
        return []
    return [float(item) for item in value if isinstance(item, int | float)]


def _analysis_diagnostics_payload(
    diagnostics: AnalysisDiagnostics, detector: Detector
) -> dict[str, Any]:
    detector_snapshot = _detector_diagnostics(detector)
    detector_counts_raw = detector_snapshot.get("counts", {})
    detector_counts = (
        detector_counts_raw if isinstance(detector_counts_raw, Mapping) else {}
    )
    detector_confidences = _diagnostic_float_values(
        detector_snapshot.get("confidence_values")
    )
    confidence_values = detector_confidences or diagnostics.tracker_confidence_values

    return {
        "total_sampled_frames": diagnostics.total_sampled_frames,
        "raw_detections_before_class_filtering": _diagnostic_count(
            detector_counts,
            "raw_candidate_rows",
            diagnostics.detections_passed_to_tracker,
        ),
        "detections_after_confidence_filtering": _diagnostic_count(
            detector_counts,
            "detections_after_confidence_filtering",
            diagnostics.detections_passed_to_tracker,
        ),
        "detections_after_class_filtering": _diagnostic_count(
            detector_counts,
            "detections_after_class_filtering",
            diagnostics.detections_passed_to_tracker,
        ),
        "detections_passed_to_tracker": diagnostics.detections_passed_to_tracker,
        "tracker_outputs": diagnostics.tracker_outputs,
        "tracks_discarded_edge_contact": diagnostics.tracks_discarded_edge_contact,
        "edge_observations_skipped": diagnostics.edge_observations_skipped,
        "tracks_discarded_min_track_frames": diagnostics.tracks_discarded_min_track_frames,
        "tracks_without_crop_candidates": diagnostics.tracks_without_crop_candidates,
        "tracks_without_crop_due_to_height": (
            diagnostics.tracks_without_crop_due_to_height
        ),
        "tracks_without_crop_due_to_short_lifetime": (
            diagnostics.tracks_without_crop_due_to_short_lifetime
        ),
        "tracks_hidden_from_render_due_to_crop_eligibility": (
            diagnostics.tracks_hidden_from_render_crop_eligibility
        ),
        "confidence_histogram": _histogram(
            confidence_values,
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01],
        ),
        "box_height_histogram": _histogram(
            diagnostics.tracker_box_height_values,
            [
                0.0,
                40.0,
                80.0,
                120.0,
                160.0,
                200.0,
                300.0,
                400.0,
                600.0,
                800.0,
                float("inf"),
            ],
        ),
        "detector": detector_snapshot,
    }


def _iter_sampled_frame_batches(
    *,
    video_path: Path,
    source_fps: float,
    target_fps: float,
    profile: CameraProfile,
    batch_size: int,
) -> Iterator[list[_SampledFrame]]:
    batch: list[_SampledFrame] = []
    for frame_index, timestamp_seconds, frame in iter_sampled_frames(
        video_path, source_fps=source_fps, target_fps=target_fps
    ):
        roi_frame, offset = crop_to_polygon(frame, profile.polygon.points)
        batch.append(
            _SampledFrame(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                frame=frame,
                roi_frame=roi_frame,
                offset=offset,
            )
        )
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _iter_detected_sampled_frames(
    *,
    detector: Detector,
    video_path: Path,
    source_fps: float,
    target_fps: float,
    profile: CameraProfile,
    batch_size: int,
) -> Iterator[tuple[_SampledFrame, list[Detection]]]:
    for batch in _iter_sampled_frame_batches(
        video_path=video_path,
        source_fps=source_fps,
        target_fps=target_fps,
        profile=profile,
        batch_size=batch_size,
    ):
        batch_detections = detector.detect_batch([item.roi_frame for item in batch])
        if len(batch_detections) != len(batch):
            raise RuntimeError(
                f"Detector returned {len(batch_detections)} detection sets for "
                f"{len(batch)} frames"
            )
        yield from zip(batch, batch_detections, strict=True)


def _score_candidate(
    crop: cv2.typing.MatLike, bbox: BBox, frame_shape: tuple[int, int, int]
) -> tuple[float, float, float]:
    sharpness = laplacian_sharpness(crop)
    area_score = bbox.area
    height, width = frame_shape[:2]
    margin_left = bbox.x1
    margin_top = bbox.y1
    margin_right = width - bbox.x2
    margin_bottom = height - bbox.y2
    edge_margin_score = min(margin_left, margin_top, margin_right, margin_bottom)
    return sharpness, edge_margin_score, area_score


def _candidate_target_score(
    *,
    bbox_height: float,
    sharpness: float,
    edge_margin_score: float,
    area_score: float,
    frame_index: int,
    min_box_height_px: float | None,
    max_box_height_px: float,
    target_ratio: float,
) -> float:
    min_height = (
        min_box_height_px if min_box_height_px is not None else max_box_height_px
    )
    target_height = min_height + ((max_box_height_px - min_height) * target_ratio)
    scale_error = abs(bbox_height - target_height) / max(target_height, 1.0)
    return (
        (-scale_error * 1_000_000_000.0)
        + (sharpness * 1_000.0)
        + (edge_margin_score * 10.0)
        + area_score
        - (frame_index * 0.000001)
    )


def _candidate_rank(
    candidate: CropCandidate,
    min_box_height_px: float | None,
    max_box_height_px: float,
    config: AppConfig,
) -> tuple[float, float, float, float, int]:
    min_height = (
        min_box_height_px if min_box_height_px is not None else max_box_height_px
    )
    target_height = min_height + (
        (max_box_height_px - min_height) * config.analysis.crop_target_box_range_ratio
    )
    vehicle_bbox = candidate.vehicle_bbox or candidate.bbox
    scale_error = abs(vehicle_bbox.height - target_height) / max(target_height, 1.0)
    return (
        -scale_error,
        candidate.sharpness,
        candidate.edge_margin_score,
        candidate.area_score,
        -candidate.frame_index,
    )


def _refresh_candidate_score(
    candidate: CropCandidate,
    min_box_height_px: float | None,
    max_box_height_px: float,
    config: AppConfig,
) -> CropCandidate:
    vehicle_bbox = candidate.vehicle_bbox or candidate.bbox
    return candidate.model_copy(
        update={
            "total_score": _candidate_target_score(
                bbox_height=vehicle_bbox.height,
                sharpness=candidate.sharpness,
                edge_margin_score=candidate.edge_margin_score,
                area_score=candidate.area_score,
                frame_index=candidate.frame_index,
                min_box_height_px=min_box_height_px,
                max_box_height_px=max_box_height_px,
                target_ratio=config.analysis.crop_target_box_range_ratio,
            )
        }
    )


def _expand_crop_bbox(bbox: BBox, config: AppConfig) -> BBox:
    padding_ratio = config.analysis.crop_padding_ratio
    padding_px = config.analysis.crop_padding_px
    pad_x = (bbox.width * padding_ratio) + padding_px
    pad_y = (bbox.height * padding_ratio) + padding_px
    return BBox(
        x1=bbox.x1 - pad_x,
        y1=bbox.y1 - pad_y,
        x2=bbox.x2 + pad_x,
        y2=bbox.y2 + pad_y,
    )


def _save_candidate(
    store: RunStore,
    track_state: MutableTrackState,
    frame,
    bbox: BBox,
    frame_index: int,
    timestamp_seconds: float,
    config: AppConfig,
) -> None:
    clipped = clip_bbox_to_frame(_expand_crop_bbox(bbox, config), frame.shape)
    if clipped is None:
        return
    if (
        track_state.last_candidate_time is not None
        and timestamp_seconds - track_state.last_candidate_time
        < config.analysis.crop_min_spacing_seconds
    ):
        return
    crop = frame[int(clipped.y1) : int(clipped.y2), int(clipped.x1) : int(clipped.x2)]
    if crop.size == 0:
        return
    sharpness, edge_margin_score, area_score = _score_candidate(
        crop, clipped, frame.shape
    )
    track_dir = staged_track_crop_dir(store.crops_dir, track_state.track_id)
    track_dir.mkdir(parents=True, exist_ok=True)
    image_path = track_dir / f"frame_{frame_index:08d}.jpg"
    candidate = CropCandidate(
        track_id=track_state.track_id,
        vehicle_index=None,
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
        bbox=clipped,
        vehicle_bbox=bbox,
        image_path=image_path,
        sharpness=sharpness,
        edge_margin_score=edge_margin_score,
        area_score=area_score,
        total_score=_candidate_target_score(
            bbox_height=bbox.height,
            sharpness=sharpness,
            edge_margin_score=edge_margin_score,
            area_score=area_score,
            frame_index=frame_index,
            min_box_height_px=track_state.min_box_height_px,
            max_box_height_px=track_state.max_box_height_px,
            target_ratio=config.analysis.crop_target_box_range_ratio,
        ),
    )
    track_state.candidates = [
        _refresh_candidate_score(
            existing,
            track_state.min_box_height_px,
            track_state.max_box_height_px,
            config,
        )
        for existing in track_state.candidates
    ]
    current = track_state.candidates[0] if track_state.candidates else None
    if current is not None and _candidate_rank(
        current,
        track_state.min_box_height_px,
        track_state.max_box_height_px,
        config,
    ) >= _candidate_rank(
        candidate,
        track_state.min_box_height_px,
        track_state.max_box_height_px,
        config,
    ):
        return

    cv2.imwrite(
        str(image_path),
        crop,
        [cv2.IMWRITE_JPEG_QUALITY, config.analysis.crop_jpeg_quality],
    )
    if current is not None and current.image_path.exists():
        current.image_path.unlink()
    track_state.candidates = [candidate]
    track_state.last_candidate_time = timestamp_seconds


def _render_bbox_for_track(
    bbox: BBox, frame_shape: tuple[int, int, int], config: AppConfig
) -> BBox:
    return clip_bbox_to_frame(_expand_crop_bbox(bbox, config), frame_shape) or bbox


def _bbox_intersection_area(left: BBox, right: BBox) -> float:
    intersection_width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    intersection_height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    return intersection_width * intersection_height


def _bbox_iou(left: BBox, right: BBox) -> float:
    intersection_area = _bbox_intersection_area(left, right)
    if intersection_area <= 0:
        return 0.0
    union_area = left.area + right.area - intersection_area
    if union_area <= 0:
        return 0.0
    return intersection_area / union_area


def _bbox_contains_point(bbox: BBox, point: tuple[float, float]) -> bool:
    x, y = point
    return bbox.x1 <= x <= bbox.x2 and bbox.y1 <= y <= bbox.y2


def _track_matches_edge_detection(
    track_bbox: BBox, edge_detection_bboxes: SequenceABC[BBox]
) -> bool:
    for detection_bbox in edge_detection_bboxes:
        if _bbox_iou(track_bbox, detection_bbox) >= 0.05:
            return True
        if _bbox_contains_point(track_bbox, detection_bbox.center):
            return True
        if _bbox_contains_point(detection_bbox, track_bbox.center):
            return True
    return False


def _track_touches_suppression_edge(
    *,
    bbox: BBox,
    frame_shape: tuple[int, int, int],
    roi_shape: tuple[int, int, int],
    roi_offset: tuple[int, int],
    profile: CameraProfile,
    config: AppConfig,
) -> bool:
    roi_height, roi_width = roi_shape[:2]
    roi_left, roi_top = roi_offset
    margin = config.tracker.edge_margin_px
    return (
        bbox_touches_frame_edge(bbox, frame_shape, margin)
        or bbox_touches_rect_edge(
            bbox=bbox,
            left=float(roi_left),
            top=float(roi_top),
            right=float(roi_left + roi_width),
            bottom=float(roi_top + roi_height),
            margin_px=margin,
        )
        or bbox_touches_polygon_edge(bbox, profile.polygon.points, margin)
    )


def analyze_video(
    project_root: Path,
    config: AppConfig,
    profile: CameraProfile,
    video_path: Path,
    run_store: RunStore,
) -> RunStore:
    metadata = read_video_metadata(video_path)
    validate_video_fps(
        metadata=metadata,
        expected_fps=config.video.fps,
        tolerance=config.video.fps_tolerance,
    )
    analysis_fps = min(config.analysis.fps, config.video.fps)
    manifest = RunManifest(
        run_id=run_store.root.name,
        video_path=video_path,
        camera_id=profile.camera_id,
        root_dir=run_store.root,
        source_fps=config.video.fps,
        analysis_fps=analysis_fps,
        width=metadata.width,
        height=metadata.height,
        frame_count=metadata.frame_count,
    )
    run_store.manifest.write(manifest)

    detector = create_detector(config, project_root=project_root)
    tracker = BotSortAdapter(config, frame_rate=analysis_fps)
    diagnostics = AnalysisDiagnostics()
    track_states: dict[int, MutableTrackState] = {}
    count_line = profile.count_line
    line_start: tuple[float, float] | None = None
    line_end: tuple[float, float] | None = None
    if count_line is not None:
        line_start = (float(count_line.start[0]), float(count_line.start[1]))
        line_end = (float(count_line.end[0]), float(count_line.end[1]))

    for sampled_frame, detections in _iter_detected_sampled_frames(
        detector=detector,
        video_path=video_path,
        source_fps=config.video.fps,
        target_fps=analysis_fps,
        profile=profile,
        batch_size=config.analysis.batch_size,
    ):
        diagnostics.total_sampled_frames += 1
        frame_index = sampled_frame.frame_index
        timestamp_seconds = sampled_frame.timestamp_seconds
        frame = sampled_frame.frame
        roi_frame = sampled_frame.roi_frame
        offset = sampled_frame.offset
        global_detections = []
        for detection in detections:
            global_bbox = map_bbox_to_global(detection.bbox, offset)
            global_detections.append(detection.model_copy(update={"bbox": global_bbox}))
        edge_detection_bboxes = (
            [
                detection.bbox
                for detection in global_detections
                if _track_touches_suppression_edge(
                    bbox=detection.bbox,
                    frame_shape=frame.shape,
                    roi_shape=roi_frame.shape,
                    roi_offset=offset,
                    profile=profile,
                    config=config,
                )
            ]
            if config.tracker.ignore_edge_touches
            else []
        )

        diagnostics.detections_passed_to_tracker += len(global_detections)
        tracked = tracker.update(global_detections, frame)
        diagnostics.tracker_outputs += len(tracked.xyxy)
        frame_tracks: list[TrackedObject] = []
        tracker_ids = (
            tracked.tracker_id.tolist() if tracked.tracker_id is not None else []
        )
        class_ids = (
            tracked.class_id.tolist()
            if tracked.class_id is not None
            else [-1] * len(tracked.xyxy)
        )
        class_names = tracked.data.get("class_name", []) if tracked.data else []
        confidences = (
            tracked.confidence.tolist()
            if tracked.confidence is not None
            else [0.0] * len(tracked.xyxy)
        )
        diagnostics.tracker_confidence_values.extend(
            float(value) for value in confidences
        )

        for index, xyxy in enumerate(tracked.xyxy.tolist()):
            track_id = int(tracker_ids[index]) if index < len(tracker_ids) else -1
            bbox = BBox(x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3])
            touches_suppression_edge = config.tracker.ignore_edge_touches and (
                _track_touches_suppression_edge(
                    bbox=bbox,
                    frame_shape=frame.shape,
                    roi_shape=roi_frame.shape,
                    roi_offset=offset,
                    profile=profile,
                    config=config,
                )
                or _track_matches_edge_detection(bbox, edge_detection_bboxes)
            )
            if track_id < 0:
                if touches_suppression_edge:
                    diagnostics.edge_observations_skipped += 1
                    diagnostics.tracks_discarded_edge_contact += 1
                continue
            state = track_states.get(track_id)
            if touches_suppression_edge:
                diagnostics.edge_observations_skipped += 1
                diagnostics.tracks_discarded_edge_contact += 1
                continue
            bottom_center = bbox.bottom_center
            diagnostics.tracker_box_height_values.append(bbox.height)
            inside_roi = point_in_polygon(bottom_center, profile.polygon.points)
            confidence = float(confidences[index])
            class_id = int(class_ids[index]) if index < len(class_ids) else None
            class_name = str(class_names[index]) if index < len(class_names) else None

            if state is None:
                state = MutableTrackState(
                    track_id=track_id,
                    first_frame_index=frame_index,
                    last_frame_index=frame_index,
                )
                track_states[track_id] = state

            crossed_direction = None
            if count_line is None:
                if inside_roi and not state.counted:
                    state.counted = True
            elif state.previous_bottom_center is not None:
                assert line_start is not None
                assert line_end is not None
                crossed_direction = line_crossing_direction(
                    previous_point=state.previous_bottom_center,
                    current_point=bottom_center,
                    line_start=line_start,
                    line_end=line_end,
                )

            crossed_line = False
            if (
                count_line is not None
                and crossed_direction is not None
                and (
                    count_line.direction == "BOTH"
                    or crossed_direction == count_line.direction
                )
                and not state.counted
                and inside_roi
            ):
                state.counted = True
                state.count_event = CountEvent(
                    track_id=track_id,
                    frame_index=frame_index,
                    timestamp_seconds=timestamp_seconds,
                    direction=crossed_direction,
                )
                run_store.counts.append(state.count_event)
                crossed_line = True

            state.frames_seen += 1
            state.last_frame_index = frame_index
            state.min_box_height_px = (
                bbox.height
                if state.min_box_height_px is None
                else min(state.min_box_height_px, bbox.height)
            )
            state.max_box_height_px = max(state.max_box_height_px, bbox.height)
            state.previous_bottom_center = bottom_center

            if bbox.height >= config.analysis.min_box_height_px:
                _save_candidate(
                    store=run_store,
                    track_state=state,
                    frame=frame,
                    bbox=bbox,
                    frame_index=frame_index,
                    timestamp_seconds=timestamp_seconds,
                    config=config,
                )

            render_bbox = _render_bbox_for_track(bbox, frame.shape, config)
            render_centroid = render_bbox.center
            render_bottom_center = render_bbox.bottom_center

            tracked_object = TrackedObject(
                track_id=track_id,
                vehicle_index=None,
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                bbox=render_bbox,
                confidence=confidence,
                class_id=class_id,
                class_name=class_name,
                centroid=render_centroid,
                bottom_center=render_bottom_center,
                inside_roi=inside_roi,
                counted=state.counted,
                crossed_line=crossed_line,
            )
            frame_tracks.append(tracked_object)

        run_store.frames.append(
            FrameRecord(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                tracks=frame_tracks,
            )
        )

    all_track_states = list(track_states.values())
    all_track_states.sort(key=lambda item: (item.first_frame_index, item.track_id))
    diagnostics.tracks_without_crop_candidates = sum(
        1 for state in all_track_states if not state.candidates
    )
    diagnostics.tracks_without_crop_due_to_height = sum(
        1
        for state in all_track_states
        if not state.candidates
        and state.max_box_height_px < config.analysis.min_box_height_px
    )
    diagnostics.tracks_without_crop_due_to_short_lifetime = sum(
        1
        for state in all_track_states
        if config.analysis.min_track_frames > 0
        and state.frames_seen < config.analysis.min_track_frames
    )
    if config.render.require_crop_eligible_track:
        diagnostics.tracks_hidden_from_render_crop_eligibility = sum(
            1
            for state in all_track_states
            if not state.candidates
            and state.frames_seen >= config.render.min_visible_track_observations
        )
    if config.analysis.min_track_frames > 0:
        for state in all_track_states:
            if state.frames_seen < config.analysis.min_track_frames:
                diagnostics.tracks_discarded_min_track_frames += 1
                discard_track_artifacts(state, run_store.crops_dir)
                state.candidates = []
    vehicle_index_by_track = finalize_vehicle_identities(run_store, all_track_states)
    run_store.frames.rewrite_vehicle_indices(vehicle_index_by_track)
    for state in all_track_states:
        summary = track_summary_from_state(state)
        run_store.tracks.append(summary)
    run_store.detection_stats.write(
        _analysis_diagnostics_payload(diagnostics, detector)
    )

    logger.info("Analysis complete. Run directory: %s", run_store.root)
    return run_store
