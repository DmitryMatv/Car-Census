from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2

from config import AppConfig, CameraProfile
from detectors.base import Detector, create_detector
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
    rewrite_frame_vehicle_indices,
    staged_track_crop_dir,
    track_summary_from_state,
)
from roi.geometry import (
    bbox_touches_frame_edge,
    bbox_touches_polygon_edge,
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
    run_store.write_manifest(manifest)

    detector = create_detector(config, project_root=project_root)
    tracker = BotSortAdapter(config, frame_rate=analysis_fps)
    track_states: dict[int, MutableTrackState] = {}
    finished_track_states: list[MutableTrackState] = []
    suppressed_edge_track_ids: set[int] = set()
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
        frame_index = sampled_frame.frame_index
        timestamp_seconds = sampled_frame.timestamp_seconds
        frame = sampled_frame.frame
        roi_frame = sampled_frame.roi_frame
        offset = sampled_frame.offset
        global_detections = []
        for detection in detections:
            if config.tracker.ignore_edge_touches and bbox_touches_frame_edge(
                detection.bbox, roi_frame.shape, config.tracker.edge_margin_px
            ):
                continue
            global_bbox = map_bbox_to_global(detection.bbox, offset)
            if config.tracker.ignore_edge_touches and bbox_touches_frame_edge(
                global_bbox, frame.shape, config.tracker.edge_margin_px
            ):
                continue
            if config.tracker.ignore_edge_touches and bbox_touches_polygon_edge(
                global_bbox, profile.polygon.points, config.tracker.edge_margin_px
            ):
                continue
            global_detections.append(detection.model_copy(update={"bbox": global_bbox}))

        tracked = tracker.update(global_detections, frame)
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

        for index, xyxy in enumerate(tracked.xyxy.tolist()):
            track_id = int(tracker_ids[index])
            if track_id < 0:
                continue
            if track_id in suppressed_edge_track_ids:
                continue
            bbox = BBox(x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3])
            state = track_states.get(track_id)
            if config.tracker.ignore_edge_touches:
                touches_source_edge = bbox_touches_frame_edge(
                    bbox, frame.shape, config.tracker.edge_margin_px
                )
                touches_roi_edge = bbox_touches_polygon_edge(
                    bbox, profile.polygon.points, config.tracker.edge_margin_px
                )
                if touches_source_edge or touches_roi_edge:
                    suppressed_edge_track_ids.add(track_id)
                    state = track_states.pop(track_id, None)
                    if state is None or state.frames_seen == 0:
                        discard_track_artifacts(state, run_store.crops_dir)
                    else:
                        finished_track_states.append(state)
                    continue
            bottom_center = bbox.bottom_center
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
                run_store.write_jsonl(
                    run_store.count_events_path,
                    state.count_event.model_dump(mode="json"),
                )
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

        run_store.write_jsonl(
            run_store.frames_path,
            FrameRecord(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                tracks=frame_tracks,
            ).model_dump(mode="json"),
        )

    all_track_states = [*finished_track_states, *track_states.values()]
    all_track_states.sort(key=lambda item: (item.first_frame_index, item.track_id))
    if config.analysis.min_track_frames > 0:
        for state in all_track_states:
            if state.frames_seen < config.analysis.min_track_frames:
                discard_track_artifacts(state, run_store.crops_dir)
                state.candidates = []
    vehicle_index_by_track = finalize_vehicle_identities(run_store, all_track_states)
    rewrite_frame_vehicle_indices(run_store.frames_path, vehicle_index_by_track)
    for state in all_track_states:
        summary = track_summary_from_state(state)
        run_store.write_jsonl(run_store.tracks_path, summary.model_dump(mode="json"))

    logger.info("Analysis complete. Run directory: %s", run_store.root)
    return run_store
