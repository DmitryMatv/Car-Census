from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from car_census.config import AppConfig, CameraProfile
from car_census.detectors.base import create_detector
from car_census.roi.geometry import (
    clip_bbox_to_frame,
    crop_to_polygon,
    line_crossing_direction,
    map_bbox_to_global,
    point_in_polygon,
)
from car_census.storage.run_store import RunStore
from car_census.trackers.bytetrack import ByteTrackAdapter
from car_census.types import BBox, CountEvent, CropCandidate, FrameRecord, RunManifest, TrackSummary, TrackedObject
from car_census.utils.image_quality import laplacian_sharpness
from car_census.utils.video import iter_sampled_frames, read_video_metadata

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MutableTrackState:
    track_id: int
    first_frame_index: int
    last_frame_index: int
    frames_seen: int = 0
    max_box_height_px: float = 0.0
    previous_bottom_center: tuple[float, float] | None = None
    counted: bool = False
    count_event: CountEvent | None = None
    candidates: list[CropCandidate] = field(default_factory=list)
    last_candidate_time: float | None = None


def _score_candidate(crop: cv2.typing.MatLike, bbox: BBox, frame_shape: tuple[int, int, int]) -> tuple[float, float, float, float]:
    sharpness = laplacian_sharpness(crop)
    area_score = bbox.area
    height, width = frame_shape[:2]
    margin_left = bbox.x1
    margin_top = bbox.y1
    margin_right = width - bbox.x2
    margin_bottom = height - bbox.y2
    edge_margin_score = min(margin_left, margin_top, margin_right, margin_bottom)
    total_score = area_score + (sharpness * 2.0) + (edge_margin_score * 10.0)
    return sharpness, edge_margin_score, area_score, total_score


def _save_candidate(
    store: RunStore,
    track_state: MutableTrackState,
    frame,
    bbox: BBox,
    frame_index: int,
    timestamp_seconds: float,
    config: AppConfig,
) -> None:
    clipped = clip_bbox_to_frame(bbox, frame.shape)
    if clipped is None:
        return
    if (
        track_state.last_candidate_time is not None
        and timestamp_seconds - track_state.last_candidate_time < config.analysis.crop_min_spacing_seconds
    ):
        return
    crop = frame[int(clipped.y1) : int(clipped.y2), int(clipped.x1) : int(clipped.x2)]
    if crop.size == 0:
        return
    sharpness, edge_margin_score, area_score, total_score = _score_candidate(crop, clipped, frame.shape)
    track_dir = store.crops_dir / f"track_{track_state.track_id:06d}"
    track_dir.mkdir(parents=True, exist_ok=True)
    image_path = track_dir / f"frame_{frame_index:08d}.jpg"
    cv2.imwrite(
        str(image_path),
        crop,
        [int(cv2.IMWRITE_JPEG_QUALITY), config.analysis.crop_jpeg_quality],
    )
    candidate = CropCandidate(
        track_id=track_state.track_id,
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
        bbox=clipped,
        image_path=image_path,
        sharpness=sharpness,
        edge_margin_score=edge_margin_score,
        area_score=area_score,
        total_score=total_score,
    )
    track_state.candidates.append(candidate)
    track_state.candidates.sort(key=lambda item: item.total_score, reverse=True)
    if len(track_state.candidates) > config.analysis.crop_limit_per_track:
        removed = track_state.candidates.pop()
        if removed.image_path.exists():
            removed.image_path.unlink()
    track_state.last_candidate_time = timestamp_seconds


def analyze_video(
    project_root: Path,
    config: AppConfig,
    profile: CameraProfile,
    video_path: Path,
    run_store: RunStore,
) -> RunStore:
    metadata = read_video_metadata(video_path)
    analysis_fps = metadata.fps if config.analysis.fps <= 0 else min(config.analysis.fps, metadata.fps)
    manifest = RunManifest(
        run_id=run_store.root.name,
        video_path=video_path,
        camera_id=profile.camera_id,
        root_dir=run_store.root,
        source_fps=metadata.fps,
        analysis_fps=analysis_fps,
        width=metadata.width,
        height=metadata.height,
    )
    run_store.write_manifest(manifest)

    detector = create_detector(config, project_root=project_root)
    tracker = ByteTrackAdapter(config, frame_rate=analysis_fps)
    track_states: dict[int, MutableTrackState] = {}
    line_start = tuple(profile.count_line.start)
    line_end = tuple(profile.count_line.end)

    for frame_index, timestamp_seconds, frame in iter_sampled_frames(video_path, analysis_fps):
        roi_frame, offset = crop_to_polygon(frame, profile.polygon.points)
        detections = detector.detect(roi_frame)
        global_detections = []
        for detection in detections:
            global_bbox = map_bbox_to_global(detection.bbox, offset)
            global_detections.append(
                detection.model_copy(update={"bbox": global_bbox})
            )

        tracked = tracker.update(global_detections)
        frame_tracks: list[TrackedObject] = []
        tracker_ids = tracked.tracker_id.tolist() if tracked.tracker_id is not None else []
        class_ids = tracked.class_id.tolist() if tracked.class_id is not None else [-1] * len(tracked.xyxy)
        class_names = tracked.data.get("class_name", []) if tracked.data else []
        confidences = tracked.confidence.tolist() if tracked.confidence is not None else [0.0] * len(tracked.xyxy)

        for index, xyxy in enumerate(tracked.xyxy.tolist()):
            track_id = int(tracker_ids[index])
            bbox = BBox(x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3])
            centroid = bbox.center
            bottom_center = ((bbox.x1 + bbox.x2) / 2.0, bbox.y2)
            inside_roi = point_in_polygon(bottom_center, profile.polygon.points)

            state = track_states.get(track_id)
            if state is None:
                state = MutableTrackState(
                    track_id=track_id,
                    first_frame_index=frame_index,
                    last_frame_index=frame_index,
                )
                track_states[track_id] = state

            crossed_direction = None
            if state.previous_bottom_center is not None:
                crossed_direction = line_crossing_direction(
                    previous_point=state.previous_bottom_center,
                    current_point=bottom_center,
                    line_start=line_start,
                    line_end=line_end,
                )

            crossed_line = False
            if (
                crossed_direction is not None
                and (
                    profile.count_line.direction == "BOTH"
                    or crossed_direction == profile.count_line.direction
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

            tracked_object = TrackedObject(
                track_id=track_id,
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                bbox=bbox,
                confidence=float(confidences[index]),
                class_id=int(class_ids[index]) if index < len(class_ids) else None,
                class_name=str(class_names[index]) if index < len(class_names) else None,
                centroid=centroid,
                bottom_center=bottom_center,
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

    for state in track_states.values():
        summary = TrackSummary(
            track_id=state.track_id,
            first_frame_index=state.first_frame_index,
            last_frame_index=state.last_frame_index,
            frames_seen=state.frames_seen,
            max_box_height_px=state.max_box_height_px,
            counted=state.counted,
            count_event=state.count_event,
            candidates=state.candidates,
        )
        run_store.write_jsonl(run_store.tracks_path, summary.model_dump(mode="json"))

    logger.info("Analysis complete. Run directory: %s", run_store.root)
    return run_store
