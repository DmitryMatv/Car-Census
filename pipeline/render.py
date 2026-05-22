from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from config import AppConfig, CameraProfile
from models import FrameRecord, MMRResult
from pipeline.smooth import smooth_render_tracks
from render.annotators import VideoAnnotator
from storage.run_store import RunStore
from utils.video import (
    build_frame_writer,
    iter_sampled_frames,
    iter_video_frames,
    read_video_metadata,
    validate_video_fps,
)

logger = logging.getLogger(__name__)


def format_label_text(result: MMRResult, unknown_label: str) -> str:
    make_model = " ".join(
        part.strip()
        for part in [result.make or None, result.model or None]
        if part and part.strip()
    ).strip()
    parts = [make_model or unknown_label]
    for part in [result.generation, result.variation]:
        if part and part.strip():
            parts.append(part.strip())
    if result.vehicle_index is not None:
        parts.insert(0, str(result.vehicle_index))
    return " | ".join(parts)


def visible_track_label_text_by_track(
    records: Iterable[FrameRecord], unknown_label: str
) -> dict[int, str]:
    labels: dict[int, str] = {}
    for record in records:
        for track in record.tracks:
            if track.track_id in labels or track.vehicle_index is None:
                continue
            labels[track.track_id] = format_label_text(
                MMRResult(vehicle_index=track.vehicle_index),
                unknown_label,
            )
    return labels


def visible_track_ids_by_observation_count(
    records: Iterable[FrameRecord], min_observations: int
) -> set[int]:
    counts: Counter[int] = Counter()
    for record in records:
        counts.update(track.track_id for track in record.tracks)
    return {track_id for track_id, count in counts.items() if count >= min_observations}


def crop_eligible_track_ids(records: Iterable[FrameRecord]) -> set[int]:
    eligible: set[int] = set()
    for record in records:
        for track in record.tracks:
            if track.vehicle_index is not None:
                eligible.add(track.track_id)
    return eligible


def render_video(
    config: AppConfig,
    profile: CameraProfile,
    video_path: Path,
    run_store: RunStore,
    allow_unclassified_annotations: bool = False,
) -> Path:
    metadata = read_video_metadata(video_path)
    validate_video_fps(
        metadata=metadata,
        expected_fps=config.video.fps,
        tolerance=config.video.fps_tolerance,
    )
    labels = run_store.read_labels()
    raw_records = run_store.read_frame_records(smoothed=False)
    visible_track_ids = visible_track_ids_by_observation_count(
        raw_records,
        config.render.min_visible_track_observations,
    )
    if (
        config.render.require_crop_eligible_track
        and not config.render.show_unclassified_tracks
    ):
        visible_track_ids &= crop_eligible_track_ids(raw_records)
    frames_path = (
        smooth_render_tracks(config=config, profile=profile, run_store=run_store)
        if config.render.smoothing.enabled
        else run_store.frames_path
    )
    if labels:
        label_text = {
            track_id: format_label_text(result, config.render.unknown_label)
            for track_id, result in labels.items()
        }
    elif allow_unclassified_annotations or config.render.show_unclassified_tracks:
        label_text = visible_track_label_text_by_track(
            run_store.iter_frame_records(
                smoothed=frames_path == run_store.render_frames_path
            ),
            config.render.unknown_label,
        )
    else:
        label_text = {}
    annotator = VideoAnnotator(config)
    output_fps = min(config.render.output_fps or config.video.fps, config.video.fps)
    writer = build_frame_writer(
        output_path=run_store.output_video_path,
        fps=output_fps,
        width=metadata.width,
        height=metadata.height,
        codec=config.render.codec,
        encode_backend=config.render.encode_backend,
        ffmpeg_path=config.render.ffmpeg_path,
        nvenc_codec=config.render.nvenc_codec,
        nvenc_preset=config.render.nvenc_preset,
        nvenc_cq=config.render.nvenc_cq,
    )
    record_iter = run_store.iter_frame_records(
        smoothed=frames_path == run_store.render_frames_path
    )
    current_record = next(record_iter, None)
    latest_tracks = []
    frame_iter = (
        iter_sampled_frames(
            video_path,
            source_fps=config.video.fps,
            target_fps=output_fps,
        )
        if output_fps < config.video.fps
        else iter_video_frames(video_path, config.video.fps)
    )

    try:
        for frame_index, _timestamp_seconds, frame in frame_iter:
            while (
                current_record is not None and current_record.frame_index <= frame_index
            ):
                latest_tracks = current_record.tracks
                current_record = next(record_iter, None)
            render_tracks = [
                track
                for track in latest_tracks
                if track.track_id in label_text and track.track_id in visible_track_ids
            ]
            annotated = annotator.annotate(
                frame=frame,
                tracks=render_tracks,
                labels_by_track=label_text,
            )
            writer.write(annotated)
    finally:
        writer.release()

    logger.info("Rendered annotated video to %s", run_store.output_video_path)
    return run_store.output_video_path
