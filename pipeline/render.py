from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import numpy as np

from config import AppConfig, CameraProfile
from models import FrameRecord, MMRResult, TrackedObject, TrackSummary
from render.annotators import VideoAnnotator
from storage.run_store import RunStore
from utils.video import (
    FrameWriter,
    VideoMetadata,
    build_frame_writer,
    iter_sampled_frames,
    iter_video_frames,
    read_video_metadata,
    validate_video_fps,
)

logger = logging.getLogger(__name__)

SmoothRenderTracks = Callable[[AppConfig, CameraProfile, RunStore], Path]


def _is_old_vehicle_placeholder(value: str | None) -> bool:
    return bool(value and value.strip().upper() == "OLD")


def _has_only_old_vehicle_details(result: MMRResult) -> bool:
    return all(
        _is_old_vehicle_placeholder(part)
        for part in [result.model, result.generation, result.variation]
    )


def format_label_text(result: MMRResult, unknown_label: str) -> str:
    has_only_old_vehicle_details = _has_only_old_vehicle_details(result)
    make_model = " ".join(
        part.strip()
        for part in [
            result.make or None,
            None if has_only_old_vehicle_details else result.model,
        ]
        if part and part.strip()
    ).strip()
    first_line = make_model or unknown_label
    parts = [first_line]
    detail_parts = (
        []
        if has_only_old_vehicle_details
        else [
            result.generation,
            result.variation,
        ]
    )
    for part in detail_parts:
        if part and part.strip():
            parts.append(part.strip())
    return "\n".join(parts)


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


def _read_validated_metadata(config: AppConfig, video_path: Path) -> VideoMetadata:
    metadata = read_video_metadata(video_path)
    validate_video_fps(
        metadata=metadata,
        expected_fps=config.video.fps,
        tolerance=config.video.fps_tolerance,
    )
    return metadata


def _resolve_render_frames_path(
    config: AppConfig,
    profile: CameraProfile,
    run_store: RunStore,
    smooth_render_tracks: SmoothRenderTracks | None,
) -> Path:
    if not config.render.smoothing.enabled:
        return run_store.frames_path
    if smooth_render_tracks is None:
        raise ValueError(
            "render smoothing is enabled but no smoothing stage was provided"
        )
    return smooth_render_tracks(config, profile, run_store)


def _uses_smoothed_frames(frames_path: Path, run_store: RunStore) -> bool:
    return frames_path == run_store.render_frames_path


def visible_track_ids_for_render(
    config: AppConfig,
    records: Iterable[FrameRecord],
    track_summaries: Iterable[TrackSummary] = (),
) -> set[int]:
    observation_counts: Counter[int] = Counter()
    crop_eligible_ids: set[int] = set()
    for record in records:
        for track in record.tracks:
            observation_counts.update([track.track_id])
            if track.vehicle_index is not None:
                crop_eligible_ids.add(track.track_id)

    visible_track_ids = {
        track_id
        for track_id, count in observation_counts.items()
        if count >= config.render.min_visible_track_observations
    }
    if (
        config.render.require_crop_eligible_track
        and not config.render.show_unclassified_tracks
    ):
        visible_track_ids &= crop_eligible_ids

    summaries = list(track_summaries)
    if summaries:
        size_eligible_ids = size_eligible_track_ids(config, summaries)
        visible_track_ids &= size_eligible_ids
    return visible_track_ids


def size_eligible_track_ids(
    config: AppConfig,
    summaries: Iterable[TrackSummary],
) -> set[int]:
    return {
        summary.track_id
        for summary in summaries
        if summary.max_box_width_px >= config.analysis.min_box_width_px
    }


def _label_text_by_track(
    config: AppConfig,
    run_store: RunStore,
    frames_path: Path,
    allow_unclassified_annotations: bool,
) -> dict[int, str]:
    labels = run_store.labels.read()
    if labels:
        return {
            track_id: format_label_text(result, config.render.unknown_label)
            for track_id, result in labels.items()
        }
    if allow_unclassified_annotations or config.render.show_unclassified_tracks:
        return visible_track_label_text_by_track(
            run_store.frames.iter(
                smoothed=_uses_smoothed_frames(frames_path, run_store)
            ),
            config.render.unknown_label,
        )
    return {}


def _output_fps(config: AppConfig) -> float:
    return min(config.render.output_fps or config.video.fps, config.video.fps)


def _build_render_writer(
    config: AppConfig,
    run_store: RunStore,
    metadata: VideoMetadata,
    output_fps: float,
) -> FrameWriter:
    return build_frame_writer(
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


def _iter_render_frames(
    config: AppConfig,
    video_path: Path,
    output_fps: float,
) -> Iterator[tuple[int, float, np.ndarray]]:
    if output_fps < config.video.fps:
        return iter_sampled_frames(
            video_path,
            source_fps=config.video.fps,
            target_fps=output_fps,
        )
    return iter_video_frames(video_path, config.video.fps)


def _render_annotated_frames(
    *,
    frame_iter: Iterator[tuple[int, float, np.ndarray]],
    record_iter: Iterator[FrameRecord],
    annotator: VideoAnnotator,
    writer: FrameWriter,
    label_text: dict[int, str],
    visible_track_ids: set[int],
) -> None:
    current_record = next(record_iter, None)
    latest_tracks: list[TrackedObject] = []
    live_count = 0

    for frame_index, _timestamp_seconds, frame in frame_iter:
        while current_record is not None and current_record.frame_index <= frame_index:
            latest_tracks = current_record.tracks
            current_record = next(record_iter, None)
        for track in latest_tracks:
            if track.counted and track.vehicle_index is not None:
                live_count = max(live_count, track.vehicle_index)
        render_tracks = [
            track
            for track in latest_tracks
            if track.track_id in label_text and track.track_id in visible_track_ids
        ]
        annotated = annotator.annotate(
            frame=frame,
            tracks=render_tracks,
            labels_by_track=label_text,
            counter_value=live_count,
        )
        writer.write(annotated)


def render_video(
    config: AppConfig,
    profile: CameraProfile,
    video_path: Path,
    run_store: RunStore,
    allow_unclassified_annotations: bool = False,
    smooth_render_tracks: SmoothRenderTracks | None = None,
) -> Path:
    metadata = _read_validated_metadata(config, video_path)
    track_summaries = run_store.tracks.read_all()
    visible_track_ids = visible_track_ids_for_render(
        config,
        run_store.frames.iter(smoothed=False),
        track_summaries=track_summaries,
    )
    frames_path = _resolve_render_frames_path(
        config, profile, run_store, smooth_render_tracks
    )
    label_text = _label_text_by_track(
        config, run_store, frames_path, allow_unclassified_annotations
    )
    annotator = VideoAnnotator(config)
    output_fps = _output_fps(config)
    writer = _build_render_writer(config, run_store, metadata, output_fps)

    try:
        _render_annotated_frames(
            frame_iter=_iter_render_frames(config, video_path, output_fps),
            record_iter=run_store.frames.iter(
                smoothed=_uses_smoothed_frames(frames_path, run_store)
            ),
            annotator=annotator,
            writer=writer,
            label_text=label_text,
            visible_track_ids=visible_track_ids,
        )
    finally:
        writer.release()

    logger.info("Rendered annotated video to %s", run_store.output_video_path)
    return run_store.output_video_path
