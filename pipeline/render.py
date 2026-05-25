from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import numpy as np

from config import AppConfig, CameraProfile
from models import FrameRecord, MMRResult, TrackedObject
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


def _visible_track_ids(config: AppConfig, records: Iterable[FrameRecord]) -> set[int]:
    records = list(records)
    visible_track_ids = visible_track_ids_by_observation_count(
        records,
        config.render.min_visible_track_observations,
    )
    if (
        config.render.require_crop_eligible_track
        and not config.render.show_unclassified_tracks
    ):
        visible_track_ids &= crop_eligible_track_ids(records)
    return visible_track_ids


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

    for frame_index, _timestamp_seconds, frame in frame_iter:
        while current_record is not None and current_record.frame_index <= frame_index:
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


def render_video(
    config: AppConfig,
    profile: CameraProfile,
    video_path: Path,
    run_store: RunStore,
    allow_unclassified_annotations: bool = False,
    smooth_render_tracks: SmoothRenderTracks | None = None,
) -> Path:
    metadata = _read_validated_metadata(config, video_path)
    raw_records = run_store.frames.read_all(smoothed=False)
    visible_track_ids = _visible_track_ids(config, raw_records)
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
