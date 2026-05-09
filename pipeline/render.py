from __future__ import annotations

import logging
from pathlib import Path

import orjson

from config import AppConfig, CameraProfile
from render.annotators import VideoAnnotator
from storage.run_store import RunStore
from models import FrameRecord, MMRResult
from pipeline.smooth import smooth_render_tracks
from utils.video import (
    build_video_writer,
    iter_sampled_frames,
    read_video_metadata,
)

logger = logging.getLogger(__name__)


def _load_labels(path: Path) -> dict[int, MMRResult]:
    if not path.exists():
        return {}
    raw = orjson.loads(path.read_bytes())
    return {
        int(track_id): MMRResult.model_validate(payload)
        for track_id, payload in raw.items()
    }


def _iter_frame_records(path: Path):
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                yield FrameRecord.model_validate(orjson.loads(line))


def format_label_text(result: MMRResult, unknown_label: str) -> str:
    base_label = (
        " ".join(
            part for part in [result.make or None, result.model or None] if part
        ).strip()
        or unknown_label
    )
    label_index = result.vehicle_index or result.api_classification_index
    if label_index is None:
        return base_label
    return f"{label_index} | {base_label}"


def visible_track_label_text_by_track(
    frames_path: Path, unknown_label: str
) -> dict[int, str]:
    records = list(_iter_frame_records(frames_path))
    has_vehicle_indices = any(
        track.vehicle_index is not None for record in records for track in record.tracks
    )
    labels: dict[int, str] = {}
    fallback_index_by_track: dict[int, int] = {}
    for record in records:
        for track in record.tracks:
            if track.track_id in labels:
                continue
            label_index = track.vehicle_index
            if label_index is None:
                if has_vehicle_indices:
                    continue
                label_index = fallback_index_by_track.setdefault(
                    track.track_id, len(fallback_index_by_track) + 1
                )
            labels[track.track_id] = format_label_text(
                MMRResult(vehicle_index=label_index),
                unknown_label,
            )
    return labels


def render_video(
    config: AppConfig,
    profile: CameraProfile,
    video_path: Path,
    run_store: RunStore,
) -> Path:
    metadata = read_video_metadata(video_path)
    manifest = run_store.read_manifest()
    output_fps = (
        metadata.fps if config.render.output_fps <= 0 else config.render.output_fps
    )
    _ = manifest
    labels = _load_labels(run_store.labels_path)
    frames_path = (
        smooth_render_tracks(config=config, profile=profile, run_store=run_store)
        if config.render.smoothing.enabled
        else run_store.frames_path
    )
    label_text = visible_track_label_text_by_track(
        frames_path, config.render.unknown_label
    )
    label_text.update(
        {
            track_id: format_label_text(result, config.render.unknown_label)
            for track_id, result in labels.items()
        }
    )
    annotator = VideoAnnotator(config)
    writer = build_video_writer(
        output_path=run_store.output_video_path,
        fps=output_fps,
        width=metadata.width,
        height=metadata.height,
        codec=config.render.codec,
    )
    record_iter = iter(_iter_frame_records(frames_path))
    current_record = next(record_iter, None)
    latest_tracks = []

    try:
        for frame_index, _timestamp_seconds, frame in iter_sampled_frames(
            video_path, metadata.fps
        ):
            while (
                current_record is not None and current_record.frame_index <= frame_index
            ):
                latest_tracks = current_record.tracks
                current_record = next(record_iter, None)
            render_tracks = [
                track for track in latest_tracks if track.track_id in label_text
            ]
            annotated = annotator.annotate(
                frame=frame,
                profile=profile,
                tracks=render_tracks,
                labels_by_track=label_text,
            )
            writer.write(annotated)
    finally:
        writer.release()

    logger.info("Rendered annotated video to %s", run_store.output_video_path)
    return run_store.output_video_path
