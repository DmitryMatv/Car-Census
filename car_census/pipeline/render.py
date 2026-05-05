from __future__ import annotations

import logging
from pathlib import Path

import orjson

from car_census.config import AppConfig, CameraProfile
from car_census.render.annotators import VideoAnnotator
from car_census.storage.run_store import RunStore
from car_census.types import FrameRecord, MMRResult
from car_census.utils.video import build_video_writer, iter_sampled_frames, read_video_metadata

logger = logging.getLogger(__name__)


def _load_labels(path: Path) -> dict[int, MMRResult]:
    if not path.exists():
        return {}
    raw = orjson.loads(path.read_bytes())
    return {int(track_id): MMRResult.model_validate(payload) for track_id, payload in raw.items()}


def _iter_frame_records(path: Path):
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                yield FrameRecord.model_validate(orjson.loads(line))


def render_video(
    config: AppConfig,
    profile: CameraProfile,
    video_path: Path,
    run_store: RunStore,
) -> Path:
    metadata = read_video_metadata(video_path)
    manifest = run_store.read_manifest()
    output_fps = metadata.fps if config.render.output_fps <= 0 else config.render.output_fps
    render_sampling_fps = manifest.analysis_fps if manifest.analysis_fps > 0 else metadata.fps
    labels = _load_labels(run_store.labels_path)
    label_text = {
        track_id: " ".join(
            part for part in [result.make or None, result.model or None] if part
        ).strip()
        or config.render.unknown_label
        for track_id, result in labels.items()
    }
    annotator = VideoAnnotator(config)
    writer = build_video_writer(
        output_path=run_store.output_video_path,
        fps=output_fps,
        width=metadata.width,
        height=metadata.height,
        codec=config.render.codec,
    )
    record_iter = iter(_iter_frame_records(run_store.frames_path))
    current_record = next(record_iter, None)

    try:
        for frame_index, _timestamp_seconds, frame in iter_sampled_frames(video_path, render_sampling_fps):
            tracks = []
            if current_record is not None and current_record.frame_index == frame_index:
                tracks = current_record.tracks
                current_record = next(record_iter, None)
            annotated = annotator.annotate(frame=frame, profile=profile, tracks=tracks, labels_by_track=label_text)
            writer.write(annotated)
    finally:
        writer.release()

    logger.info("Rendered annotated video to %s", run_store.output_video_path)
    return run_store.output_video_path
