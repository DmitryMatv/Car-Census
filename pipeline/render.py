from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path

import numpy as np

from config import AppConfig, CameraProfile
from mmr.make_country import MakeCountryCatalog, load_default_make_country_catalog
from mmr.powertrain_catalog import (
    PowertrainCatalog,
    PowertrainClass,
    load_default_powertrain_catalog,
    lookup_powertrain_class,
)
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


def format_label_text(
    result: MMRResult,
    unknown_label: str,
    origin_country_by_make: Mapping[str, str] | None = None,
) -> str:
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
    if result.make and origin_country_by_make:
        origin_flag = origin_country_by_make.get(result.make)
        if origin_flag:
            first_line = f"{origin_flag} {first_line}"
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


def _powertrain_text_color(
    config: AppConfig,
    powertrain_class: PowertrainClass | None,
) -> str | None:
    if powertrain_class == PowertrainClass.BEV:
        return config.render.label_bev_text_color
    if powertrain_class == PowertrainClass.MIXED:
        return config.render.label_mixed_text_color
    return None


def _label_text_and_colors_by_track(
    config: AppConfig,
    run_store: RunStore,
    frames_path: Path,
    allow_unclassified_annotations: bool,
    origin_country_by_make: Mapping[str, str],
    powertrain_catalog: PowertrainCatalog,
) -> tuple[dict[int, str], dict[int, str]]:
    labels = run_store.labels.read()
    if labels:
        accepted_labels = {
            track_id: result for track_id, result in labels.items() if result.accepted
        }
        label_text = {
            track_id: format_label_text(
                result,
                config.render.unknown_label,
                origin_country_by_make,
            )
            for track_id, result in accepted_labels.items()
        }
        label_text_colors = {
            track_id: text_color
            for track_id, result in accepted_labels.items()
            if (
                text_color := _powertrain_text_color(
                    config,
                    lookup_powertrain_class(powertrain_catalog, result),
                )
            )
            is not None
        }
        return label_text, label_text_colors
    if allow_unclassified_annotations or config.render.show_unclassified_tracks:
        return (
            visible_track_label_text_by_track(
                run_store.frames.iter(
                    smoothed=_uses_smoothed_frames(frames_path, run_store)
                ),
                config.render.unknown_label,
            ),
            {},
        )
    return {}, {}


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
    label_text_colors: dict[int, str],
    visible_track_ids: set[int],
) -> None:
    current_record = next(record_iter, None)
    latest_tracks: list[TrackedObject] = []
    counted_vehicle_indices: set[int] = set()

    for frame_index, _timestamp_seconds, frame in frame_iter:
        while current_record is not None and current_record.frame_index <= frame_index:
            latest_tracks = current_record.tracks
            current_record = next(record_iter, None)
        render_tracks = [
            track
            for track in latest_tracks
            if track.track_id in label_text and track.track_id in visible_track_ids
        ]
        for track in render_tracks:
            if track.counted and track.vehicle_index is not None:
                counted_vehicle_indices.add(track.vehicle_index)
        annotated = annotator.annotate(
            frame=frame,
            tracks=render_tracks,
            labels_by_track=label_text,
            counter_value=len(counted_vehicle_indices),
            label_text_colors_by_track=label_text_colors,
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
    origin_country_by_make: MakeCountryCatalog = load_default_make_country_catalog()
    powertrain_catalog: PowertrainCatalog = load_default_powertrain_catalog()
    label_text, label_text_colors = _label_text_and_colors_by_track(
        config,
        run_store,
        frames_path,
        allow_unclassified_annotations,
        origin_country_by_make,
        powertrain_catalog,
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
            label_text_colors=label_text_colors,
            visible_track_ids=visible_track_ids,
        )
    finally:
        writer.release()

    logger.info("Rendered annotated video to %s", run_store.output_video_path)
    return run_store.output_video_path
