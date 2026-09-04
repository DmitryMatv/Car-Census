from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

from config import AppConfig, CameraProfile
from detectors.base import Detector
from detectors.factory import create_detector
from models import RunManifest
from pipeline.analysis_crops import CropCandidateSelector
from pipeline.analysis_diagnostics import (
    AnalysisDiagnostics,
    analysis_diagnostics_payload,
)
from pipeline.analysis_edges import EdgeSuppression
from pipeline.analysis_frames import (
    iter_detected_sampled_frames,
    map_detections_to_global,
)
from pipeline.analysis_track_state import MutableTrackState
from pipeline.analysis_tracking import (
    FrameTrackingInput,
    build_track_state_updater,
)
from pipeline.detections import cached_frame_detections
from pipeline.link import link_analysis_tracks
from pipeline.vehicles import (
    compute_track_world_speeds,
    discard_track_artifacts,
    finalize_vehicle_identities,
    track_summary_from_state,
)
from roi.transform import ViewTransformer, build_view_transformer
from storage.json_artifacts import write_json
from storage.run_store import RunStore
from tracking_adapters.botsort import BotSortAdapter
from utils.video import iter_sampled_frames, read_video_metadata, validate_video_fps

logger = logging.getLogger(__name__)


def _iter_analysis_sampled_frames(
    video_path: Path,
    source_fps: float,
    target_fps: float,
) -> Iterator[tuple[int, float, np.ndarray]]:
    return iter_sampled_frames(
        video_path,
        source_fps,
        target_fps,
        include_terminal_frame=True,
    )


def _finalize_analysis(
    *,
    run_store: RunStore,
    track_states: Sequence[MutableTrackState],
    diagnostics: AnalysisDiagnostics,
    detector: Detector,
    config: AppConfig,
    view_transformer: ViewTransformer | None,
) -> None:
    diagnostics.tracks_without_crop_candidates = sum(
        1 for state in track_states if not state.candidates
    )
    diagnostics.tracks_without_crop_due_to_width = sum(
        1
        for state in track_states
        if not state.candidates
        and state.max_box_width_px < config.analysis.min_box_width_px
    )
    diagnostics.tracks_without_crop_due_to_height = sum(
        1
        for state in track_states
        if not state.candidates
        and state.max_box_width_px >= config.analysis.min_box_width_px
        and state.max_box_height_px < config.analysis.min_box_height_px
    )
    diagnostics.tracks_without_crop_due_to_short_lifetime = sum(
        1
        for state in track_states
        if config.analysis.min_track_frames > 0
        and state.frames_seen < config.analysis.min_track_frames
    )
    if config.render.require_crop_eligible_track:
        diagnostics.tracks_hidden_from_render_crop_eligibility = sum(
            1
            for state in track_states
            if not state.candidates
            and state.frames_seen >= config.render.min_visible_track_observations
        )
    if config.analysis.min_track_frames > 0:
        for state in track_states:
            if state.frames_seen < config.analysis.min_track_frames:
                diagnostics.tracks_discarded_min_track_frames += 1
                discard_track_artifacts(state, run_store.crops_dir)
                state.candidates = []

    vehicle_index_by_track = finalize_vehicle_identities(run_store, track_states)
    run_store.frames.rewrite_vehicle_indices(vehicle_index_by_track)
    world_speeds = compute_track_world_speeds(
        run_store.frames.read_all(smoothed=False), view_transformer
    )
    run_store.tracks.write_all(
        track_summary_from_state(state, world_speeds.get(state.track_id))
        for state in track_states
    )
    run_store.detection_stats.write(analysis_diagnostics_payload(diagnostics, detector))


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
        video_path=video_path.expanduser().resolve(),
        camera_id=profile.camera_id,
        root_dir=run_store.root,
        source_fps=config.video.fps,
        analysis_fps=analysis_fps,
        width=metadata.width,
        height=metadata.height,
        frame_count=metadata.frame_count,
        retrieval_cache_dir=(
            project_root
            / config.project.output_root
            / config.project.retrieval_cache_dir
        ).resolve(),
    )
    run_store.manifest.write(manifest)

    detector = create_detector(config, project_root=project_root)
    tracker = BotSortAdapter(
        config,
        frame_rate=analysis_fps,
        view_transformer=build_view_transformer(config, profile),
    )
    diagnostics = AnalysisDiagnostics()
    crop_selector = CropCandidateSelector(config, run_store)
    edge_suppression = EdgeSuppression(config, profile)
    track_updater = build_track_state_updater(
        config=config,
        profile=profile,
        crop_selector=crop_selector,
        edge_suppression=edge_suppression,
        diagnostics=diagnostics,
    )

    with (
        run_store.frames.open_writer() as frame_writer,
        run_store.counts.open_writer() as count_writer,
        run_store.detections.open_writer() as detection_writer,
    ):
        for sampled_frame, detections in iter_detected_sampled_frames(
            detector=detector,
            video_path=video_path,
            source_fps=config.video.fps,
            target_fps=analysis_fps,
            profile=profile,
            batch_size=(
                config.analysis.detector_batch_size or config.analysis.batch_size
            ),
            iter_sampled_frames_func=_iter_analysis_sampled_frames,
        ):
            diagnostics.total_sampled_frames += 1
            global_detections = map_detections_to_global(
                detections, sampled_frame.offset
            )
            edge_detection_bboxes = edge_suppression.detection_edge_bboxes(
                global_detections,
                frame_shape=sampled_frame.frame.shape,
                roi_shape=sampled_frame.roi_frame.shape,
                roi_offset=sampled_frame.offset,
            )
            detection_writer.write(
                cached_frame_detections(
                    frame_index=sampled_frame.frame_index,
                    timestamp_seconds=sampled_frame.timestamp_seconds,
                    detections=global_detections,
                    edge_suppressed_bboxes=edge_detection_bboxes,
                )
            )

            diagnostics.detections_passed_to_tracker += len(global_detections)
            tracked = tracker.update(
                global_detections,
                sampled_frame.frame,
                timestamp=sampled_frame.timestamp_seconds,
            )
            update_result = track_updater.process_tracker_outputs(
                tracked=tracked,
                frame_input=FrameTrackingInput(
                    frame_index=sampled_frame.frame_index,
                    timestamp_seconds=sampled_frame.timestamp_seconds,
                    frame=sampled_frame.frame,
                    roi_frame=sampled_frame.roi_frame,
                    roi_offset=sampled_frame.offset,
                    detections=global_detections,
                ),
                edge_detection_bboxes=edge_detection_bboxes,
            )
            if update_result.tracker_ids_to_drop:
                tracker.drop_tracks(update_result.tracker_ids_to_drop)
            frame_writer.write(update_result.frame_record)
            for count_event in update_result.counted_events:
                count_writer.write(count_event)

    audit_payload = getattr(tracker, "rescue_audit_payload", None)
    if callable(audit_payload):
        write_json(
            run_store.analysis_dir / "rescue_reassociations.json",
            audit_payload(),
        )
    _finalize_analysis(
        run_store=run_store,
        track_states=track_updater.sorted_track_states(),
        diagnostics=diagnostics,
        detector=detector,
        config=config,
        view_transformer=build_view_transformer(config, profile),
    )
    link_analysis_tracks(config=config, profile=profile, run_store=run_store)

    logger.info("Analysis complete. Run directory: %s", run_store.root)
    return run_store
