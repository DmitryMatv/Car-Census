from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from pathlib import Path

import supervision as sv

from config import AppConfig, CameraProfile
from detectors.base import Detector
from detectors.factory import create_detector
from models import RunManifest
from pipeline.analysis_crops import (
    CropCandidateSelector,
    candidate_rank,
    candidate_target_score,
    expand_crop_bbox,
    refresh_candidate_score,
    render_bbox_for_track,
    save_candidate,
    score_candidate,
)
from pipeline.analysis_diagnostics import (
    AnalysisDiagnostics,
    analysis_diagnostics_payload,
    detector_diagnostics,
    diagnostic_count,
    diagnostic_float_values,
    histogram,
)
from pipeline.analysis_edges import (
    EdgeSuppression,
    bbox_contains_point,
    bbox_intersection_area,
    bbox_iou,
    track_matches_edge_detection,
    track_touches_suppression_edge,
)
from pipeline.analysis_frames import (
    SampledFrame,
    iter_detected_sampled_frames,
    iter_sampled_frame_batches,
    map_detections_to_global,
)
from pipeline.analysis_tracking import (
    FrameTrackingInput,
    MutableTrackState,
    TrackStateUpdater,
)
from pipeline.vehicles import (
    discard_track_artifacts,
    finalize_vehicle_identities,
    track_summary_from_state,
)
from storage.run_store import RunStore
from tracking_adapters.botsort import BotSortAdapter
from utils.video import iter_sampled_frames, read_video_metadata, validate_video_fps

logger = logging.getLogger(__name__)


# Compatibility exports for tests and any internal callers that imported the old
# private helpers from this module before the extraction.
_SampledFrame = SampledFrame
_histogram = histogram
_detector_diagnostics = detector_diagnostics
_diagnostic_count = diagnostic_count
_diagnostic_float_values = diagnostic_float_values
_analysis_diagnostics_payload = analysis_diagnostics_payload
_score_candidate = score_candidate
_candidate_target_score = candidate_target_score
_candidate_rank = candidate_rank
_refresh_candidate_score = refresh_candidate_score
_expand_crop_bbox = expand_crop_bbox
_save_candidate = save_candidate
_render_bbox_for_track = render_bbox_for_track
_bbox_intersection_area = bbox_intersection_area
_bbox_iou = bbox_iou
_bbox_contains_point = bbox_contains_point
_track_matches_edge_detection = track_matches_edge_detection
_track_touches_suppression_edge = track_touches_suppression_edge


def _iter_sampled_frame_batches(
    *,
    video_path: Path,
    source_fps: float,
    target_fps: float,
    profile: CameraProfile,
    batch_size: int,
) -> Iterator[list[SampledFrame]]:
    return iter_sampled_frame_batches(
        video_path=video_path,
        source_fps=source_fps,
        target_fps=target_fps,
        profile=profile,
        batch_size=batch_size,
        iter_sampled_frames_func=iter_sampled_frames,
    )


def _iter_detected_sampled_frames(
    *,
    detector: Detector,
    video_path: Path,
    source_fps: float,
    target_fps: float,
    profile: CameraProfile,
    batch_size: int,
) -> Iterator[tuple[SampledFrame, sv.Detections]]:
    return iter_detected_sampled_frames(
        detector=detector,
        video_path=video_path,
        source_fps=source_fps,
        target_fps=target_fps,
        profile=profile,
        batch_size=batch_size,
        iter_sampled_frames_func=iter_sampled_frames,
    )


def _finalize_analysis(
    *,
    run_store: RunStore,
    track_states: Sequence[MutableTrackState],
    diagnostics: AnalysisDiagnostics,
    detector: Detector,
    config: AppConfig,
) -> None:
    diagnostics.tracks_without_crop_candidates = sum(
        1 for state in track_states if not state.candidates
    )
    diagnostics.tracks_without_crop_due_to_height = sum(
        1
        for state in track_states
        if not state.candidates
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
    for state in track_states:
        run_store.tracks.append(track_summary_from_state(state))
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
    crop_selector = CropCandidateSelector(config, run_store)
    edge_suppression = EdgeSuppression(config, profile)
    track_updater = TrackStateUpdater(
        config=config,
        profile=profile,
        crop_selector=crop_selector,
        edge_suppression=edge_suppression,
        diagnostics=diagnostics,
    )

    for sampled_frame, detections in _iter_detected_sampled_frames(
        detector=detector,
        video_path=video_path,
        source_fps=config.video.fps,
        target_fps=analysis_fps,
        profile=profile,
        batch_size=config.analysis.batch_size,
    ):
        diagnostics.total_sampled_frames += 1
        global_detections = map_detections_to_global(detections, sampled_frame.offset)
        edge_detection_bboxes = edge_suppression.detection_edge_bboxes(
            global_detections,
            frame_shape=sampled_frame.frame.shape,
            roi_shape=sampled_frame.roi_frame.shape,
            roi_offset=sampled_frame.offset,
        )

        diagnostics.detections_passed_to_tracker += len(global_detections)
        tracked = tracker.update(global_detections, sampled_frame.frame)
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
        run_store.frames.append(update_result.frame_record)
        for count_event in update_result.counted_events:
            run_store.counts.append(count_event)

    _finalize_analysis(
        run_store=run_store,
        track_states=track_updater.sorted_track_states(),
        diagnostics=diagnostics,
        detector=detector,
        config=config,
    )

    logger.info("Analysis complete. Run directory: %s", run_store.root)
    return run_store
