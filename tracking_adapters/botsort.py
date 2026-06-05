from __future__ import annotations

from collections.abc import Collection
from typing import Protocol

import numpy as np
import supervision as sv
from trackers import BoTSORTTracker

from config import AppConfig


class _BoTSortLike(Protocol):
    def update(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections: ...


def _effective_frame_rate(config: AppConfig, frame_rate: float | None) -> float:
    return float(config.tracker.frame_rate or frame_rate or 30.0)


def _create_botsort_tracker(
    config: AppConfig, frame_rate: float | None
) -> BoTSORTTracker:
    tracker_config = config.tracker
    return BoTSORTTracker(
        lost_track_buffer=tracker_config.lost_track_buffer,
        frame_rate=_effective_frame_rate(config, frame_rate),
        track_activation_threshold=tracker_config.track_activation_threshold,
        minimum_consecutive_frames=tracker_config.minimum_consecutive_frames,
        minimum_iou_threshold_first_assoc=tracker_config.minimum_iou_threshold_first_assoc,
        minimum_iou_threshold_second_assoc=tracker_config.minimum_iou_threshold_second_assoc,
        minimum_iou_threshold_unconfirmed_assoc=tracker_config.minimum_iou_threshold_unconfirmed_assoc,
        high_conf_det_threshold=tracker_config.high_conf_det_threshold,
        enable_cmc=tracker_config.enable_cmc,
        cmc_method=tracker_config.cmc_method,
        cmc_downscale=tracker_config.cmc_downscale,
        instant_first_frame_activation=tracker_config.instant_first_frame_activation,
    )


class BotSortAdapter:
    def __init__(
        self,
        config: AppConfig,
        frame_rate: float | None = None,
        tracker: _BoTSortLike | None = None,
    ) -> None:
        self.tracker: _BoTSortLike = tracker or _create_botsort_tracker(
            config, frame_rate
        )

    def update(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
        return self.tracker.update(detections, frame=frame)

    def drop_tracks(self, track_ids: Collection[int]) -> None:
        if not track_ids:
            return
        tracks = getattr(self.tracker, "tracks", None)
        if not isinstance(tracks, list):
            return
        suppressed_ids = set(track_ids)
        filtered_tracks = [
            track for track in tracks if _track_identity(track) not in suppressed_ids
        ]
        setattr(self.tracker, "tracks", filtered_tracks)


def _track_identity(track: object) -> int | None:
    tracker_id = getattr(track, "tracker_id", None)
    if isinstance(tracker_id, int):
        return tracker_id
    track_id = getattr(track, "track_id", None)
    return track_id if isinstance(track_id, int) else None
