from __future__ import annotations

import logging
from collections.abc import Collection
from typing import Any, Protocol

import numpy as np
import supervision as sv
from config import AppConfig
from reid import AppearanceEmbedder, TrackAppearanceMemory, build_embedder
from roi.transform import ViewTransformer
from trackers import BoTSORTTracker

from tracking_adapters.rescue import RescueEngine

logger = logging.getLogger(__name__)


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


def _xyxy_tuple(bbox: np.ndarray | Any) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(value) for value in bbox)
    return (x1, y1, x2, y2)


def _is_fresh_spawn(track: object) -> bool:
    """True for a tracklet spawned from an unmatched detection this frame.

    BoT-SORT counts the initial bbox as the first successful update, so a
    just-spawned tracklet has exactly one successful update and has not been
    predicted yet. Matured unconfirmed tracklets have >= 2 updates.
    """
    successful = getattr(track, "number_of_successful_updates", None)
    since_update = getattr(track, "time_since_update", None)
    state_bbox = getattr(track, "get_state_bbox", None)
    return successful == 1 and since_update == 0 and callable(state_bbox)


def _crop_with_padding(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    padding_ratio: float,
    padding_px: int,
) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    pad_x = (x2 - x1) * padding_ratio + padding_px
    pad_y = (y2 - y1) * padding_ratio + padding_px
    left = max(0, int(x1 - pad_x))
    top = max(0, int(y1 - pad_y))
    right = min(width, int(x2 + pad_x))
    bottom = min(height, int(y2 + pad_y))
    if right <= left or bottom <= top:
        return None
    return frame[top:bottom, left:right]


class BotSortAdapter:
    def __init__(
        self,
        config: AppConfig,
        frame_rate: float | None = None,
        tracker: _BoTSortLike | None = None,
        view_transformer: ViewTransformer | None = None,
        embedder: AppearanceEmbedder | None = None,
    ) -> None:
        self.tracker: Any = tracker or _create_botsort_tracker(config, frame_rate)
        self._config = config
        self._embedder: AppearanceEmbedder | None = (
            embedder if embedder is not None else build_embedder(config.reid)
        )
        self._memory = TrackAppearanceMemory(config.reid.history_length)
        self._rescue = RescueEngine(config.rescue, view_transformer)
        self._embed_clock: dict[int, int] = {}
        self._events: list[dict[str, Any]] = []

    def update(
        self,
        detections: sv.Detections,
        frame: np.ndarray,
        timestamp: float | None = None,
    ) -> sv.Detections:
        if timestamp is None:
            tracked = self.tracker.update(detections, frame=frame)
        else:
            tracked = self.tracker.update(detections, frame=frame, timestamp=timestamp)
        if timestamp is not None:
            self._observe_world(tracked, timestamp)
        self._observe_appearance(tracked, frame)
        if timestamp is not None:
            self._attempt_rescues(tracked, frame, timestamp)
        return tracked

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
        for track_id in suppressed_ids:
            self._memory.forget(track_id)
            self._rescue.forget(track_id)

    def rescue_audit_payload(self) -> dict[str, Any]:
        return {
            "enabled": self._config.rescue.enabled,
            "world_gate_active": self._rescue.active,
            "reid_active": self._embedder is not None,
            "events": self._events,
        }

    def _observe_world(self, tracked: sv.Detections, timestamp: float) -> None:
        if tracked.tracker_id is None:
            return
        for bbox, track_id in zip(tracked.xyxy, tracked.tracker_id):
            if track_id is None or int(track_id) < 0:
                continue
            x1, _y1, x2, y2 = (float(value) for value in bbox)
            self._rescue.observe(int(track_id), timestamp, ((x1 + x2) / 2.0, y2))

    def _observe_appearance(self, tracked: sv.Detections, frame: np.ndarray) -> None:
        if self._embedder is None or tracked.tracker_id is None:
            return
        pending: list[tuple[int, np.ndarray]] = []
        for bbox, track_id in zip(tracked.xyxy, tracked.tracker_id):
            if track_id is None or int(track_id) < 0:
                continue
            track_id = int(track_id)
            clock = self._embed_clock.get(track_id, 0) + 1
            self._embed_clock[track_id] = clock
            if (clock - 1) % self._config.reid.embed_every_n_frames != 0:
                continue
            crop = _crop_with_padding(
                frame,
                _xyxy_tuple(bbox),
                self._config.analysis.crop_padding_ratio,
                self._config.analysis.crop_padding_px,
            )
            if crop is not None:
                pending.append((track_id, crop))
        if not pending:
            return
        try:
            vectors = self._embedder.embed([crop for _id, crop in pending])
        except Exception:
            logger.warning(
                "ReID embedder failed; appearance evidence disabled", exc_info=True
            )
            self._embedder = None
            return
        for (track_id, _crop), vector in zip(pending, vectors):
            self._memory.observe(track_id, vector)

    def _attempt_rescues(
        self, tracked: sv.Detections, frame: np.ndarray, timestamp: float
    ) -> None:
        if not self._rescue.active:
            return
        tracks = getattr(self.tracker, "tracks", None)
        if not isinstance(tracks, list):
            return
        spawns = [track for track in tracks if _is_fresh_spawn(track)]
        if not spawns:
            return
        output_ids: Any = tracked.tracker_id if tracked.tracker_id is not None else []
        busy_ids = {
            int(track_id)
            for track_id in output_ids
            if track_id is not None and int(track_id) >= 0
        }
        spawn_boxes: list[tuple[float, float, float, float]] = [
            _xyxy_tuple(track.get_state_bbox()) for track in spawns
        ]
        vectors: list[np.ndarray | None] = [None] * len(spawns)
        if self._embedder is not None:
            crops = [
                _crop_with_padding(
                    frame,
                    bbox,
                    self._config.analysis.crop_padding_ratio,
                    self._config.analysis.crop_padding_px,
                )
                for bbox in spawn_boxes
            ]
            indexed = [
                (index, crop) for index, crop in enumerate(crops) if crop is not None
            ]
            if indexed:
                try:
                    embedded = self._embedder.embed([crop for _index, crop in indexed])
                except Exception:
                    logger.warning(
                        "ReID embedder failed; appearance evidence disabled",
                        exc_info=True,
                    )
                    self._embedder = None
                    embedded = np.empty((0, 0), dtype=np.float32)
                for (index, _crop), vector in zip(indexed, embedded):
                    vectors[index] = vector

        for spawn, bbox, vector in zip(spawns, spawn_boxes, vectors):
            match, rejections = self._rescue.match(
                bbox,
                timestamp,
                busy_ids=busy_ids,
                memory=self._memory,
                candidate_vector=vector,
                min_appearance_similarity=self._config.reid.min_appearance_similarity,
            )
            for rejection in rejections:
                self._events.append(
                    self._event("rejected", timestamp, bbox, rejection.__dict__)
                )
            if match is None:
                continue
            old_id = match.old_track_id
            self.tracker.tracks = [
                track
                for track in self.tracker.tracks
                if _track_identity(track) != old_id
            ]
            spawn.tracker_id = old_id
            busy_ids.add(old_id)
            self._rescue.observe(
                old_id, timestamp, ((bbox[0] + bbox[2]) / 2.0, bbox[3])
            )
            self._events.append(
                self._event(
                    "accepted",
                    timestamp,
                    bbox,
                    {
                        "old_track_id": old_id,
                        "gap_seconds": match.gap_seconds,
                        "distance_m": match.distance_m,
                        "lateral_m": match.lateral_m,
                        "implied_speed_mps": match.implied_speed_mps,
                        "appearance_similarity": match.appearance_similarity,
                    },
                )
            )

    def _event(
        self,
        outcome: str,
        timestamp: float,
        bbox: tuple[float, float, float, float],
        details: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "outcome": outcome,
            "timestamp_seconds": timestamp,
            "bbox": [float(value) for value in bbox],
            **details,
        }


def _track_identity(track: object) -> int | None:
    tracker_id = getattr(track, "tracker_id", None)
    if isinstance(tracker_id, int):
        return tracker_id
    track_id = getattr(track, "track_id", None)
    return track_id if isinstance(track_id, int) else None
