from __future__ import annotations

from pathlib import Path
import sys
from typing import Protocol

import numpy as np
import supervision as sv

from config import AppConfig
from models import Detection


class TrackerAdapter(Protocol):
    def update(
        self, detections: list[Detection], frame: np.ndarray
    ) -> sv.Detections: ...


def _resolve_device(config: AppConfig) -> str:
    device = config.tracker.reid_device
    if device != "auto":
        return device
    project_device = config.project.device
    if project_device == "auto":
        return "cpu"
    return project_device


def _resolve_cmc_method(config: AppConfig) -> str | None:
    cmc_method = config.tracker.cmc_method
    if cmc_method is None or cmc_method.lower() == "none":
        return None
    return cmc_method


def _create_botsort_tracker(config: AppConfig, frame_rate: float | None):
    try:
        from boxmot.trackers import BotSort
    except ImportError as exc:
        if sys.version_info >= (3, 13):
            raise RuntimeError(
                "BoT-SORT tracking requires BoxMOT, but BoxMOT currently supports "
                "Python versions below 3.13. Use a Python 3.12 environment and "
                'install with: pip install -e ".[tracking]"'
            ) from exc
        raise RuntimeError(
            'BoT-SORT tracking requires BoxMOT. Install with: pip install -e ".[tracking]"'
        ) from exc

    effective_frame_rate = (
        int(round(frame_rate or 0)) or config.tracker.frame_rate or 30
    )
    kwargs = {
        "reid_weights": Path(config.tracker.reid_weights),
        "device": _resolve_device(config),
        "half": config.tracker.reid_half,
        "with_reid": config.tracker.with_reid,
        "track_high_thresh": config.tracker.track_high_thresh,
        "track_low_thresh": config.tracker.track_low_thresh,
        "new_track_thresh": config.tracker.new_track_thresh,
        "track_buffer": config.tracker.track_buffer,
        "match_thresh": config.tracker.match_thresh,
        "frame_rate": effective_frame_rate,
        "fuse_first_associate": config.tracker.fuse_first_associate,
        "cmc_method": _resolve_cmc_method(config),
        "proximity_thresh": config.tracker.proximity_thresh,
        "appearance_thresh": config.tracker.appearance_thresh,
        "min_hits": config.tracker.minimum_consecutive_frames,
    }
    return BotSort(**kwargs)


class BotSortAdapter:
    def __init__(
        self,
        config: AppConfig,
        frame_rate: float | None = None,
        tracker: object | None = None,
    ) -> None:
        self.tracker = tracker or _create_botsort_tracker(config, frame_rate)

    def update(self, detections: list[Detection], frame: np.ndarray) -> sv.Detections:
        dets = self._to_boxmot_detections(detections)
        tracks = np.asarray(self.tracker.update(dets, frame), dtype=np.float32)
        return self._to_supervision_detections(tracks, detections)

    @staticmethod
    def _to_boxmot_detections(detections: list[Detection]) -> np.ndarray:
        if not detections:
            return np.empty((0, 6), dtype=np.float32)
        return np.array(
            [
                [
                    detection.bbox.x1,
                    detection.bbox.y1,
                    detection.bbox.x2,
                    detection.bbox.y2,
                    detection.confidence,
                    detection.class_id if detection.class_id is not None else -1,
                ]
                for detection in detections
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _to_supervision_detections(
        tracks: np.ndarray, detections: list[Detection]
    ) -> sv.Detections:
        if tracks.size == 0:
            return sv.Detections(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidence=np.empty((0,), dtype=np.float32),
                class_id=np.empty((0,), dtype=np.int32),
                tracker_id=np.empty((0,), dtype=np.int32),
                data={"class_name": np.array([], dtype=object)},
            )

        if tracks.ndim == 1:
            tracks = tracks.reshape(1, -1)
        if tracks.shape[1] < 7:
            raise ValueError(
                f"Expected BoxMOT tracks with at least 7 columns, got {tracks.shape[1]}"
            )

        xyxy = tracks[:, 0:4].astype(np.float32)
        tracker_id = tracks[:, 4].astype(np.int32)
        confidence = tracks[:, 5].astype(np.float32)
        class_id = tracks[:, 6].astype(np.int32)
        class_names = np.array([""] * len(tracks), dtype=object)

        if tracks.shape[1] >= 8:
            det_indices = tracks[:, 7].astype(np.int32)
            for index, det_index in enumerate(det_indices):
                if 0 <= det_index < len(detections):
                    class_names[index] = detections[det_index].class_name or ""

        return sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
            tracker_id=tracker_id,
            data={"class_name": class_names},
        )
