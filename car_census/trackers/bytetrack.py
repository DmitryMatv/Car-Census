from __future__ import annotations

from collections import defaultdict, deque
from typing import Protocol

import numpy as np
import supervision as sv

from car_census.config import AppConfig
from car_census.types import Detection


class TrackerAdapter(Protocol):
    def update(self, detections: list[Detection]) -> sv.Detections: ...


class ByteTrackAdapter:
    def __init__(self, config: AppConfig, frame_rate: float | None = None) -> None:
        effective_frame_rate = (
            int(round(frame_rate or 0)) or config.tracker.frame_rate or 30
        )
        self.tracker = sv.ByteTrack(
            track_activation_threshold=config.tracker.track_activation_threshold,
            lost_track_buffer=config.tracker.lost_track_buffer,
            minimum_matching_threshold=config.tracker.minimum_matching_threshold,
            minimum_consecutive_frames=config.tracker.minimum_consecutive_frames,
            frame_rate=effective_frame_rate,
        )
        self.smoothing_alpha = getattr(config.tracker, "smoothing_alpha", 0.8)
        self.smoothing_history = max(
            1, getattr(config.tracker, "smoothing_history", 10)
        )
        self._track_history: defaultdict[int, deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=self.smoothing_history)
        )

    def _smooth_box(self, box: np.ndarray, track_id: int) -> np.ndarray:
        if self.smoothing_history <= 1 or self.smoothing_alpha <= 0:
            return box
        history = self._track_history[track_id]
        history.append(box.copy())
        if len(history) == 1:
            return box
        weights = np.array(
            [self.smoothing_alpha**i for i in range(len(history) - 1, -1, -1)]
        )
        weights /= weights.sum()
        return np.average(np.stack(history), axis=0, weights=weights)

    def update(self, detections: list[Detection]) -> sv.Detections:
        if not detections:
            empty = sv.Detections(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidence=np.empty((0,), dtype=np.float32),
                class_id=np.empty((0,), dtype=np.int32),
            )
            self._track_history.clear()
            return self.tracker.update_with_detections(empty)

        xyxy = np.array(
            [[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2] for d in detections],
            dtype=np.float32,
        )
        confidence = np.array([d.confidence for d in detections], dtype=np.float32)
        class_id = np.array(
            [d.class_id if d.class_id is not None else -1 for d in detections],
            dtype=np.int32,
        )
        payload = {
            "class_name": np.array(
                [d.class_name or "" for d in detections], dtype=object
            )
        }
        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
            data=payload,
        )
        tracked = self.tracker.update_with_detections(sv_detections)

        if tracked.tracker_id is not None:
            active_track_ids = set()
            for i in range(len(tracked)):
                track_id = int(tracked.tracker_id[i])
                active_track_ids.add(track_id)
                original_box = tracked.xyxy[i]
                smoothed_box = self._smooth_box(original_box, track_id)
                tracked.xyxy[i] = smoothed_box
            stale_track_ids = set(self._track_history) - active_track_ids
            for track_id in stale_track_ids:
                self._track_history.pop(track_id, None)
        else:
            self._track_history.clear()

        return tracked
