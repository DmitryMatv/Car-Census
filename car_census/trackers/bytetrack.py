from __future__ import annotations

import numpy as np
import supervision as sv

from car_census.config import AppConfig
from car_census.types import Detection


class ByteTrackAdapter:
    def __init__(self, config: AppConfig, frame_rate: float | None = None) -> None:
        effective_frame_rate = int(round(frame_rate or 0)) or config.tracker.frame_rate or 30
        self.tracker = sv.ByteTrack(
            track_activation_threshold=config.tracker.track_activation_threshold,
            lost_track_buffer=config.tracker.lost_track_buffer,
            minimum_matching_threshold=config.tracker.minimum_matching_threshold,
            minimum_consecutive_frames=config.tracker.minimum_consecutive_frames,
            frame_rate=effective_frame_rate,
        )

    def update(self, detections: list[Detection]) -> sv.Detections:
        if not detections:
            empty = sv.Detections(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidence=np.empty((0,), dtype=np.float32),
                class_id=np.empty((0,), dtype=np.int32),
            )
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
        payload = {"class_name": np.array([d.class_name or "" for d in detections], dtype=object)}
        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
            data=payload,
        )
        return self.tracker.update_with_detections(sv_detections)
