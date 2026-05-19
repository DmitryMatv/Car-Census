from __future__ import annotations

from typing import Protocol

import numpy as np
import supervision as sv
from trackers import BoTSORTTracker

from config import AppConfig
from models import Detection


class TrackerAdapter(Protocol):
    def update(
        self, detections: list[Detection], frame: np.ndarray
    ) -> sv.Detections: ...


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

    def update(self, detections: list[Detection], frame: np.ndarray) -> sv.Detections:
        sv_detections = self._to_supervision_detections(detections)
        return self.tracker.update(sv_detections, frame=frame)

    @staticmethod
    def _to_supervision_detections(detections: list[Detection]) -> sv.Detections:
        if not detections:
            return sv.Detections(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidence=np.empty((0,), dtype=np.float32),
                class_id=np.empty((0,), dtype=np.int32),
                data={"class_name": np.array([], dtype=object)},
            )

        return sv.Detections(
            xyxy=np.array(
                [
                    [
                        detection.bbox.x1,
                        detection.bbox.y1,
                        detection.bbox.x2,
                        detection.bbox.y2,
                    ]
                    for detection in detections
                ],
                dtype=np.float32,
            ),
            confidence=np.array(
                [detection.confidence for detection in detections], dtype=np.float32
            ),
            class_id=np.array(
                [
                    detection.class_id if detection.class_id is not None else -1
                    for detection in detections
                ],
                dtype=np.int32,
            ),
            data={
                "class_name": np.array(
                    [detection.class_name or "" for detection in detections],
                    dtype=object,
                )
            },
        )
