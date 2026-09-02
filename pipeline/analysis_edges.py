from __future__ import annotations

from collections.abc import Sequence as SequenceABC

import supervision as sv

from config import AppConfig, CameraProfile
from models import BBox
from pipeline.detections import detection_bboxes
from roi.geometry import (
    bbox_touches_frame_edge,
    bbox_touches_polygon_edge,
    bbox_touches_rect_edge,
)


def track_matches_edge_detection(
    track_bbox: BBox, edge_detection_bboxes: SequenceABC[BBox]
) -> bool:
    """True when the track observation IS an edge detection, not merely near one.

    The IoU bar is high on purpose: far-field vehicles routinely brush past
    low-confidence misfires at the crop boundary (IoU 0.05-0.2) while passing
    static structures, and suppressing those observations makes the real
    vehicle invisible for its whole far-field approach. Only a substantial
    duplicate counts.
    """
    for detection_bbox in edge_detection_bboxes:
        if track_bbox.iou(detection_bbox) >= 0.5:
            return True
    return False


def track_touches_suppression_edge(
    *,
    bbox: BBox,
    frame_shape: tuple[int, int, int],
    roi_shape: tuple[int, int, int],
    roi_offset: tuple[int, int],
    profile: CameraProfile,
    config: AppConfig,
) -> bool:
    roi_height, roi_width = roi_shape[:2]
    roi_left, roi_top = roi_offset
    margin = config.tracker.edge_margin_px
    return (
        bbox_touches_frame_edge(bbox, frame_shape, margin)
        or bbox_touches_rect_edge(
            bbox=bbox,
            left=float(roi_left),
            top=float(roi_top),
            right=float(roi_left + roi_width),
            bottom=float(roi_top + roi_height),
            margin_px=margin,
        )
        or bbox_touches_polygon_edge(bbox, profile.polygon.points, margin)
    )


class EdgeSuppression:
    def __init__(self, config: AppConfig, profile: CameraProfile) -> None:
        self.config = config
        self.profile = profile

    def detection_edge_bboxes(
        self,
        detections: sv.Detections,
        *,
        frame_shape: tuple[int, int, int],
        roi_shape: tuple[int, int, int],
        roi_offset: tuple[int, int],
    ) -> list[BBox]:
        return [
            bbox
            for bbox in detection_bboxes(detections)
            if self._hits_suppression_edge(
                bbox,
                frame_shape=frame_shape,
                roi_shape=roi_shape,
                roi_offset=roi_offset,
            )
        ]

    def track_touches_suppression_edge(
        self,
        *,
        bbox: BBox,
        frame_shape: tuple[int, int, int],
        roi_shape: tuple[int, int, int],
        roi_offset: tuple[int, int],
    ) -> bool:
        return track_touches_suppression_edge(
            bbox=bbox,
            frame_shape=frame_shape,
            roi_shape=roi_shape,
            roi_offset=roi_offset,
            profile=self.profile,
            config=self.config,
        )

    def track_matches_edge_detection(
        self,
        track_bbox: BBox,
        edge_detection_bboxes: SequenceABC[BBox],
    ) -> bool:
        return track_matches_edge_detection(track_bbox, edge_detection_bboxes)

    def should_skip_track_observation(
        self,
        *,
        bbox: BBox,
        edge_detection_bboxes: SequenceABC[BBox],
        frame_shape: tuple[int, int, int],
        roi_shape: tuple[int, int, int],
        roi_offset: tuple[int, int],
    ) -> bool:
        if not self.config.tracker.ignore_edge_touches:
            return False
        return self._hits_suppression_edge(
            bbox,
            frame_shape=frame_shape,
            roi_shape=roi_shape,
            roi_offset=roi_offset,
        ) or self.track_matches_edge_detection(bbox, edge_detection_bboxes)

    def _hits_suppression_edge(
        self,
        bbox: BBox,
        *,
        frame_shape: tuple[int, int, int],
        roi_shape: tuple[int, int, int],
        roi_offset: tuple[int, int],
    ) -> bool:
        if not self.config.tracker.ignore_edge_touches:
            return False
        return self.track_touches_suppression_edge(
            bbox=bbox,
            frame_shape=frame_shape,
            roi_shape=roi_shape,
            roi_offset=roi_offset,
        )
