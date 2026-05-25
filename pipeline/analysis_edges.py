from __future__ import annotations

from collections.abc import Sequence as SequenceABC

from config import AppConfig, CameraProfile
from models import BBox, Detection
from roi.geometry import (
    bbox_touches_frame_edge,
    bbox_touches_polygon_edge,
    bbox_touches_rect_edge,
)


def bbox_intersection_area(left: BBox, right: BBox) -> float:
    intersection_width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    intersection_height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    return intersection_width * intersection_height


def bbox_iou(left: BBox, right: BBox) -> float:
    intersection_area = bbox_intersection_area(left, right)
    if intersection_area <= 0:
        return 0.0
    union_area = left.area + right.area - intersection_area
    if union_area <= 0:
        return 0.0
    return intersection_area / union_area


def bbox_contains_point(bbox: BBox, point: tuple[float, float]) -> bool:
    x, y = point
    return bbox.x1 <= x <= bbox.x2 and bbox.y1 <= y <= bbox.y2


def track_matches_edge_detection(
    track_bbox: BBox, edge_detection_bboxes: SequenceABC[BBox]
) -> bool:
    for detection_bbox in edge_detection_bboxes:
        if bbox_iou(track_bbox, detection_bbox) >= 0.05:
            return True
        if bbox_contains_point(track_bbox, detection_bbox.center):
            return True
        if bbox_contains_point(detection_bbox, track_bbox.center):
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
        detections: SequenceABC[Detection],
        *,
        frame_shape: tuple[int, int, int],
        roi_shape: tuple[int, int, int],
        roi_offset: tuple[int, int],
    ) -> list[BBox]:
        if not self.config.tracker.ignore_edge_touches:
            return []
        return [
            detection.bbox
            for detection in detections
            if self.track_touches_suppression_edge(
                bbox=detection.bbox,
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
        return self.track_touches_suppression_edge(
            bbox=bbox,
            frame_shape=frame_shape,
            roi_shape=roi_shape,
            roi_offset=roi_offset,
        ) or self.track_matches_edge_detection(bbox, edge_detection_bboxes)
