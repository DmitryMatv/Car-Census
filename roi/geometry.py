from __future__ import annotations

import math
from typing import Iterable

import cv2
import numpy as np

from models import BBox


def polygon_array(points: Iterable[Iterable[int]]) -> np.ndarray:
    return np.array(list(points), dtype=np.int32)


def polygon_bounding_rect(points: list[list[int]]) -> tuple[int, int, int, int]:
    contour = polygon_array(points)
    x, y, w, h = cv2.boundingRect(contour)
    return x, y, w, h


def crop_to_polygon(
    frame: np.ndarray, polygon: list[list[int]]
) -> tuple[np.ndarray, tuple[int, int]]:
    x, y, w, h = polygon_bounding_rect(polygon)
    roi = frame[y : y + h, x : x + w].copy()
    local_polygon = np.array([[px - x, py - y] for px, py in polygon], dtype=np.int32)
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [local_polygon], 255)
    roi[mask == 0] = 0
    return roi, (x, y)


def map_bbox_to_global(bbox: BBox, offset: tuple[int, int]) -> BBox:
    ox, oy = offset
    return BBox(x1=bbox.x1 + ox, y1=bbox.y1 + oy, x2=bbox.x2 + ox, y2=bbox.y2 + oy)


def bbox_touches_frame_edge(
    bbox: BBox,
    frame_shape: tuple[int, int] | tuple[int, int, int],
    margin_px: int = 0,
) -> bool:
    height, width = frame_shape[:2]
    return bbox_touches_rect_edge(
        bbox=bbox,
        left=0,
        top=0,
        right=width,
        bottom=height,
        margin_px=margin_px,
    )


def bbox_touches_rect_edge(
    bbox: BBox,
    left: float,
    top: float,
    right: float,
    bottom: float,
    margin_px: int = 0,
) -> bool:
    margin = max(0, margin_px)
    if (
        not all(math.isfinite(value) for value in [bbox.x1, bbox.y1, bbox.x2, bbox.y2])
        or bbox.x2 <= bbox.x1
        or bbox.y2 <= bbox.y1
    ):
        return True
    return (
        math.floor(bbox.x1) <= left + margin
        or math.floor(bbox.y1) <= top + margin
        or math.ceil(bbox.x2) >= right - 1 - margin
        or math.ceil(bbox.y2) >= bottom - 1 - margin
    )


def _point_in_rect(
    point: tuple[float, float],
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> bool:
    x, y = point
    return left <= x <= right and top <= y <= bottom


def _orientation(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]
) -> float:
    return ((b[0] - a[0]) * (c[1] - a[1])) - ((b[1] - a[1]) * (c[0] - a[0]))


def _point_on_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> bool:
    return (
        min(start[0], end[0]) <= point[0] <= max(start[0], end[0])
        and min(start[1], end[1]) <= point[1] <= max(start[1], end[1])
        and _orientation(start, end, point) == 0
    )


def _segments_intersect(
    a_start: tuple[float, float],
    a_end: tuple[float, float],
    b_start: tuple[float, float],
    b_end: tuple[float, float],
) -> bool:
    o1 = _orientation(a_start, a_end, b_start)
    o2 = _orientation(a_start, a_end, b_end)
    o3 = _orientation(b_start, b_end, a_start)
    o4 = _orientation(b_start, b_end, a_end)

    if o1 == 0 and _point_on_segment(b_start, a_start, a_end):
        return True
    if o2 == 0 and _point_on_segment(b_end, a_start, a_end):
        return True
    if o3 == 0 and _point_on_segment(a_start, b_start, b_end):
        return True
    if o4 == 0 and _point_on_segment(a_end, b_start, b_end):
        return True
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


def _segment_intersects_rect(
    start: tuple[float, float],
    end: tuple[float, float],
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> bool:
    if _point_in_rect(start, left, top, right, bottom) or _point_in_rect(
        end, left, top, right, bottom
    ):
        return True

    edges = [
        ((left, top), (right, top)),
        ((right, top), (right, bottom)),
        ((right, bottom), (left, bottom)),
        ((left, bottom), (left, top)),
    ]
    return any(
        _segments_intersect(start, end, edge_start, edge_end)
        for edge_start, edge_end in edges
    )


def bbox_touches_polygon_edge(
    bbox: BBox, polygon: list[list[int]], margin_px: int = 0
) -> bool:
    margin = max(0, margin_px)
    if (
        not all(math.isfinite(value) for value in [bbox.x1, bbox.y1, bbox.x2, bbox.y2])
        or bbox.x2 <= bbox.x1
        or bbox.y2 <= bbox.y1
    ):
        return True
    left = math.floor(bbox.x1) - margin
    top = math.floor(bbox.y1) - margin
    right = math.ceil(bbox.x2) + margin
    bottom = math.ceil(bbox.y2) + margin
    polygon_points = [(float(x), float(y)) for x, y in polygon]
    if len(polygon_points) < 2:
        return False

    for vertex in polygon_points:
        if _point_in_rect(vertex, left, top, right, bottom):
            return True

    for index, start in enumerate(polygon_points):
        end = polygon_points[(index + 1) % len(polygon_points)]
        if _segment_intersects_rect(start, end, left, top, right, bottom):
            return True

    contour = polygon_array(polygon)
    corners = [
        (float(math.floor(bbox.x1)), float(math.floor(bbox.y1))),
        (float(math.ceil(bbox.x2)), float(math.floor(bbox.y1))),
        (float(math.ceil(bbox.x2)), float(math.ceil(bbox.y2))),
        (float(math.floor(bbox.x1)), float(math.ceil(bbox.y2))),
    ]
    corner_distances = [
        cv2.pointPolygonTest(contour, corner, True) for corner in corners
    ]
    return any(distance < 0 for distance in corner_distances) or any(
        abs(distance) <= margin for distance in corner_distances
    )


def point_in_polygon(point: tuple[float, float], polygon: list[list[int]]) -> bool:
    contour = polygon_array(polygon)
    return cv2.pointPolygonTest(contour, point, False) >= 0


def signed_line_side(
    point: tuple[float, float],
    line_start: tuple[float, float],
    line_end: tuple[float, float],
) -> float:
    return ((line_end[0] - line_start[0]) * (point[1] - line_start[1])) - (
        (line_end[1] - line_start[1]) * (point[0] - line_start[0])
    )


def line_crossing_direction(
    previous_point: tuple[float, float],
    current_point: tuple[float, float],
    line_start: tuple[float, float],
    line_end: tuple[float, float],
) -> str | None:
    previous_side = signed_line_side(previous_point, line_start, line_end)
    current_side = signed_line_side(current_point, line_start, line_end)
    if previous_side == 0 or current_side == 0:
        return None
    if previous_side < 0 < current_side:
        return "A_TO_B"
    if previous_side > 0 > current_side:
        return "B_TO_A"
    return None


def clip_bbox_to_frame(bbox: BBox, frame_shape: tuple[int, int, int]) -> BBox | None:
    height, width = frame_shape[:2]
    x1 = min(max(int(bbox.x1), 0), width - 1)
    y1 = min(max(int(bbox.y1), 0), height - 1)
    x2 = min(max(int(bbox.x2), 0), width)
    y2 = min(max(int(bbox.y2), 0), height)
    if x2 <= x1 or y2 <= y1:
        return None
    return BBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))
