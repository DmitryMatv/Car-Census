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
    return (
        math.floor(bbox.x1) <= left + margin
        or math.floor(bbox.y1) <= top + margin
        or math.ceil(bbox.x2) >= right - 1 - margin
        or math.ceil(bbox.y2) >= bottom - 1 - margin
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
