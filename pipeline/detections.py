from __future__ import annotations

from copy import deepcopy
from typing import cast

import numpy as np
import supervision as sv

from models import BBox


def empty_detections() -> sv.Detections:
    return sv.Detections(
        xyxy=np.empty((0, 4), dtype=np.float32),
        confidence=np.empty((0,), dtype=np.float32),
        class_id=np.empty((0,), dtype=np.int32),
        data={"class_name": np.empty((0,), dtype=object)},
    )


def detection_bboxes(detections: sv.Detections) -> list[BBox]:
    return [
        BBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))
        for x1, y1, x2, y2 in detections.xyxy.tolist()
    ]


def clone_detections(detections: sv.Detections) -> sv.Detections:
    mask = (
        detections.mask.copy()
        if isinstance(detections.mask, np.ndarray)
        else detections.mask
    )
    return sv.Detections(
        xyxy=np.asarray(detections.xyxy, dtype=np.float32).copy(),
        mask=mask,
        confidence=np.asarray(detections.confidence, dtype=np.float32).copy()
        if detections.confidence is not None
        else None,
        class_id=np.asarray(detections.class_id, dtype=np.int32).copy()
        if detections.class_id is not None
        else None,
        tracker_id=np.asarray(detections.tracker_id, dtype=np.int32).copy()
        if detections.tracker_id is not None
        else None,
        data={
            key: value.copy() if isinstance(value, np.ndarray) else deepcopy(value)
            for key, value in detections.data.items()
        },
        metadata=dict(detections.metadata),
    )


def map_detections_to_global(
    detections: sv.Detections, offset: tuple[int, int]
) -> sv.Detections:
    if len(detections) == 0:
        return clone_detections(detections)
    ox, oy = offset
    mapped = clone_detections(detections)
    xyxy = np.asarray(mapped.xyxy, dtype=np.float32)
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]] + float(ox)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]] + float(oy)
    mapped.xyxy = xyxy
    return mapped


def clip_detections_to_shape(
    detections: sv.Detections, image_shape: tuple[int, ...]
) -> sv.Detections:
    if len(detections) == 0:
        return clone_detections(detections)
    height, width = image_shape[:2]
    clipped = clone_detections(detections)
    clipped.xyxy[:, [0, 2]] = np.clip(clipped.xyxy[:, [0, 2]], 0, width - 1)
    clipped.xyxy[:, [1, 3]] = np.clip(clipped.xyxy[:, [1, 3]], 0, height - 1)
    xyxy = np.asarray(clipped.xyxy, dtype=np.float32)
    valid = (xyxy[:, 2] > xyxy[:, 0]) & (xyxy[:, 3] > xyxy[:, 1])
    return cast(sv.Detections, clipped[valid])


def class_names(detections: sv.Detections) -> list[str]:
    raw_names = detections.data.get("class_name") if detections.data else None
    names: list[str] = []
    for index in range(len(detections)):
        if raw_names is not None and index < len(raw_names):
            names.append(str(raw_names[index]).lower())
        else:
            names.append("")
    return names
