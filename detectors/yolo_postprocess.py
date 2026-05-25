from __future__ import annotations

import ast
import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from config import AppConfig
from models import BBox, Detection

logger = logging.getLogger(__name__)


def _metadata_names(raw: str | None) -> dict[int, str]:
    if not raw:
        return {}
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return {}
    if isinstance(value, dict):
        return {int(key): str(name).lower() for key, name in value.items()}
    if isinstance(value, list):
        return {index: str(name).lower() for index, name in enumerate(value)}
    return {}


def _allowed_class_names(
    configured_names: Sequence[str], class_names: dict[int, str]
) -> set[str]:
    configured_allowed_names = {name.lower() for name in configured_names}
    model_class_names = {name.lower() for name in class_names.values()}
    discovered_vehicle_names = {"van", "pickup", "crossover"} & set(model_class_names)
    return configured_allowed_names | discovered_vehicle_names


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    output = boxes.copy()
    output[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    output[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    output[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    output[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return output


def _nms_indices(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[current], x1[rest])
        yy1 = np.maximum(y1[current], y1[rest])
        xx2 = np.minimum(x2[current], x2[rest])
        yy2 = np.minimum(y2[current], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[current] + areas[rest] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[iou <= threshold]
    return keep


@dataclass(frozen=True, slots=True)
class YoloDetectionPolicy:
    confidence: float
    iou: float
    class_names: dict[int, str]
    allowed_names: set[str]
    allowed_ids: list[int] | None


@dataclass(slots=True)
class YoloDiagnostics:
    counts: Counter[str] = field(default_factory=Counter)
    confidence_values: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "counts": dict(self.counts),
            "confidence_values": list(self.confidence_values),
        }


class YoloOutputParser:
    def __init__(self, policy: YoloDetectionPolicy) -> None:
        self.policy = policy
        self.diagnostics = YoloDiagnostics()

    def parse_single(
        self,
        output: Any,
        image_shape: tuple[int, ...],
        scale: float,
        pad_x: float,
        pad_y: float,
    ) -> list[Detection]:
        return self.parse_outputs([output], image_shape, scale, pad_x, pad_y)

    def parse_outputs(
        self,
        outputs: list[Any],
        image_shape: tuple[int, ...],
        scale: float,
        pad_x: float,
        pad_y: float,
    ) -> list[Detection]:
        raw = np.asarray(outputs[0], dtype=np.float32)
        if raw.ndim == 3:
            raw = raw[0]
        if raw.ndim != 2:
            raise ValueError(f"Unsupported ONNX output shape: {raw.shape}")
        if raw.shape[0] in {6, 84, 85} and raw.shape[1] != raw.shape[0]:
            raw = raw.T

        rows = self._candidate_rows(raw)
        if len(rows) == 0:
            return []

        boxes = rows[:, :4]
        if np.any(boxes[:, 2] <= boxes[:, 0]) or np.any(boxes[:, 3] <= boxes[:, 1]):
            boxes = _xywh_to_xyxy(boxes)
        scores = rows[:, 4]
        class_ids = rows[:, 5].astype(int)
        keep: list[int] = []
        for class_id in np.unique(class_ids):
            class_indices = np.where(class_ids == class_id)[0]
            keep.extend(
                class_indices[
                    _nms_indices(
                        boxes[class_indices],
                        scores[class_indices],
                        self.policy.iou,
                    )
                ].tolist()
            )

        height, width = image_shape[:2]
        detections: list[Detection] = []
        for index in sorted(keep, key=lambda item: scores[item], reverse=True):
            class_id = int(class_ids[index])
            class_name = self.policy.class_names.get(class_id)
            if (
                self.policy.allowed_ids is not None
                and class_id not in self.policy.allowed_ids
            ):
                continue
            if (
                self.policy.allowed_names
                and class_name not in self.policy.allowed_names
            ):
                continue
            self.diagnostics.counts["detections_after_class_filtering"] += 1
            self.diagnostics.confidence_values.append(float(scores[index]))
            box = boxes[index].copy()
            box[[0, 2]] = (box[[0, 2]] - pad_x) / scale
            box[[1, 3]] = (box[[1, 3]] - pad_y) / scale
            box[[0, 2]] = np.clip(box[[0, 2]], 0, width - 1)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, height - 1)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            detections.append(
                Detection(
                    bbox=BBox(
                        x1=float(box[0]),
                        y1=float(box[1]),
                        x2=float(box[2]),
                        y2=float(box[3]),
                    ),
                    confidence=float(scores[index]),
                    class_id=class_id,
                    class_name=class_name,
                )
            )
        return detections

    def _candidate_rows(self, raw: np.ndarray) -> np.ndarray:
        self.diagnostics.counts["raw_candidate_rows"] += int(raw.shape[0])
        if raw.shape[1] == 6:
            rows = raw[raw[:, 4] >= self.policy.confidence].copy()
            self.diagnostics.counts["detections_after_confidence_filtering"] += len(
                rows
            )
            return rows[:, [0, 1, 2, 3, 4, 5]]

        if raw.shape[1] < 5:
            raise ValueError(f"Unsupported ONNX output shape: {raw.shape}")

        class_count = len(self.policy.class_names)
        if class_count and raw.shape[1] == class_count + 5:
            class_scores = raw[:, 5:] * raw[:, 4:5]
        else:
            class_scores = raw[:, 4:]
        class_ids = class_scores.argmax(axis=1)
        scores = class_scores[np.arange(len(raw)), class_ids]
        keep = scores >= self.policy.confidence
        self.diagnostics.counts["detections_after_confidence_filtering"] += int(
            np.count_nonzero(keep)
        )
        return np.column_stack((raw[keep, :4], scores[keep], class_ids[keep]))


def build_yolo_output_parser(
    config: AppConfig, metadata: Mapping[str, str]
) -> YoloOutputParser:
    class_names = {
        class_id: name.lower() for class_id, name in config.detector.class_names.items()
    }
    class_names.update(_metadata_names(metadata.get("names")))
    discovered_vehicle_names = {"van", "pickup", "crossover"} & set(
        class_names.values()
    )
    logger.info("ONNX model class names: %s", class_names)
    if discovered_vehicle_names:
        logger.info(
            "Added discovered vehicle classes to detector allow-list: %s",
            sorted(discovered_vehicle_names),
        )
    return YoloOutputParser(
        YoloDetectionPolicy(
            confidence=config.detector.confidence,
            iou=config.detector.iou,
            class_names=class_names,
            allowed_names=_allowed_class_names(
                config.detector.allowed_class_names, class_names
            ),
            allowed_ids=config.detector.allowed_class_ids,
        )
    )
