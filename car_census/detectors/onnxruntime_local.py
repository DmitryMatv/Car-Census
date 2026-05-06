from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from car_census.config import AppConfig
from car_census.detectors.base import Detector
from car_census.types import BBox, Detection

logger = logging.getLogger(__name__)


def _letterbox(
    image: np.ndarray, size: int
) -> tuple[np.ndarray, float, tuple[float, float]]:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    pad_x = (size - new_width) / 2
    pad_y = (size - new_height) / 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top = int(round(pad_y - 0.1))
    left = int(round(pad_x - 0.1))
    canvas[top : top + new_height, left : left + new_width] = resized
    return canvas, scale, (left, top)


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


class OnnxRuntimeLocalDetector(Detector):
    def __init__(self, config: AppConfig, project_root: Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required for detector.provider=onnxruntime_local. "
                "Install the project with `pip install -e .`."
            ) from exc

        self.config = config
        weights_path = Path(config.detector.weights)
        if not weights_path.is_absolute():
            weights_path = project_root / weights_path
        if not weights_path.exists():
            raise FileNotFoundError(f"ONNX detector weights not found: {weights_path}")

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, config.detector.onnx_threads)
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(weights_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        metadata = self.session.get_modelmeta().custom_metadata_map
        self.class_names = {
            int(class_id): name.lower()
            for class_id, name in config.detector.class_names.items()
        }
        self.class_names.update(_metadata_names(metadata.get("names")))
        self.allowed_names = {
            name.lower() for name in config.detector.allowed_class_names
        }
        self.allowed_ids = config.detector.allowed_class_ids
        logger.info(
            "Device active: cpu | provider=onnxruntime | threads=%s",
            options.intra_op_num_threads,
        )

    def detect(self, image: np.ndarray) -> list[Detection]:
        input_size = self.config.analysis.imgsz
        frame, scale, (pad_x, pad_y) = _letterbox(image, input_size)
        tensor = frame[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        tensor = np.expand_dims(tensor, axis=0)
        outputs = self.session.run(None, {self.input_name: tensor})
        return self._parse_outputs(outputs, image.shape, scale, pad_x, pad_y)

    def _parse_outputs(
        self,
        outputs: list[Any],
        image_shape: tuple[int, ...],
        scale: float,
        pad_x: float,
        pad_y: float,
    ) -> list[Detection]:
        raw = np.asarray(outputs[0])
        if raw.ndim == 3:
            raw = raw[0]
        if raw.ndim != 2:
            raise ValueError(f"Unsupported ONNX output shape: {raw.shape}")
        if raw.shape[0] in {6, 84, 85} and raw.shape[1] > raw.shape[0]:
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
                        self.config.detector.iou,
                    )
                ].tolist()
            )

        height, width = image_shape[:2]
        detections: list[Detection] = []
        for index in sorted(keep, key=lambda item: scores[item], reverse=True):
            class_id = int(class_ids[index])
            class_name = self.class_names.get(class_id)
            if self.allowed_ids is not None and class_id not in self.allowed_ids:
                continue
            if self.allowed_names and class_name not in self.allowed_names:
                continue
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
        confidence = self.config.detector.confidence
        if raw.shape[1] == 6:
            rows = raw[raw[:, 4] >= confidence].copy()
            return rows[:, [0, 1, 2, 3, 4, 5]]

        if raw.shape[1] < 5:
            raise ValueError(f"Unsupported ONNX output shape: {raw.shape}")

        class_count = len(self.class_names)
        if class_count and raw.shape[1] == class_count + 5:
            class_scores = raw[:, 5:] * raw[:, 4:5]
        else:
            class_scores = raw[:, 4:]
        class_ids = class_scores.argmax(axis=1)
        scores = class_scores[np.arange(len(raw)), class_ids]
        keep = scores >= confidence
        return np.column_stack((raw[keep, :4], scores[keep], class_ids[keep]))
