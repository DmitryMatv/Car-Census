from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from config import AppConfig
from detectors.base import Detector
from models import BBox, Detection

logger = logging.getLogger(__name__)

GPU_EXECUTION_PROVIDERS = {"CUDAExecutionProvider", "TensorrtExecutionProvider"}


def _is_dynamic_batch_dim(batch_dim: Any) -> bool:
    if batch_dim is None:
        return True
    if isinstance(batch_dim, str):
        return True
    if isinstance(batch_dim, int):
        return batch_dim < 0
    return False


def _fixed_batch_size(batch_dim: Any) -> int | None:
    if _is_dynamic_batch_dim(batch_dim):
        return None
    if isinstance(batch_dim, int) and batch_dim > 0:
        return batch_dim
    return None


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


def _select_execution_providers(
    *,
    requested: list[str],
    available: list[str],
    require_gpu: bool,
) -> list[str]:
    selected = [provider for provider in requested if provider in available]
    selected_gpu = [
        provider for provider in selected if provider in GPU_EXECUTION_PROVIDERS
    ]
    requested_gpu = [
        provider for provider in requested if provider in GPU_EXECUTION_PROVIDERS
    ]

    if require_gpu and not selected_gpu:
        install_hint = (
            "Install `onnxruntime-gpu` in Colab for CUDA support. "
            "TensorRT acceleration also requires ONNX Runtime TensorRT provider "
            "dependencies to be available in the runtime."
        )
        raise RuntimeError(
            "ONNX Runtime GPU execution was requested, but none of the requested "
            f"GPU providers are available. requested={requested_gpu or requested}; "
            f"available={available}. {install_hint}"
        )

    if selected:
        return selected

    if "CPUExecutionProvider" in available:
        logger.warning(
            "None of the requested ONNX Runtime providers are available; "
            "falling back to CPUExecutionProvider. requested=%s available=%s",
            requested,
            available,
        )
        return ["CPUExecutionProvider"]

    raise RuntimeError(
        "No usable ONNX Runtime execution provider is available. "
        f"requested={requested}; available={available}"
    )


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
        requested_providers = list(config.detector.onnx_execution_providers)
        available_providers = list(ort.get_available_providers())
        providers = _select_execution_providers(
            requested=requested_providers,
            available=available_providers,
            require_gpu=config.detector.onnx_require_gpu,
        )
        self.session = ort.InferenceSession(
            str(weights_path),
            sess_options=options,
            providers=providers,
        )
        input_info = self.session.get_inputs()[0]
        self.input_name = input_info.name
        input_shape = list(input_info.shape)
        self.input_batch_dim = input_shape[0] if input_shape else None
        self.dynamic_batch = _is_dynamic_batch_dim(self.input_batch_dim)
        self.fixed_batch_size = _fixed_batch_size(self.input_batch_dim)
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
        active_providers = list(self.session.get_providers())
        logger.info(
            "Device active: %s | provider=onnxruntime | threads=%s | "
            "requested_providers=%s | available_providers=%s | active_providers=%s",
            "gpu"
            if any(provider in GPU_EXECUTION_PROVIDERS for provider in active_providers)
            else "cpu",
            options.intra_op_num_threads,
            requested_providers,
            available_providers,
            active_providers,
        )
        logger.info(
            "ONNX batch support: dynamic=%s input_batch=%s configured_batch_size=%s",
            self.dynamic_batch,
            self.input_batch_dim,
            config.analysis.batch_size,
        )
        if (
            config.analysis.batch_size > 1
            and self.fixed_batch_size == 1
            and not self.dynamic_batch
        ):
            logger.warning(
                "analysis.batch_size=%s requested, but ONNX model input batch is "
                "fixed to 1; falling back to single-frame inference. Re-export ONNX "
                "with dynamic=True to enable batching.",
                config.analysis.batch_size,
            )

    def detect(self, image: np.ndarray) -> list[Detection]:
        tensor, scale, pad_x, pad_y, image_shape = self._preprocess(image)
        outputs = self.session.run(
            None,
            {self.input_name: np.expand_dims(tensor, axis=0)},
        )
        return self._parse_single_output(outputs[0], image_shape, scale, pad_x, pad_y)

    def detect_batch(self, images: Sequence[np.ndarray]) -> list[list[Detection]]:
        if not images:
            return []
        if len(images) == 1 or (self.fixed_batch_size == 1 and not self.dynamic_batch):
            return [self.detect(image) for image in images]

        preprocessed = [self._preprocess(image) for image in images]
        tensors = [item[0] for item in preprocessed]
        requested_count = len(tensors)
        run_count = requested_count
        if self.fixed_batch_size is not None:
            run_count = self.fixed_batch_size
            if requested_count > run_count:
                results: list[list[Detection]] = []
                for start in range(0, requested_count, run_count):
                    results.extend(self.detect_batch(images[start : start + run_count]))
                return results
            if requested_count < run_count:
                pad_tensor = np.zeros_like(tensors[0])
                tensors.extend([pad_tensor] * (run_count - requested_count))

        batch_tensor = np.stack(tensors, axis=0)
        outputs = self.session.run(None, {self.input_name: batch_tensor})
        output = np.asarray(outputs[0])
        if output.ndim < 3 or output.shape[0] < requested_count:
            raise ValueError(f"Unsupported batched ONNX output shape: {output.shape}")

        detections_by_image: list[list[Detection]] = []
        for index, (_tensor, scale, pad_x, pad_y, image_shape) in enumerate(
            preprocessed
        ):
            detections_by_image.append(
                self._parse_single_output(
                    output[index],
                    image_shape,
                    scale,
                    pad_x,
                    pad_y,
                )
            )
        return detections_by_image

    def _preprocess(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, float, float, float, tuple[int, ...]]:
        input_size = self.config.analysis.imgsz
        frame, scale, (pad_x, pad_y) = _letterbox(image, input_size)
        tensor = frame[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return tensor, scale, float(pad_x), float(pad_y), image.shape

    def _parse_single_output(
        self,
        output: Any,
        image_shape: tuple[int, ...],
        scale: float,
        pad_x: float,
        pad_y: float,
    ) -> list[Detection]:
        return self._parse_outputs([output], image_shape, scale, pad_x, pad_y)

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
