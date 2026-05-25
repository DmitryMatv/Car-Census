from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import supervision as sv

from config import AppConfig
from detectors.base import Detector
from models import BBox, Detection


def _load_rfdetr_small() -> type[Any]:
    try:
        from rfdetr import RFDETRSmall
    except ImportError as exc:
        raise RuntimeError(
            "rfdetr is required for RF-DETR-S detection. "
            "Install the project with `pip install -e .`."
        ) from exc
    return RFDETRSmall


def _coerce_class_names(raw: object) -> dict[int, str]:
    if isinstance(raw, Mapping):
        return {int(class_id): str(name).lower() for class_id, name in raw.items()}
    if isinstance(raw, Sequence) and not isinstance(raw, str):
        return {index: str(name).lower() for index, name in enumerate(raw)}
    return {}


class RfDetrSmallDetector(Detector):
    def __init__(self, config: AppConfig, project_root: Path) -> None:
        self.config = config
        self.input_size = config.detector.input_size
        self.allowed_class_names = {
            name.lower() for name in config.detector.allowed_class_names
        }
        self._counts: Counter[str] = Counter()
        self._confidence_values: list[float] = []

        weights = config.detector.pretrain_weights
        model_class = _load_rfdetr_small()
        if weights is None:
            self.model = model_class(resolution=self.input_size)
        else:
            weights_path = Path(weights)
            if not weights_path.is_absolute():
                weights_path = project_root / weights_path
            if not weights_path.exists():
                raise FileNotFoundError(
                    f"RF-DETR-S checkpoint not found: {weights_path}"
                )
            self.model = model_class(
                pretrain_weights=str(weights_path),
                resolution=self.input_size,
            )
        self.class_names = _coerce_class_names(getattr(self.model, "class_names", {}))

    def detect(self, image: np.ndarray) -> list[Detection]:
        return self.detect_batch([image])[0]

    def detect_batch(self, images: Sequence[np.ndarray]) -> list[list[Detection]]:
        if not images:
            return []
        rgb_images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
        predictions = self.model.predict(
            rgb_images,
            threshold=self.config.detector.confidence,
            shape=(self.input_size, self.input_size),
            include_source_image=self.config.detector.include_source_image,
        )
        detections_by_image = self._normalize_prediction_result(predictions)
        if len(detections_by_image) != len(images):
            raise RuntimeError(
                "RF-DETR-S returned "
                f"{len(detections_by_image)} detection sets for {len(images)} images"
            )
        return [
            self._convert_detections(detections, image.shape)
            for detections, image in zip(detections_by_image, images, strict=True)
        ]

    def detection_diagnostics(self) -> dict[str, object]:
        return {
            "counts": dict(self._counts),
            "confidence_values": list(self._confidence_values),
            "model": "rfdetr-small",
            "input_size": self.input_size,
            "runtime": "rfdetr",
        }

    @staticmethod
    def _normalize_prediction_result(predictions: object) -> list[sv.Detections]:
        if isinstance(predictions, sv.Detections):
            return [predictions]
        if isinstance(predictions, Sequence):
            return [
                prediction
                for prediction in predictions
                if isinstance(prediction, sv.Detections)
            ]
        raise TypeError(
            "Unsupported RF-DETR-S prediction result type: "
            f"{type(predictions).__name__}"
        )

    def _convert_detections(
        self, detections: sv.Detections, image_shape: tuple[int, ...]
    ) -> list[Detection]:
        height, width = image_shape[:2]
        confidences = (
            detections.confidence
            if detections.confidence is not None
            else np.ones(len(detections), dtype=np.float32)
        )
        class_ids = (
            detections.class_id
            if detections.class_id is not None
            else np.full(len(detections), -1, dtype=np.int32)
        )
        class_names = self._class_names_from_detections(detections)

        self._counts["raw_candidate_rows"] += len(detections)
        self._counts["detections_after_confidence_filtering"] += len(detections)

        converted: list[Detection] = []
        for index, xyxy in enumerate(detections.xyxy):
            confidence = float(confidences[index])
            class_id = int(class_ids[index])
            class_name = class_names[index]
            if self.allowed_class_names and class_name not in self.allowed_class_names:
                continue

            box = np.asarray(xyxy, dtype=np.float32).copy()
            box[[0, 2]] = np.clip(box[[0, 2]], 0, width - 1)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, height - 1)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue

            self._counts["detections_after_class_filtering"] += 1
            self._confidence_values.append(confidence)
            converted.append(
                Detection(
                    bbox=BBox(
                        x1=float(box[0]),
                        y1=float(box[1]),
                        x2=float(box[2]),
                        y2=float(box[3]),
                    ),
                    confidence=confidence,
                    class_id=class_id if class_id >= 0 else None,
                    class_name=class_name,
                )
            )
        return converted

    def _class_names_from_detections(self, detections: sv.Detections) -> list[str]:
        raw_names = detections.data.get("class_name") if detections.data else None
        names: list[str] = []
        for index in range(len(detections)):
            if raw_names is not None and index < len(raw_names):
                names.append(str(raw_names[index]).lower())
                continue
            class_id = (
                int(detections.class_id[index])
                if detections.class_id is not None
                else -1
            )
            names.append(self.class_names.get(class_id, ""))
        return names
