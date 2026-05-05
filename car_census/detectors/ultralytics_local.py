from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO

from car_census.config import AppConfig
from car_census.detectors.base import Detector
from car_census.types import BBox, Detection
from car_census.utils.device import log_device_status


class UltralyticsLocalDetector(Detector):
    def __init__(self, config: AppConfig, project_root: Path) -> None:
        self.config = config
        self.device_status = log_device_status(config.project.device)
        self.device = self.device_status.resolved
        weights_path = Path(config.detector.weights)
        if not weights_path.is_absolute():
            weights_path = project_root / weights_path
        weights_value: str
        if weights_path.exists():
            weights_value = str(weights_path)
        else:
            weights_value = config.detector.weights
        self.model = YOLO(weights_value)

    def detect(self, image: np.ndarray) -> list[Detection]:
        result = self.model.predict(
            source=image,
            imgsz=self.config.analysis.imgsz,
            conf=self.config.detector.confidence,
            iou=self.config.detector.iou,
            device=self.device,
            verbose=False,
        )[0]

        class_names = {
            int(class_id): str(name).lower()
            for class_id, name in getattr(result, "names", {}).items()
        }
        allowed_names = {name.lower() for name in self.config.detector.allowed_class_names}
        detections: list[Detection] = []

        if result.boxes is None:
            return detections

        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        confidences = result.boxes.conf.detach().cpu().numpy()
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)

        for box, confidence, class_id in zip(xyxy, confidences, class_ids, strict=True):
            class_name = class_names.get(class_id)
            if self.config.detector.allowed_class_ids is not None and class_id not in self.config.detector.allowed_class_ids:
                continue
            if allowed_names and class_name and class_name.lower() not in allowed_names:
                continue
            detections.append(
                Detection(
                    bbox=BBox(x1=float(box[0]), y1=float(box[1]), x2=float(box[2]), y2=float(box[3])),
                    confidence=float(confidence),
                    class_id=int(class_id),
                    class_name=class_name,
                )
            )
        return detections
