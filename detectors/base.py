from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from config import AppConfig
from models import Detection


class Detector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Detection]:
        raise NotImplementedError


def create_detector(config: AppConfig, project_root: Path) -> Detector:
    if config.detector.provider == "onnxruntime_local":
        from detectors.onnxruntime_local import OnnxRuntimeLocalDetector

        return OnnxRuntimeLocalDetector(config=config, project_root=project_root)
    if config.detector.provider == "ultralytics_local":
        from detectors.ultralytics_local import UltralyticsLocalDetector

        return UltralyticsLocalDetector(config=config, project_root=project_root)
    raise ValueError(f"Unsupported detector provider: {config.detector.provider}")
