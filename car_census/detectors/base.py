from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from car_census.config import AppConfig
from car_census.types import Detection


class Detector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Detection]:
        raise NotImplementedError


def create_detector(config: AppConfig, project_root: Path) -> Detector:
    if config.detector.provider != "ultralytics_local":
        raise ValueError(f"Unsupported detector provider: {config.detector.provider}")
    from car_census.detectors.ultralytics_local import UltralyticsLocalDetector

    return UltralyticsLocalDetector(config=config, project_root=project_root)
