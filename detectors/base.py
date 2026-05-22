from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import numpy as np

from config import AppConfig
from models import Detection


class Detector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Detection]:
        raise NotImplementedError

    def detect_batch(self, images: Sequence[np.ndarray]) -> list[list[Detection]]:
        return [self.detect(image) for image in images]


def create_detector(config: AppConfig, project_root: Path) -> Detector:
    from detectors.onnxruntime_local import OnnxRuntimeLocalDetector

    return OnnxRuntimeLocalDetector(config=config, project_root=project_root)
