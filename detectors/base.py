from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import supervision as sv


class Detector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> sv.Detections:
        raise NotImplementedError

    def detect_batch(self, images: Sequence[np.ndarray]) -> list[sv.Detections]:
        return [self.detect(image) for image in images]
