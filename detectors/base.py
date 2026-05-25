from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from models import Detection


class Detector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Detection]:
        raise NotImplementedError

    def detect_batch(self, images: Sequence[np.ndarray]) -> list[list[Detection]]:
        return [self.detect(image) for image in images]
