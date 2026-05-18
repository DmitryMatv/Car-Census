import numpy as np

from detectors.base import Detector
from models import BBox, Detection


class BatchFallbackDetector(Detector):
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, image: np.ndarray) -> list[Detection]:
        self.calls += 1
        return [
            Detection(
                bbox=BBox(
                    x1=0,
                    y1=0,
                    x2=float(image.shape[1]),
                    y2=float(image.shape[0]),
                ),
                confidence=0.9,
                class_id=2,
                class_name="car",
            )
        ]


def test_detector_detect_batch_defaults_to_single_frame_calls() -> None:
    detector = BatchFallbackDetector()
    images = [
        np.zeros((10, 20, 3), dtype=np.uint8),
        np.zeros((30, 40, 3), dtype=np.uint8),
    ]

    detections = detector.detect_batch(images)

    assert detector.calls == 2
    assert len(detections) == 2
    assert detections[0][0].bbox.x2 == 20
    assert detections[1][0].bbox.x2 == 40
