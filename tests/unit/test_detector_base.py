import numpy as np
import supervision as sv

from detectors.base import Detector


class BatchFallbackDetector(Detector):
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, image: np.ndarray) -> sv.Detections:
        self.calls += 1
        return sv.Detections(
            xyxy=np.array(
                [[0, 0, float(image.shape[1]), float(image.shape[0])]],
                dtype=np.float32,
            ),
            confidence=np.array([0.9], dtype=np.float32),
            class_id=np.array([2], dtype=np.int32),
            data={"class_name": np.array(["car"], dtype=object)},
        )


def test_detector_detect_batch_defaults_to_single_frame_calls() -> None:
    detector = BatchFallbackDetector()
    images = [
        np.zeros((10, 20, 3), dtype=np.uint8),
        np.zeros((30, 40, 3), dtype=np.uint8),
    ]

    detections = detector.detect_batch(images)

    assert detector.calls == 2
    assert len(detections) == 2
    assert detections[0].xyxy[0, 2] == 20
    assert detections[1].xyxy[0, 2] == 40
