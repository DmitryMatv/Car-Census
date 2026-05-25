import numpy as np

from config import AppConfig
from detectors.yolo_postprocess import (
    YoloDetectionPolicy,
    YoloOutputParser,
    _allowed_class_names,
    _metadata_names,
)


def _parser() -> YoloOutputParser:
    config = AppConfig()
    return YoloOutputParser(
        YoloDetectionPolicy(
            confidence=config.detector.confidence,
            iou=config.detector.iou,
            class_names={0: "person", 2: "car"},
            allowed_names={"car"},
            allowed_ids=None,
        )
    )


def test_metadata_names_accepts_python_mapping() -> None:
    assert _metadata_names("{0: 'person', 2: 'car'}") == {
        0: "person",
        2: "car",
    }


def test_allowed_class_names_adds_discovered_vehicle_metadata() -> None:
    assert _allowed_class_names(
        ["car", "truck"],
        {0: "person", 2: "car", 7: "truck", 80: "van", 81: "pickup"},
    ) == {"car", "truck", "van", "pickup"}


def test_parse_yolo_detection_output_filters_allowed_car() -> None:
    parser = _parser()
    raw = np.zeros((1, 6, 3), dtype=np.float32)
    raw[0, :, 0] = [50, 50, 40, 20, 0.9, 2]
    raw[0, :, 1] = [52, 52, 40, 20, 0.8, 2]
    raw[0, :, 2] = [100, 100, 10, 20, 0.95, 0]

    detections = parser.parse_outputs(
        [raw],
        image_shape=(640, 640, 3),
        scale=1.0,
        pad_x=0,
        pad_y=0,
    )

    assert len(detections) == 1
    assert detections[0].class_id == 2
    assert detections[0].class_name == "car"
    assert detections[0].bbox.x1 == 30
    assert detections[0].bbox.y1 == 40


def test_parse_batched_detection_output_per_image() -> None:
    parser = _parser()
    raw = np.zeros((2, 3, 6), dtype=np.float32)
    raw[0, 0] = [10, 10, 30, 30, 0.9, 2]
    raw[0, 1] = [100, 100, 120, 120, 0.2, 2]
    raw[0, 2] = [50, 50, 70, 70, 0.95, 0]
    raw[1, 0] = [40, 40, 80, 90, 0.8, 2]
    raw[1, 1] = [42, 42, 82, 92, 0.7, 2]
    raw[1, 2] = [100, 100, 120, 120, 0.95, 0]

    first = parser.parse_single(raw[0], (100, 100, 3), 1.0, 0, 0)
    second = parser.parse_single(raw[1], (100, 100, 3), 1.0, 0, 0)

    assert len(first) == 1
    assert first[0].bbox.x1 == 10
    assert first[0].bbox.y2 == 30
    assert len(second) == 1
    assert second[0].bbox.x1 == 40
    assert second[0].bbox.y2 == 90


def test_parse_outputs_converts_model_output_to_float32_before_postprocessing() -> None:
    class CapturingParser(YoloOutputParser):
        captured_dtype: object = None

        def _candidate_rows(self, raw: np.ndarray) -> np.ndarray:
            self.captured_dtype = raw.dtype
            return np.empty((0, 6), dtype=np.float32)

    parser = CapturingParser(_parser().policy)

    detections = parser.parse_outputs(
        [np.zeros((1, 6, 1), dtype=np.float16)],
        image_shape=(640, 640, 3),
        scale=1.0,
        pad_x=0,
        pad_y=0,
    )

    assert detections == []
    assert parser.captured_dtype == np.float32


def test_parser_records_diagnostics() -> None:
    parser = _parser()
    raw = np.array([[10, 10, 30, 30, 0.9, 2]], dtype=np.float32)

    parser.parse_single(raw, (100, 100, 3), 1.0, 0, 0)

    diagnostics = parser.diagnostics.as_dict()
    assert diagnostics["counts"] == {
        "raw_candidate_rows": 1,
        "detections_after_confidence_filtering": 1,
        "detections_after_class_filtering": 1,
    }
    assert diagnostics["confidence_values"] == [0.8999999761581421]
