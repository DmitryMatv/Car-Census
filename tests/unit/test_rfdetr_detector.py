from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import supervision as sv

from config import AppConfig
from detectors import rfdetr_local
from detectors.rfdetr_local import RfDetrSmallDetector


class FakeRfDetrSmall:
    class_names = {2: "car", 3: "motorcycle", 7: "truck"}
    calls: list[dict[str, Any]] = []
    predictions: object = []

    def __init__(
        self, pretrain_weights: str | None = None, resolution: int | None = None
    ) -> None:
        self.pretrain_weights = pretrain_weights
        self.resolution = resolution

    def predict(self, images: list[np.ndarray], **kwargs: Any) -> object:
        self.calls.append({"images": images, "kwargs": kwargs})
        return self.predictions


@pytest.fixture(autouse=True)
def fake_model(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeRfDetrSmall.calls = []
    FakeRfDetrSmall.predictions = []
    monkeypatch.setattr(rfdetr_local, "_load_rfdetr_small", lambda: FakeRfDetrSmall)


def _detections(
    *,
    xyxy: list[list[float]],
    confidence: list[float],
    class_id: list[int],
    class_name: list[str] | None = None,
) -> sv.Detections:
    data = {} if class_name is None else {"class_name": np.array(class_name)}
    return sv.Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        confidence=np.array(confidence, dtype=np.float32),
        class_id=np.array(class_id, dtype=np.int32),
        data=data,
    )


def test_detect_batch_calls_rfdetr_with_rgb_images_and_configured_options() -> None:
    FakeRfDetrSmall.predictions = [
        _detections(
            xyxy=[[1, 2, 11, 22]],
            confidence=[0.91],
            class_id=[2],
            class_name=["car"],
        )
    ]
    config = AppConfig.model_validate(
        {"detector": {"confidence": 0.42, "input_size": 512}}
    )
    detector = RfDetrSmallDetector(config=config, project_root=Path("."))
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[0, 0] = [10, 20, 30]

    detections = detector.detect_batch([image])

    assert len(detections) == 1
    assert detections[0][0].class_name == "car"
    call = FakeRfDetrSmall.calls[0]
    assert call["kwargs"] == {
        "threshold": 0.42,
        "shape": (512, 512),
        "include_source_image": False,
    }
    assert call["images"][0][0, 0].tolist() == [30, 20, 10]


def test_detect_delegates_to_single_item_batch() -> None:
    FakeRfDetrSmall.predictions = _detections(
        xyxy=[[1, 2, 11, 22]],
        confidence=[0.91],
        class_id=[2],
        class_name=["car"],
    )
    detector = RfDetrSmallDetector(config=AppConfig(), project_root=Path("."))

    detections = detector.detect(np.zeros((24, 32, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert detections[0].bbox.x1 == 1


def test_detect_batch_filters_to_allowed_class_names() -> None:
    FakeRfDetrSmall.predictions = [
        _detections(
            xyxy=[[1, 2, 11, 22], [3, 4, 13, 24]],
            confidence=[0.91, 0.85],
            class_id=[2, 7],
            class_name=["car", "truck"],
        )
    ]
    detector = RfDetrSmallDetector(config=AppConfig(), project_root=Path("."))

    detections = detector.detect_batch([np.zeros((24, 32, 3), dtype=np.uint8)])

    assert [detection.class_name for detection in detections[0]] == ["car"]


def test_detect_batch_preserves_multi_image_prediction_order() -> None:
    FakeRfDetrSmall.predictions = [
        _detections(
            xyxy=[[1, 2, 11, 22]],
            confidence=[0.91],
            class_id=[2],
            class_name=["car"],
        ),
        _detections(
            xyxy=[[3, 4, 13, 24]],
            confidence=[0.85],
            class_id=[2],
            class_name=["car"],
        ),
    ]
    detector = RfDetrSmallDetector(config=AppConfig(), project_root=Path("."))
    images = [
        np.zeros((24, 32, 3), dtype=np.uint8),
        np.zeros((48, 64, 3), dtype=np.uint8),
    ]

    detections = detector.detect_batch(images)

    assert len(FakeRfDetrSmall.calls) == 1
    assert len(FakeRfDetrSmall.calls[0]["images"]) == 2
    assert len(detections) == 2
    assert detections[0][0].bbox.x1 == 1
    assert detections[1][0].bbox.x1 == 3


def test_detect_batch_falls_back_to_model_class_names() -> None:
    FakeRfDetrSmall.predictions = [
        _detections(
            xyxy=[[1, 2, 11, 22], [3, 4, 13, 24]],
            confidence=[0.91, 0.85],
            class_id=[2, 3],
        )
    ]
    detector = RfDetrSmallDetector(config=AppConfig(), project_root=Path("."))

    detections = detector.detect_batch([np.zeros((24, 32, 3), dtype=np.uint8)])

    assert [detection.class_name for detection in detections[0]] == ["car"]


def test_detect_batch_clips_boxes_and_drops_invalid_boxes() -> None:
    FakeRfDetrSmall.predictions = [
        _detections(
            xyxy=[[-5, -6, 50, 40], [20, 20, 20, 25]],
            confidence=[0.91, 0.85],
            class_id=[2, 2],
            class_name=["car", "car"],
        )
    ]
    detector = RfDetrSmallDetector(config=AppConfig(), project_root=Path("."))

    detections = detector.detect_batch([np.zeros((24, 32, 3), dtype=np.uint8)])

    assert len(detections[0]) == 1
    assert detections[0][0].bbox.model_dump() == {
        "x1": 0.0,
        "y1": 0.0,
        "x2": 31.0,
        "y2": 23.0,
    }


def test_detection_diagnostics_uses_existing_count_keys() -> None:
    FakeRfDetrSmall.predictions = [
        _detections(
            xyxy=[[1, 2, 11, 22], [3, 4, 13, 24]],
            confidence=[0.91, 0.85],
            class_id=[2, 7],
            class_name=["car", "truck"],
        )
    ]
    detector = RfDetrSmallDetector(config=AppConfig(), project_root=Path("."))

    detector.detect_batch([np.zeros((24, 32, 3), dtype=np.uint8)])

    assert detector.detection_diagnostics() == {
        "counts": {
            "raw_candidate_rows": 2,
            "detections_after_confidence_filtering": 2,
            "detections_after_class_filtering": 1,
        },
        "confidence_values": [pytest.approx(0.91)],
        "model": "rfdetr-small",
        "input_size": 512,
        "runtime": "rfdetr",
    }


def test_local_pretrain_weights_are_resolved_from_project_root(tmp_path: Path) -> None:
    weights = tmp_path / "weights" / "rfdetr-small.pth"
    weights.parent.mkdir()
    weights.write_bytes(b"placeholder")
    config = AppConfig.model_validate(
        {"detector": {"pretrain_weights": "weights/rfdetr-small.pth"}}
    )

    detector = RfDetrSmallDetector(config=config, project_root=tmp_path)

    assert detector.model.pretrain_weights == str(weights)
    assert detector.model.resolution == 512
