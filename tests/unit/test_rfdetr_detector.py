from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import supervision as sv

from config import AppConfig
from detectors import rfdetr_local
from detectors.rfdetr_local import RfDetrMediumDetector
from pipeline.detections import class_names, detection_bboxes


class FakeRfDetrMedium:
    class_names = {2: "car", 3: "motorcycle", 7: "truck"}
    calls: list[dict[str, Any]] = []
    default_model_device: object = "cpu"
    optimize_calls: list[dict[str, Any]] = []
    predictions: object = []

    def __init__(
        self,
        pretrain_weights: str | None = None,
        resolution: int | None = None,
        device: str | None = None,
    ) -> None:
        self.pretrain_weights = pretrain_weights
        self.resolution = resolution
        self.device = device
        self.model = SimpleNamespace(
            device=device if device is not None else self.default_model_device
        )

    def optimize_for_inference(self, **kwargs: Any) -> None:
        self.optimize_calls.append(kwargs)

    def predict(self, images: list[np.ndarray], **kwargs: Any) -> object:
        self.calls.append({"images": images, "kwargs": kwargs})
        return self.predictions


@pytest.fixture(autouse=True)
def fake_model(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeRfDetrMedium.calls = []
    FakeRfDetrMedium.default_model_device = "cpu"
    FakeRfDetrMedium.optimize_calls = []
    FakeRfDetrMedium.predictions = []
    monkeypatch.setattr(rfdetr_local, "_load_rfdetr_medium", lambda: FakeRfDetrMedium)


def _detections(
    *,
    xyxy: list[list[float]],
    confidence: list[float],
    class_id: list[int],
    class_name: list[str] | None = None,
) -> sv.Detections:
    data: dict[str, Any] = (
        {} if class_name is None else {"class_name": np.array(class_name)}
    )
    return sv.Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        confidence=np.array(confidence, dtype=np.float32),
        class_id=np.array(class_id, dtype=np.int32),
        data=data,
    )


def test_detect_batch_calls_rfdetr_with_rgb_images_and_configured_options() -> None:
    FakeRfDetrMedium.predictions = [
        _detections(
            xyxy=[[1, 2, 11, 22]],
            confidence=[0.91],
            class_id=[2],
            class_name=["car"],
        )
    ]
    config = AppConfig.model_validate(
        {"detector": {"confidence": 0.42, "input_size": 576}}
    )
    detector = RfDetrMediumDetector(config=config, project_root=Path("."))
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[0, 0] = [10, 20, 30]

    detections = detector.detect_batch([image])

    assert len(detections) == 1
    assert isinstance(detections[0], sv.Detections)
    assert class_names(detections[0]) == ["car"]
    call = FakeRfDetrMedium.calls[0]
    assert call["kwargs"] == {
        "threshold": 0.42,
        "shape": (576, 576),
        "include_source_image": False,
    }
    assert call["images"][0][0, 0].tolist() == [30, 20, 10]


def test_detector_passes_configured_cpu_device_to_rfdetr() -> None:
    config = AppConfig.model_validate({"detector": {"device": "cpu"}})
    detector = RfDetrMediumDetector(config=config, project_root=Path("."))

    assert getattr(detector, "model").device == "cpu"


def test_detector_optimizes_cuda_auto_dtype_as_float16() -> None:
    config = AppConfig.model_validate({"detector": {"device": "cuda"}})

    RfDetrMediumDetector(config=config, project_root=Path("."))

    assert FakeRfDetrMedium.optimize_calls == [{"compile": False, "dtype": "float16"}]


def test_detector_optimizes_cpu_auto_dtype_as_float32() -> None:
    config = AppConfig.model_validate({"detector": {"device": "cpu"}})

    RfDetrMediumDetector(config=config, project_root=Path("."))

    assert FakeRfDetrMedium.optimize_calls == [{"compile": False, "dtype": "float32"}]


def test_detector_uses_cuda_fallback_when_model_device_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    FakeRfDetrMedium.default_model_device = None
    monkeypatch.setattr(rfdetr_local, "_cuda_is_available", lambda: True)

    RfDetrMediumDetector(config=AppConfig(), project_root=Path("."))

    assert FakeRfDetrMedium.optimize_calls == [{"compile": False, "dtype": "float16"}]
    assert "using float16 because CUDA is available" in caplog.text


def test_detector_uses_float32_fallback_when_model_device_and_cuda_are_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    FakeRfDetrMedium.default_model_device = None
    monkeypatch.setattr(rfdetr_local, "_cuda_is_available", lambda: False)

    RfDetrMediumDetector(config=AppConfig(), project_root=Path("."))

    assert FakeRfDetrMedium.optimize_calls == [{"compile": False, "dtype": "float32"}]
    assert "using float32 because CUDA is not available" in caplog.text


def test_detector_skips_optimization_when_disabled() -> None:
    config = AppConfig.model_validate({"detector": {"optimize_for_inference": False}})

    RfDetrMediumDetector(config=config, project_root=Path("."))

    assert FakeRfDetrMedium.optimize_calls == []


def test_detector_uses_explicit_inference_dtype() -> None:
    config = AppConfig.model_validate(
        {"detector": {"device": "cpu", "inference_dtype": "float16"}}
    )

    RfDetrMediumDetector(config=config, project_root=Path("."))

    assert FakeRfDetrMedium.optimize_calls == [{"compile": False, "dtype": "float16"}]


def test_detector_passes_batch_size_only_when_compiling_for_inference() -> None:
    config = AppConfig.model_validate(
        {
            "analysis": {"batch_size": 8, "detector_batch_size": 3},
            "detector": {"compile_for_inference": True},
        }
    )

    RfDetrMediumDetector(config=config, project_root=Path("."))

    assert FakeRfDetrMedium.optimize_calls == [
        {"compile": True, "dtype": "float32", "batch_size": 3}
    ]


def test_detect_delegates_to_single_item_batch() -> None:
    FakeRfDetrMedium.predictions = _detections(
        xyxy=[[1, 2, 11, 22]],
        confidence=[0.91],
        class_id=[2],
        class_name=["car"],
    )
    detector = RfDetrMediumDetector(config=AppConfig(), project_root=Path("."))

    detections = detector.detect(np.zeros((24, 32, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert isinstance(detections, sv.Detections)
    assert detection_bboxes(detections)[0].x1 == 1


def test_detect_batch_filters_to_allowed_class_names() -> None:
    FakeRfDetrMedium.predictions = [
        _detections(
            xyxy=[[1, 2, 11, 22], [3, 4, 13, 24]],
            confidence=[0.91, 0.85],
            class_id=[2, 7],
            class_name=["car", "truck"],
        )
    ]
    detector = RfDetrMediumDetector(config=AppConfig(), project_root=Path("."))

    detections = detector.detect_batch([np.zeros((24, 32, 3), dtype=np.uint8)])

    assert class_names(detections[0]) == ["car"]


def test_detect_batch_preserves_multi_image_prediction_order() -> None:
    FakeRfDetrMedium.predictions = [
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
    detector = RfDetrMediumDetector(config=AppConfig(), project_root=Path("."))
    images = [
        np.zeros((24, 32, 3), dtype=np.uint8),
        np.zeros((48, 64, 3), dtype=np.uint8),
    ]

    detections = detector.detect_batch(images)

    assert len(FakeRfDetrMedium.calls) == 1
    assert len(FakeRfDetrMedium.calls[0]["images"]) == 2
    assert len(detections) == 2
    assert isinstance(detections[0], sv.Detections)
    assert isinstance(detections[1], sv.Detections)
    assert detection_bboxes(detections[0])[0].x1 == 1
    assert detection_bboxes(detections[1])[0].x1 == 3


def test_detect_batch_falls_back_to_model_class_names() -> None:
    FakeRfDetrMedium.predictions = [
        _detections(
            xyxy=[[1, 2, 11, 22], [3, 4, 13, 24]],
            confidence=[0.91, 0.85],
            class_id=[2, 3],
        )
    ]
    detector = RfDetrMediumDetector(config=AppConfig(), project_root=Path("."))

    detections = detector.detect_batch([np.zeros((24, 32, 3), dtype=np.uint8)])

    assert class_names(detections[0]) == ["car"]


def test_detect_batch_preserves_existing_class_names_when_filling_missing() -> None:
    FakeRfDetrMedium.predictions = [
        _detections(
            xyxy=[[1, 2, 11, 22], [3, 4, 13, 24]],
            confidence=[0.91, 0.85],
            class_id=[2, 7],
            class_name=["automobile", ""],
        )
    ]
    detector = RfDetrMediumDetector(
        config=AppConfig.model_validate(
            {"detector": {"allowed_class_names": ["automobile", "truck"]}}
        ),
        project_root=Path("."),
    )

    detections = detector.detect_batch([np.zeros((24, 32, 3), dtype=np.uint8)])

    assert class_names(detections[0]) == ["automobile", "truck"]


def test_detect_batch_clips_boxes_and_drops_invalid_boxes() -> None:
    FakeRfDetrMedium.predictions = [
        _detections(
            xyxy=[[-5, -6, 50, 40], [20, 20, 20, 25]],
            confidence=[0.91, 0.85],
            class_id=[2, 2],
            class_name=["car", "car"],
        )
    ]
    detector = RfDetrMediumDetector(config=AppConfig(), project_root=Path("."))

    detections = detector.detect_batch([np.zeros((24, 32, 3), dtype=np.uint8)])

    assert len(detections[0]) == 1
    assert detection_bboxes(detections[0])[0].model_dump() == {
        "x1": 0.0,
        "y1": 0.0,
        "x2": 31.0,
        "y2": 23.0,
    }


def test_detection_diagnostics_uses_existing_count_keys() -> None:
    FakeRfDetrMedium.predictions = [
        _detections(
            xyxy=[[1, 2, 11, 22], [3, 4, 13, 24]],
            confidence=[0.91, 0.85],
            class_id=[2, 7],
            class_name=["car", "truck"],
        )
    ]
    detector = RfDetrMediumDetector(config=AppConfig(), project_root=Path("."))

    detector.detect_batch([np.zeros((24, 32, 3), dtype=np.uint8)])

    diagnostics = detector.detection_diagnostics()
    assert diagnostics["counts"] == {
        "raw_candidate_rows": 2,
        "detections_after_confidence_filtering": 2,
        "detections_after_class_filtering": 1,
    }
    confidence_histogram = diagnostics["confidence_histogram"]
    assert isinstance(confidence_histogram, list)
    assert sum(bucket["count"] for bucket in confidence_histogram) == 1
    assert diagnostics == {
        "counts": {
            "raw_candidate_rows": 2,
            "detections_after_confidence_filtering": 2,
            "detections_after_class_filtering": 1,
        },
        "confidence_histogram": confidence_histogram,
        "model": "rfdetr-medium",
        "input_size": 576,
        "runtime": "rfdetr",
        "optimized_for_inference": True,
        "inference_dtype": "float32",
        "compiled_for_inference": False,
    }


def test_local_pretrain_weights_are_resolved_from_project_root(tmp_path: Path) -> None:
    weights = tmp_path / "weights" / "rfdetr-medium.pth"
    weights.parent.mkdir()
    weights.write_bytes(b"placeholder")
    config = AppConfig.model_validate(
        {"detector": {"pretrain_weights": "weights/rfdetr-medium.pth"}}
    )

    detector = RfDetrMediumDetector(config=config, project_root=tmp_path)

    model = getattr(detector, "model")
    assert model.pretrain_weights == str(weights)
    assert model.resolution == 576
