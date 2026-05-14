import numpy as np

from config import AppConfig
from detectors.onnxruntime_local import (
    OnnxRuntimeLocalDetector,
    _letterbox,
    _metadata_names,
    _select_execution_providers,
)


def test_metadata_names_accepts_ultralytics_mapping() -> None:
    assert _metadata_names("{0: 'person', 2: 'car'}") == {
        0: "person",
        2: "car",
    }


def test_letterbox_preserves_aspect_ratio() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    output, scale, padding = _letterbox(image, 640)

    assert output.shape == (640, 640, 3)
    assert scale == 3.2
    assert padding == (0, 160)


def test_parse_ultralytics_detection_output_filters_allowed_car() -> None:
    config = AppConfig()
    detector = OnnxRuntimeLocalDetector.__new__(OnnxRuntimeLocalDetector)
    detector.config = config
    detector.class_names = {0: "person", 2: "car"}
    detector.allowed_names = {"car"}
    detector.allowed_ids = None

    raw = np.zeros((1, 6, 3), dtype=np.float32)
    raw[0, :, 0] = [50, 50, 40, 20, 0.9, 2]
    raw[0, :, 1] = [52, 52, 40, 20, 0.8, 2]
    raw[0, :, 2] = [100, 100, 10, 20, 0.95, 0]

    detections = detector._parse_outputs(
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


def test_select_execution_providers_keeps_cpu_default() -> None:
    assert _select_execution_providers(
        requested=["CPUExecutionProvider"],
        available=["AzureExecutionProvider", "CPUExecutionProvider"],
        require_gpu=False,
    ) == ["CPUExecutionProvider"]


def test_select_execution_providers_uses_cuda_when_available() -> None:
    assert _select_execution_providers(
        requested=["CUDAExecutionProvider", "CPUExecutionProvider"],
        available=["CUDAExecutionProvider", "CPUExecutionProvider"],
        require_gpu=True,
    ) == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_select_execution_providers_rejects_required_missing_gpu() -> None:
    try:
        _select_execution_providers(
            requested=["CUDAExecutionProvider", "CPUExecutionProvider"],
            available=["CPUExecutionProvider"],
            require_gpu=True,
        )
    except RuntimeError as exc:
        assert "onnxruntime-gpu" in str(exc)
    else:
        raise AssertionError("Expected missing required GPU provider to fail")


def test_select_execution_providers_uses_tensorrt_order_when_available() -> None:
    assert _select_execution_providers(
        requested=[
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
        available=[
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
        require_gpu=True,
    ) == [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
