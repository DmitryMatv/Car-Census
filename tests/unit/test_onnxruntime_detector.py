import numpy as np

from config import AppConfig
from detectors.onnxruntime_local import (
    OnnxRuntimeLocalDetector,
    _fixed_batch_size,
    _is_dynamic_batch_dim,
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


def test_onnx_batch_dimension_detection() -> None:
    assert _is_dynamic_batch_dim(1) is False
    assert _fixed_batch_size(1) == 1
    assert _is_dynamic_batch_dim("batch") is True
    assert _fixed_batch_size("batch") is None
    assert _is_dynamic_batch_dim(None) is True
    assert _fixed_batch_size(None) is None
    assert _is_dynamic_batch_dim(-1) is True
    assert _fixed_batch_size(-1) is None
    assert _is_dynamic_batch_dim(16) is False
    assert _fixed_batch_size(16) == 16


def test_parse_batched_detection_output_per_image() -> None:
    config = AppConfig()
    detector = OnnxRuntimeLocalDetector.__new__(OnnxRuntimeLocalDetector)
    detector.config = config
    detector.class_names = {0: "person", 2: "car"}
    detector.allowed_names = {"car"}
    detector.allowed_ids = None

    raw = np.zeros((2, 3, 6), dtype=np.float32)
    raw[0, 0] = [10, 10, 30, 30, 0.9, 2]
    raw[0, 1] = [100, 100, 120, 120, 0.2, 2]
    raw[0, 2] = [50, 50, 70, 70, 0.95, 0]
    raw[1, 0] = [40, 40, 80, 90, 0.8, 2]
    raw[1, 1] = [42, 42, 82, 92, 0.7, 2]
    raw[1, 2] = [100, 100, 120, 120, 0.95, 0]

    first = detector._parse_single_output(raw[0], (100, 100, 3), 1.0, 0, 0)
    second = detector._parse_single_output(raw[1], (100, 100, 3), 1.0, 0, 0)

    assert len(first) == 1
    assert first[0].bbox.x1 == 10
    assert first[0].bbox.y2 == 30
    assert len(second) == 1
    assert second[0].bbox.x1 == 40
    assert second[0].bbox.y2 == 90


def test_detect_batch_falls_back_for_fixed_batch_one_model() -> None:
    detector = OnnxRuntimeLocalDetector.__new__(OnnxRuntimeLocalDetector)
    detector.fixed_batch_size = 1
    detector.dynamic_batch = False
    calls = []

    def fake_detect(image):
        calls.append(image)
        return []

    detector.detect = fake_detect
    images = [
        np.zeros((10, 10, 3), dtype=np.uint8),
        np.zeros((20, 20, 3), dtype=np.uint8),
    ]

    assert detector.detect_batch(images) == [[], []]
    assert calls[0] is images[0]
    assert calls[1] is images[1]
