import numpy as np
import pytest

from config import AppConfig
from detectors.onnxruntime_local import (
    OnnxRuntimeLocalDetector,
    _allowed_class_names,
    _fixed_batch_size,
    _input_dtype_from_onnx_type,
    _is_dynamic_batch_dim,
    _letterbox,
    _metadata_names,
    _select_execution_providers,
    _validate_onnx_input_dtype,
)
from models import Detection


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


def test_letterbox_preserves_aspect_ratio() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    output, scale, padding = _letterbox(image, 640)

    assert output.shape == (640, 640, 3)
    assert scale == 3.2
    assert padding == (0, 160)


def test_parse_yolo_detection_output_filters_allowed_car() -> None:
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


def test_input_dtype_from_onnx_type_maps_float32() -> None:
    dtype, dtype_name = _input_dtype_from_onnx_type("tensor(float)")

    assert dtype == np.dtype(np.float32)
    assert dtype_name == "float32"


def test_input_dtype_from_onnx_type_maps_float16() -> None:
    dtype, dtype_name = _input_dtype_from_onnx_type("tensor(float16)")

    assert dtype == np.dtype(np.float16)
    assert dtype_name == "float16"


def test_input_dtype_from_onnx_type_rejects_unsupported_type() -> None:
    with pytest.raises(ValueError, match="Unsupported ONNX model input dtype"):
        _input_dtype_from_onnx_type("tensor(int8)")


def test_validate_onnx_input_dtype_accepts_auto() -> None:
    _validate_onnx_input_dtype(
        configured="auto", inferred="float16", onnx_type="tensor(float16)"
    )


def test_validate_onnx_input_dtype_rejects_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match ONNX model input type"):
        _validate_onnx_input_dtype(
            configured="float16", inferred="float32", onnx_type="tensor(float)"
        )


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
    class FakeDetector(OnnxRuntimeLocalDetector):
        calls: list[np.ndarray]

        def detect(self, image: np.ndarray) -> list[Detection]:
            self.calls.append(image)
            return []

    detector = FakeDetector.__new__(FakeDetector)
    detector.fixed_batch_size = 1
    detector.dynamic_batch = False
    detector.calls = []
    images = [
        np.zeros((10, 10, 3), dtype=np.uint8),
        np.zeros((20, 20, 3), dtype=np.uint8),
    ]

    assert detector.detect_batch(images) == [[], []]
    assert detector.calls[0] is images[0]
    assert detector.calls[1] is images[1]


def test_detect_batch_pads_fixed_batch_with_letterbox_background() -> None:
    captured_inputs: list[np.ndarray] = []

    class FakeSession:
        def run(
            self, output_names: object, input_feed: dict[str, np.ndarray]
        ) -> list[np.ndarray]:
            captured_inputs.append(input_feed["images"])
            return [np.zeros((4, 1, 6), dtype=np.float32)]

    tensors = [
        np.full((3, 8, 8), 0.1, dtype=np.float32),
        np.full((3, 8, 8), 0.2, dtype=np.float32),
    ]

    class FakeDetector(OnnxRuntimeLocalDetector):
        captured_preprocesses: list[np.ndarray]

        def _preprocess(
            self, image: np.ndarray
        ) -> tuple[np.ndarray, float, float, float, tuple[int, ...]]:
            index = len(self.captured_preprocesses)
            self.captured_preprocesses.append(image)
            return tensors[index], 1.0, 0, 0, image.shape

        def _parse_single_output(
            self,
            output: object,
            image_shape: tuple[int, ...],
            scale: float,
            pad_x: float,
            pad_y: float,
        ) -> list[Detection]:
            return []

    detector = FakeDetector.__new__(FakeDetector)
    detector.fixed_batch_size = 4
    detector.dynamic_batch = False
    detector.input_name = "images"
    detector.session = FakeSession()
    detector.captured_preprocesses = []
    images = [
        np.zeros((10, 10, 3), dtype=np.uint8),
        np.zeros((20, 20, 3), dtype=np.uint8),
    ]

    assert detector.detect_batch(images) == [[], []]
    assert len(detector.captured_preprocesses) == 2
    assert captured_inputs[0].shape == (4, 3, 8, 8)
    np.testing.assert_allclose(captured_inputs[0][0], tensors[0])
    np.testing.assert_allclose(captured_inputs[0][1], tensors[1])
    np.testing.assert_allclose(captured_inputs[0][2:], 114.0 / 255.0)


def test_preprocess_uses_configured_model_input_dtype() -> None:
    config = AppConfig.model_validate({"analysis": {"imgsz": 8}})
    detector = OnnxRuntimeLocalDetector.__new__(OnnxRuntimeLocalDetector)
    detector.config = config
    detector.input_dtype = np.dtype(np.float16)

    tensor, scale, pad_x, pad_y, image_shape = detector._preprocess(
        np.full((4, 8, 3), 255, dtype=np.uint8)
    )

    assert tensor.dtype == np.float16
    assert tensor.shape == (3, 8, 8)
    assert scale == 1.0
    assert pad_x == 0.0
    assert pad_y == 2.0
    assert image_shape == (4, 8, 3)


def test_parse_outputs_converts_model_output_to_float32_before_postprocessing() -> None:
    config = AppConfig()

    class CapturingDetector(OnnxRuntimeLocalDetector):
        captured_dtype: object = None

        def _candidate_rows(self, raw: np.ndarray) -> np.ndarray:
            self.captured_dtype = raw.dtype
            return np.empty((0, 6), dtype=np.float32)

    detector = CapturingDetector.__new__(CapturingDetector)
    detector.config = config
    detector.class_names = {2: "car"}
    detector.allowed_names = {"car"}
    detector.allowed_ids = None

    detections = detector._parse_outputs(
        [np.zeros((1, 6, 1), dtype=np.float16)],
        image_shape=(640, 640, 3),
        scale=1.0,
        pad_x=0,
        pad_y=0,
    )

    assert detections == []
    assert detector.captured_dtype == np.float32
