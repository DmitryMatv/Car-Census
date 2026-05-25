import numpy as np
import pytest

from detectors.onnxruntime_session import (
    _fixed_batch_size,
    _input_dtype_from_onnx_type,
    _is_dynamic_batch_dim,
    _select_execution_providers,
    _validate_onnx_input_dtype,
)


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
    with pytest.raises(RuntimeError, match="onnxruntime-gpu"):
        _select_execution_providers(
            requested=["CUDAExecutionProvider", "CPUExecutionProvider"],
            available=["CPUExecutionProvider"],
            require_gpu=True,
        )


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
