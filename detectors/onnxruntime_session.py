from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from config import AppConfig

logger = logging.getLogger(__name__)

GPU_EXECUTION_PROVIDERS = {"CUDAExecutionProvider", "TensorrtExecutionProvider"}
ONNX_INPUT_DTYPES: dict[str, tuple[str, np.dtype[Any]]] = {
    "tensor(float)": ("float32", np.dtype(np.float32)),
    "tensor(float16)": ("float16", np.dtype(np.float16)),
}


@dataclass(frozen=True, slots=True)
class OnnxRuntimeInputSpec:
    name: str
    dtype: np.dtype[Any]
    dtype_name: str
    batch_dim: Any
    dynamic_batch: bool
    fixed_batch_size: int | None


@dataclass(frozen=True, slots=True)
class OnnxRuntimeSessionBundle:
    session: Any
    input_spec: OnnxRuntimeInputSpec
    active_providers: list[str]


def _is_dynamic_batch_dim(batch_dim: Any) -> bool:
    if batch_dim is None:
        return True
    if isinstance(batch_dim, str):
        return True
    if isinstance(batch_dim, int):
        return batch_dim < 0
    return False


def _fixed_batch_size(batch_dim: Any) -> int | None:
    if _is_dynamic_batch_dim(batch_dim):
        return None
    if isinstance(batch_dim, int) and batch_dim > 0:
        return batch_dim
    return None


def _input_dtype_from_onnx_type(onnx_type: str) -> tuple[np.dtype[Any], str]:
    dtype_info = ONNX_INPUT_DTYPES.get(onnx_type)
    if dtype_info is None:
        supported = ", ".join(sorted(ONNX_INPUT_DTYPES))
        raise ValueError(
            "Unsupported ONNX model input dtype "
            f"{onnx_type!r}. Supported input types: {supported}."
        )
    dtype_name, dtype = dtype_info
    return dtype, dtype_name


def _validate_onnx_input_dtype(
    *, configured: str, inferred: str, onnx_type: str
) -> None:
    if configured == "auto" or configured == inferred:
        return
    raise ValueError(
        "detector.onnx_input_dtype="
        f"{configured!r} does not match ONNX model input type {onnx_type!r} "
        f"({inferred}). Re-export the model with the requested precision or set "
        "detector.onnx_input_dtype to 'auto'."
    )


def _select_execution_providers(
    *,
    requested: list[str],
    available: list[str],
    require_gpu: bool,
) -> list[str]:
    selected = [provider for provider in requested if provider in available]
    selected_gpu = [
        provider for provider in selected if provider in GPU_EXECUTION_PROVIDERS
    ]
    requested_gpu = [
        provider for provider in requested if provider in GPU_EXECUTION_PROVIDERS
    ]

    if require_gpu and not selected_gpu:
        install_hint = (
            "Install `onnxruntime-gpu` in Colab for CUDA support. "
            "TensorRT acceleration also requires ONNX Runtime TensorRT provider "
            "dependencies to be available in the runtime."
        )
        raise RuntimeError(
            "ONNX Runtime GPU execution was requested, but none of the requested "
            f"GPU providers are available. requested={requested_gpu or requested}; "
            f"available={available}. {install_hint}"
        )

    if selected:
        return selected

    if "CPUExecutionProvider" in available:
        logger.warning(
            "None of the requested ONNX Runtime providers are available; "
            "falling back to CPUExecutionProvider. requested=%s available=%s",
            requested,
            available,
        )
        return ["CPUExecutionProvider"]

    raise RuntimeError(
        "No usable ONNX Runtime execution provider is available. "
        f"requested={requested}; available={available}"
    )


def _resolve_weights_path(config: AppConfig, project_root: Path) -> Path:
    weights_path = Path(config.detector.weights)
    if not weights_path.is_absolute():
        weights_path = project_root / weights_path
    if not weights_path.exists():
        raise FileNotFoundError(f"ONNX detector weights not found: {weights_path}")
    return weights_path


def load_onnxruntime_session(
    config: AppConfig, project_root: Path
) -> OnnxRuntimeSessionBundle:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required for local ONNX detection. "
            "Install the project with `pip install -e .`."
        ) from exc

    weights_path = _resolve_weights_path(config, project_root)
    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, config.detector.onnx_threads)
    options.inter_op_num_threads = 1
    requested_providers = list(config.detector.onnx_execution_providers)
    available_providers = list(ort.get_available_providers())
    providers = _select_execution_providers(
        requested=requested_providers,
        available=available_providers,
        require_gpu=config.detector.onnx_require_gpu,
    )
    session = ort.InferenceSession(
        str(weights_path),
        sess_options=options,
        providers=providers,
    )
    input_info = session.get_inputs()[0]
    input_dtype, input_dtype_name = _input_dtype_from_onnx_type(input_info.type)
    _validate_onnx_input_dtype(
        configured=config.detector.onnx_input_dtype,
        inferred=input_dtype_name,
        onnx_type=input_info.type,
    )
    input_shape = list(input_info.shape)
    input_batch_dim = input_shape[0] if input_shape else None
    input_spec = OnnxRuntimeInputSpec(
        name=input_info.name,
        dtype=input_dtype,
        dtype_name=input_dtype_name,
        batch_dim=input_batch_dim,
        dynamic_batch=_is_dynamic_batch_dim(input_batch_dim),
        fixed_batch_size=_fixed_batch_size(input_batch_dim),
    )
    active_providers = list(session.get_providers())
    logger.info(
        "Device active: %s | provider=onnxruntime | dtype=%s | threads=%s | "
        "requested_providers=%s | available_providers=%s | active_providers=%s",
        "gpu"
        if any(provider in GPU_EXECUTION_PROVIDERS for provider in active_providers)
        else "cpu",
        input_spec.dtype_name,
        options.intra_op_num_threads,
        requested_providers,
        available_providers,
        active_providers,
    )
    logger.info(
        "ONNX batch support: dynamic=%s input_batch=%s configured_batch_size=%s",
        input_spec.dynamic_batch,
        input_spec.batch_dim,
        config.analysis.batch_size,
    )
    if (
        config.analysis.batch_size > 1
        and input_spec.fixed_batch_size == 1
        and not input_spec.dynamic_batch
    ):
        logger.warning(
            "analysis.batch_size=%s requested, but ONNX model input batch is "
            "fixed to 1; falling back to single-frame inference. Re-export ONNX "
            "with dynamic=True to enable batching.",
            config.analysis.batch_size,
        )
    return OnnxRuntimeSessionBundle(
        session=session,
        input_spec=input_spec,
        active_providers=active_providers,
    )
