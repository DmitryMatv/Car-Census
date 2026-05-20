from car_census_cli import _accelerator_overrides


def test_default_accelerator_preserves_device_choice() -> None:
    assert _accelerator_overrides("default", "cpu") == {"project": {"device": "cpu"}}


def test_colab_t4_accelerator_enables_cuda_onnx_and_auto_nvenc() -> None:
    overrides = _accelerator_overrides("colab-t4", "cpu")

    assert overrides["project"]["device"] == "cuda"
    assert overrides["detector"]["onnx_execution_providers"] == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert overrides["detector"]["onnx_require_gpu"] is True
    assert overrides["render"]["encode_backend"] == "auto-nvenc"
    assert overrides["render"]["output_fps"] == 30.0
    assert overrides["render"]["nvenc_preset"] == "p4"
    assert overrides["render"]["nvenc_cq"] == 23


def test_onnx_cuda_accelerator_enables_cuda_without_render_override() -> None:
    overrides = _accelerator_overrides("onnx-cuda", "cpu")

    assert overrides == {
        "project": {"device": "cuda"},
        "detector": {
            "onnx_execution_providers": [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
            "onnx_require_gpu": True,
        },
    }


def test_tensorrt_accelerator_requires_gpu_providers_in_order() -> None:
    overrides = _accelerator_overrides("tensorrt", "cpu")

    assert overrides["project"]["device"] == "cuda"
    assert overrides["detector"]["onnx_execution_providers"] == [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert overrides["detector"]["onnx_require_gpu"] is True
