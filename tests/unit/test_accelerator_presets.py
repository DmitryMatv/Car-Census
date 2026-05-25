from car_census_cli import _accelerator_overrides


def test_default_accelerator_has_no_overrides() -> None:
    assert _accelerator_overrides("default") == {}


def test_colab_t4_accelerator_enables_auto_nvenc_only() -> None:
    overrides = _accelerator_overrides("colab-t4")

    assert "detector" not in overrides
    assert overrides["render"]["encode_backend"] == "auto-nvenc"
    assert overrides["render"]["output_fps"] == 30.0
    assert overrides["render"]["nvenc_preset"] == "p4"
    assert overrides["render"]["nvenc_cq"] == 23


def test_removed_gpu_accelerator_is_rejected() -> None:
    try:
        _accelerator_overrides("gpu")
    except Exception as exc:
        assert "Unsupported accelerator" in str(exc)
    else:
        raise AssertionError("gpu should be rejected")
