import torch

from car_census.utils.device import resolve_device


def test_resolve_device_prefers_cpu_when_cuda_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"


def test_resolve_device_falls_back_for_legacy_gpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=0: (5, 2))
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_75", "sm_80"])
    assert resolve_device("auto") == "cpu"


def test_resolve_device_keeps_supported_gpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=0: (7, 5))
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_75", "sm_80"])
    assert resolve_device("auto") == "cuda:0"


def test_resolve_device_cuda_raises_when_unsupported(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=0: (5, 2))
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_75", "sm_80"])
    try:
        resolve_device("cuda")
    except RuntimeError as exc:
        assert "sm_52" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for unsupported CUDA")
