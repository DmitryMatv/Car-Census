import torch

from utils.device import get_device_status


def test_get_device_status_reports_supported_gpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=0: (5, 2))
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_50", "sm_52"])
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda index=0: "NVIDIA GeForce GTX 970"
    )
    status = get_device_status("cuda")
    assert status.resolved == "cuda:0"
    assert status.using_gpu is True
    assert status.gpu_capability == "sm_52"
