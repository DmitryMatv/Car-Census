from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DeviceStatus:
    requested: str
    resolved: str
    gpu_available: bool
    gpu_supported: bool
    gpu_name: str | None
    gpu_capability: str | None
    arch_list: tuple[str, ...]

    @property
    def using_gpu(self) -> bool:
        return self.resolved.startswith("cuda")


def _supported_arch_list() -> set[str]:
    if hasattr(torch.cuda, "get_arch_list"):
        return {arch.lower() for arch in torch.cuda.get_arch_list()}
    return set()


def _supports_current_gpu() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    supported_arches = _supported_arch_list()
    if supported_arches and arch not in supported_arches:
        return False
    return True


def get_device_status(preferred: str) -> DeviceStatus:
    gpu_available = torch.cuda.is_available()
    gpu_supported = _supports_current_gpu()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    capability = None
    if gpu_available:
        major, minor = torch.cuda.get_device_capability(0)
        capability = f"sm_{major}{minor}"
    return DeviceStatus(
        requested=preferred,
        resolved=resolve_device(preferred),
        gpu_available=gpu_available,
        gpu_supported=gpu_supported,
        gpu_name=gpu_name,
        gpu_capability=capability,
        arch_list=tuple(sorted(_supported_arch_list())),
    )


def log_device_status(preferred: str) -> DeviceStatus:
    status = get_device_status(preferred)
    logger.info(
        "Device active: %s | requested=%s | gpu=%s | capability=%s | wheel_arches=%s",
        status.resolved,
        status.requested,
        status.gpu_name or "unavailable",
        status.gpu_capability or "unknown",
        ",".join(status.arch_list) or "unknown",
    )
    return status


def resolve_device(preferred: str) -> str:
    value = preferred.strip().lower()
    if value == "cpu":
        return "cpu"
    if value in {"auto", "cuda", "cuda:0"}:
        if _supports_current_gpu():
            return "cuda:0"
        if value == "auto":
            logger.warning("CUDA is unavailable or unsupported; using CPU")
            return "cpu"
        major, minor = (
            torch.cuda.get_device_capability(0)
            if torch.cuda.is_available()
            else (None, None)
        )
        arch = f"sm_{major}{minor}" if major is not None else "unknown"
        supported = ", ".join(sorted(_supported_arch_list())) or "unknown"
        raise RuntimeError(
            f"Requested CUDA explicitly, but the current GPU arch ({arch}) is not supported by this PyTorch build. "
            f"Supported arches: {supported}. Install a torch wheel that includes sm_52 support or use --device auto/cpu."
        )
    if value.startswith("cuda"):
        if _supports_current_gpu():
            return preferred
        major, minor = (
            torch.cuda.get_device_capability(0)
            if torch.cuda.is_available()
            else (None, None)
        )
        arch = f"sm_{major}{minor}" if major is not None else "unknown"
        supported = ", ".join(sorted(_supported_arch_list())) or "unknown"
        raise RuntimeError(
            f"Requested CUDA explicitly, but the current GPU arch ({arch}) is not supported by this PyTorch build. "
            f"Supported arches: {supported}. Install a torch wheel that includes sm_52 support or use --device auto/cpu."
        )
    return "cpu"
