from __future__ import annotations

from pathlib import Path

from config import AppConfig
from detectors.base import Detector
from detectors.onnxruntime_local import OnnxRuntimeLocalDetector


def create_detector(config: AppConfig, project_root: Path) -> Detector:
    return OnnxRuntimeLocalDetector(config=config, project_root=project_root)
