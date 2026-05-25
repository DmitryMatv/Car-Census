from __future__ import annotations

from pathlib import Path

from config import AppConfig
from detectors.base import Detector
from detectors.rfdetr_local import RfDetrSmallDetector


def create_detector(config: AppConfig, project_root: Path) -> Detector:
    return RfDetrSmallDetector(config=config, project_root=project_root)
