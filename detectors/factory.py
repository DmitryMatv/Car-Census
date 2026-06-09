from __future__ import annotations

from pathlib import Path

from config import AppConfig
from detectors.base import Detector
from detectors.rfdetr_local import RfDetrMediumDetector


def create_detector(config: AppConfig, project_root: Path) -> Detector:
    return RfDetrMediumDetector(config=config, project_root=project_root)
