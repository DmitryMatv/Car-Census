from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from config import AppConfig
from detectors.base import Detector
from detectors.onnxruntime_inference import OnnxRuntimeInferenceRunner
from detectors.onnxruntime_session import load_onnxruntime_session
from detectors.yolo_postprocess import build_yolo_output_parser
from models import Detection

if TYPE_CHECKING:
    import numpy as np


class OnnxRuntimeLocalDetector(Detector):
    def __init__(self, config: AppConfig, project_root: Path) -> None:
        bundle = load_onnxruntime_session(config=config, project_root=project_root)
        parser = build_yolo_output_parser(
            config=config,
            metadata=bundle.session.get_modelmeta().custom_metadata_map,
        )
        self.runner = OnnxRuntimeInferenceRunner(
            session=bundle.session,
            input_name=bundle.input_spec.name,
            input_dtype=bundle.input_spec.dtype,
            input_size=config.analysis.imgsz,
            dynamic_batch=bundle.input_spec.dynamic_batch,
            fixed_batch_size=bundle.input_spec.fixed_batch_size,
            parser=parser,
        )
        self.dynamic_batch = bundle.input_spec.dynamic_batch
        self.fixed_batch_size = bundle.input_spec.fixed_batch_size

    def detect(self, image: np.ndarray) -> list[Detection]:
        return self.runner.detect(image)

    def detect_batch(self, images: Sequence[np.ndarray]) -> list[list[Detection]]:
        return self.runner.detect_batch(images)

    def detection_diagnostics(self) -> dict[str, object]:
        return self.runner.parser.diagnostics.as_dict()
