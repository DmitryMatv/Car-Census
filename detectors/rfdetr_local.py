from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import supervision as sv

from config import AppConfig
from detectors.base import Detector
from pipeline.analysis_diagnostics import CONFIDENCE_BINS, HistogramAccumulator
from pipeline.detections import class_names, clip_detections_to_shape, clone_detections

logger = logging.getLogger(__name__)


def _load_rfdetr_small() -> type[Any]:
    try:
        from rfdetr import RFDETRSmall
    except ImportError as exc:
        raise RuntimeError(
            "rfdetr is required for RF-DETR-S detection. "
            "Install the project with `pip install -e .`."
        ) from exc
    return RFDETRSmall


def _coerce_class_names(raw: object) -> dict[int, str]:
    if isinstance(raw, Mapping):
        return {int(class_id): str(name).lower() for class_id, name in raw.items()}
    if isinstance(raw, Sequence) and not isinstance(raw, str):
        return {index: str(name).lower() for index, name in enumerate(raw)}
    return {}


def _cuda_is_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def _resolved_inference_dtype(configured_dtype: str, device: object) -> str:
    if configured_dtype != "auto":
        return configured_dtype
    if getattr(device, "type", None) == "cuda" or str(device).startswith("cuda"):
        return "float16"
    if device is None:
        if _cuda_is_available():
            logger.warning(
                "RF-DETR model device was unavailable while resolving auto "
                "inference dtype; using float16 because CUDA is available."
            )
            return "float16"
        logger.warning(
            "RF-DETR model device was unavailable while resolving auto "
            "inference dtype; using float32 because CUDA is not available."
        )
    return "float32"


class RfDetrSmallDetector(Detector):
    def __init__(self, config: AppConfig, project_root: Path) -> None:
        self.config = config
        self.input_size = config.detector.input_size
        self.allowed_class_names = {
            name.lower() for name in config.detector.allowed_class_names
        }
        self._counts: Counter[str] = Counter()
        self._confidence_histogram = HistogramAccumulator(CONFIDENCE_BINS)

        weights = config.detector.pretrain_weights
        model_class = _load_rfdetr_small()
        model_kwargs: dict[str, object] = {"resolution": self.input_size}
        if config.detector.device != "auto":
            model_kwargs["device"] = config.detector.device
        if weights is None:
            self.model = model_class(**model_kwargs)
        else:
            weights_path = Path(weights)
            if not weights_path.is_absolute():
                weights_path = project_root / weights_path
            if not weights_path.exists():
                raise FileNotFoundError(
                    f"RF-DETR-S checkpoint not found: {weights_path}"
                )
            self.model = model_class(pretrain_weights=str(weights_path), **model_kwargs)
        self.class_names = _coerce_class_names(getattr(self.model, "class_names", {}))
        self._inference_dtype = _resolved_inference_dtype(
            config.detector.inference_dtype,
            getattr(getattr(self.model, "model", None), "device", None),
        )
        if config.detector.optimize_for_inference:
            optimize_kwargs: dict[str, object] = {
                "compile": config.detector.compile_for_inference,
                "dtype": self._inference_dtype,
            }
            if config.detector.compile_for_inference:
                optimize_kwargs["batch_size"] = (
                    config.analysis.detector_batch_size or config.analysis.batch_size
                )
            self.model.optimize_for_inference(**optimize_kwargs)

    def detect(self, image: np.ndarray) -> sv.Detections:
        return self.detect_batch([image])[0]

    def detect_batch(self, images: Sequence[np.ndarray]) -> list[sv.Detections]:
        if not images:
            return []
        rgb_images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
        predictions = self.model.predict(
            rgb_images,
            threshold=self.config.detector.confidence,
            shape=(self.input_size, self.input_size),
            include_source_image=self.config.detector.include_source_image,
        )
        detections_by_image = self._normalize_prediction_result(predictions)
        if len(detections_by_image) != len(images):
            raise RuntimeError(
                "RF-DETR-S returned "
                f"{len(detections_by_image)} detection sets for {len(images)} images"
            )
        return [
            self._filter_detections(detections, image.shape)
            for detections, image in zip(detections_by_image, images, strict=True)
        ]

    def detection_diagnostics(self) -> dict[str, object]:
        return {
            "counts": dict(self._counts),
            "confidence_histogram": self._confidence_histogram.payload(),
            "model": "rfdetr-small",
            "input_size": self.input_size,
            "runtime": "rfdetr",
            "optimized_for_inference": self.config.detector.optimize_for_inference,
            "inference_dtype": self._inference_dtype,
            "compiled_for_inference": self.config.detector.compile_for_inference,
        }

    @staticmethod
    def _normalize_prediction_result(predictions: object) -> list[sv.Detections]:
        if isinstance(predictions, sv.Detections):
            return [predictions]
        if isinstance(predictions, Sequence):
            return [
                prediction
                for prediction in predictions
                if isinstance(prediction, sv.Detections)
            ]
        raise TypeError(
            "Unsupported RF-DETR-S prediction result type: "
            f"{type(predictions).__name__}"
        )

    def _filter_detections(
        self, detections: sv.Detections, image_shape: tuple[int, ...]
    ) -> sv.Detections:
        detections = self._ensure_class_names(detections)
        self._counts["raw_candidate_rows"] += len(detections)
        self._counts["detections_after_confidence_filtering"] += len(detections)

        if self.allowed_class_names:
            keep = np.array(
                [name in self.allowed_class_names for name in class_names(detections)],
                dtype=bool,
            )
            detections = cast(sv.Detections, detections[keep])

        detections = clip_detections_to_shape(detections, image_shape)
        self._counts["detections_after_class_filtering"] += len(detections)
        if detections.confidence is not None:
            self._confidence_histogram.extend(
                float(confidence) for confidence in detections.confidence.tolist()
            )
        return detections

    def _ensure_class_names(self, detections: sv.Detections) -> sv.Detections:
        names = class_names(detections)
        if names and all(names):
            return detections
        updated = clone_detections(detections)
        resolved_names: list[str] = []
        for index in range(len(detections)):
            if index < len(names) and names[index]:
                resolved_names.append(names[index])
                continue
            class_id = (
                int(detections.class_id[index])
                if detections.class_id is not None
                else -1
            )
            resolved_names.append(self.class_names.get(class_id, ""))
        updated.data["class_name"] = np.array(resolved_names, dtype=object)
        return updated
