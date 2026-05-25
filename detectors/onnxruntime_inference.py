from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from detectors.yolo_postprocess import YoloOutputParser
from detectors.yolo_preprocess import PreprocessedImage, preprocess_image
from models import Detection


class OnnxRuntimeInferenceRunner:
    def __init__(
        self,
        *,
        session: Any,
        input_name: str,
        input_dtype: np.dtype,
        input_size: int,
        dynamic_batch: bool,
        fixed_batch_size: int | None,
        parser: YoloOutputParser,
    ) -> None:
        self.session = session
        self.input_name = input_name
        self.input_dtype = input_dtype
        self.input_size = input_size
        self.dynamic_batch = dynamic_batch
        self.fixed_batch_size = fixed_batch_size
        self.parser = parser

    def detect(self, image: np.ndarray) -> list[Detection]:
        preprocessed = self.preprocess(image)
        outputs = self.session.run(
            None,
            {self.input_name: np.expand_dims(preprocessed.tensor, axis=0)},
        )
        return self.parse_single_output(outputs[0], preprocessed)

    def detect_batch(self, images: Sequence[np.ndarray]) -> list[list[Detection]]:
        if not images:
            return []
        if len(images) == 1 or (self.fixed_batch_size == 1 and not self.dynamic_batch):
            return [self.detect(image) for image in images]
        preprocessed = [self.preprocess(image) for image in images]
        tensors = [item.tensor for item in preprocessed]
        requested_count = len(tensors)
        run_count = requested_count
        if self.fixed_batch_size is not None:
            run_count = self.fixed_batch_size
            if requested_count > run_count:
                results: list[list[Detection]] = []
                for start in range(0, requested_count, run_count):
                    results.extend(self.detect_batch(images[start : start + run_count]))
                return results
            if requested_count < run_count:
                pad_tensor = np.full_like(tensors[0], 114.0 / 255.0)
                tensors.extend([pad_tensor] * (run_count - requested_count))

        batch_tensor = np.stack(tensors, axis=0)
        outputs = self.session.run(None, {self.input_name: batch_tensor})
        output = np.asarray(outputs[0])
        if output.ndim < 3 or output.shape[0] < requested_count:
            raise ValueError(f"Unsupported batched ONNX output shape: {output.shape}")

        detections_by_image: list[list[Detection]] = []
        for index, item in enumerate(preprocessed):
            detections_by_image.append(self.parse_single_output(output[index], item))
        return detections_by_image

    def preprocess(self, image: np.ndarray) -> PreprocessedImage:
        return preprocess_image(
            image=image,
            input_size=self.input_size,
            input_dtype=self.input_dtype,
        )

    def parse_single_output(
        self, output: object, preprocessed: PreprocessedImage
    ) -> list[Detection]:
        return self.parser.parse_single(
            output,
            preprocessed.image_shape,
            preprocessed.scale,
            preprocessed.pad_x,
            preprocessed.pad_y,
        )
