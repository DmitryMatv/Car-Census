from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class PreprocessedImage:
    tensor: np.ndarray
    scale: float
    pad_x: float
    pad_y: float
    image_shape: tuple[int, ...]


def _letterbox(
    image: np.ndarray, size: int
) -> tuple[np.ndarray, float, tuple[float, float]]:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    pad_x = (size - new_width) / 2
    pad_y = (size - new_height) / 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top = round(pad_y - 0.1)
    left = round(pad_x - 0.1)
    canvas[top : top + new_height, left : left + new_width] = resized
    return canvas, scale, (left, top)


def preprocess_image(
    image: np.ndarray, input_size: int, input_dtype: np.dtype[Any]
) -> PreprocessedImage:
    frame, scale, (pad_x, pad_y) = _letterbox(image, input_size)
    tensor = frame[:, :, ::-1].transpose(2, 0, 1).astype(input_dtype)
    tensor /= np.asarray(255.0, dtype=input_dtype)
    return PreprocessedImage(
        tensor=tensor,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        image_shape=image.shape,
    )
