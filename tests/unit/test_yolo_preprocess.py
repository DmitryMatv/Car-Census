import numpy as np

from detectors.yolo_preprocess import _letterbox, preprocess_image


def test_letterbox_preserves_aspect_ratio() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    output, scale, padding = _letterbox(image, 640)

    assert output.shape == (640, 640, 3)
    assert scale == 3.2
    assert padding == (0, 160)


def test_preprocess_uses_configured_model_input_dtype() -> None:
    preprocessed = preprocess_image(
        image=np.full((4, 8, 3), 255, dtype=np.uint8),
        input_size=8,
        input_dtype=np.dtype(np.float16),
    )

    assert preprocessed.tensor.dtype == np.float16
    assert preprocessed.tensor.shape == (3, 8, 8)
    assert preprocessed.scale == 1.0
    assert preprocessed.pad_x == 0.0
    assert preprocessed.pad_y == 2.0
    assert preprocessed.image_shape == (4, 8, 3)
