import numpy as np

from detectors.onnxruntime_inference import OnnxRuntimeInferenceRunner
from detectors.yolo_preprocess import PreprocessedImage
from models import Detection


class EmptyParser:
    def parse_single(self, *args) -> list[Detection]:
        return []


def test_detect_calls_session_with_expanded_single_image_tensor() -> None:
    captured_inputs: list[np.ndarray] = []

    class FakeSession:
        def run(
            self, output_names: object, input_feed: dict[str, np.ndarray]
        ) -> list[np.ndarray]:
            captured_inputs.append(input_feed["images"])
            return [np.zeros((1, 1, 6), dtype=np.float32)]

    runner = OnnxRuntimeInferenceRunner(
        session=FakeSession(),
        input_name="images",
        input_dtype=np.dtype(np.float32),
        input_size=8,
        dynamic_batch=True,
        fixed_batch_size=None,
        parser=EmptyParser(),
    )

    assert runner.detect(np.zeros((4, 4, 3), dtype=np.uint8)) == []
    assert captured_inputs[0].shape == (1, 3, 8, 8)


def test_detect_batch_falls_back_for_fixed_batch_one_model() -> None:
    class FakeRunner(OnnxRuntimeInferenceRunner):
        calls: list[np.ndarray]

        def detect(self, image: np.ndarray) -> list[Detection]:
            self.calls.append(image)
            return []

    runner = FakeRunner(
        session=object(),
        input_name="images",
        input_dtype=np.dtype(np.float32),
        input_size=8,
        dynamic_batch=False,
        fixed_batch_size=1,
        parser=EmptyParser(),
    )
    runner.calls = []
    images = [
        np.zeros((10, 10, 3), dtype=np.uint8),
        np.zeros((20, 20, 3), dtype=np.uint8),
    ]

    assert runner.detect_batch(images) == [[], []]
    assert runner.calls[0] is images[0]
    assert runner.calls[1] is images[1]


def test_detect_batch_pads_fixed_batch_with_letterbox_background() -> None:
    captured_inputs: list[np.ndarray] = []

    class FakeSession:
        def run(
            self, output_names: object, input_feed: dict[str, np.ndarray]
        ) -> list[np.ndarray]:
            captured_inputs.append(input_feed["images"])
            return [np.zeros((4, 1, 6), dtype=np.float32)]

    tensors = [
        np.full((3, 8, 8), 0.1, dtype=np.float32),
        np.full((3, 8, 8), 0.2, dtype=np.float32),
    ]

    class FakeRunner(OnnxRuntimeInferenceRunner):
        captured_preprocesses: list[np.ndarray]

        def preprocess(self, image: np.ndarray) -> PreprocessedImage:
            index = len(self.captured_preprocesses)
            self.captured_preprocesses.append(image)
            return PreprocessedImage(tensors[index], 1.0, 0, 0, image.shape)

        def parse_single_output(
            self, output: object, preprocessed: PreprocessedImage
        ) -> list[Detection]:
            return []

    runner = FakeRunner(
        session=FakeSession(),
        input_name="images",
        input_dtype=np.dtype(np.float32),
        input_size=8,
        dynamic_batch=False,
        fixed_batch_size=4,
        parser=EmptyParser(),
    )
    runner.captured_preprocesses = []
    images = [
        np.zeros((10, 10, 3), dtype=np.uint8),
        np.zeros((20, 20, 3), dtype=np.uint8),
    ]

    assert runner.detect_batch(images) == [[], []]
    assert len(runner.captured_preprocesses) == 2
    assert captured_inputs[0].shape == (4, 3, 8, 8)
    np.testing.assert_allclose(captured_inputs[0][0], tensors[0])
    np.testing.assert_allclose(captured_inputs[0][1], tensors[1])
    np.testing.assert_allclose(captured_inputs[0][2:], 114.0 / 255.0)
