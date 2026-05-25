from detectors.onnxruntime_local import OnnxRuntimeLocalDetector


def test_detector_delegates_detection_to_runner() -> None:
    class FakeRunner:
        parser = type("Parser", (), {"diagnostics": None})()

        def __init__(self) -> None:
            self.detect_called = False
            self.detect_batch_called = False

        def detect(self, image):
            self.detect_called = True
            return ["single"]

        def detect_batch(self, images):
            self.detect_batch_called = True
            return [["batch"]]

    detector = OnnxRuntimeLocalDetector.__new__(OnnxRuntimeLocalDetector)
    runner = FakeRunner()
    detector.runner = runner

    assert detector.detect(object()) == ["single"]
    assert detector.detect_batch([object()]) == [["batch"]]
    assert runner.detect_called is True
    assert runner.detect_batch_called is True


def test_detection_diagnostics_delegates_to_runner_parser() -> None:
    class FakeDiagnostics:
        def as_dict(self) -> dict[str, object]:
            return {"counts": {"raw_candidate_rows": 1}, "confidence_values": []}

    class FakeParser:
        diagnostics = FakeDiagnostics()

    class FakeRunner:
        parser = FakeParser()

    detector = OnnxRuntimeLocalDetector.__new__(OnnxRuntimeLocalDetector)
    detector.runner = FakeRunner()

    assert detector.detection_diagnostics() == {
        "counts": {"raw_candidate_rows": 1},
        "confidence_values": [],
    }
