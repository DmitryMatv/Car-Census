import numpy as np
import pytest
import supervision as sv

from pipeline.detections import (
    class_names,
    clip_detections_to_shape,
    clone_detections,
    detection_bboxes,
    empty_detections,
    map_detections_to_global,
)


def _detections() -> sv.Detections:
    return sv.Detections(
        xyxy=np.array([[1, 2, 11, 22], [20, 20, 20, 25]], dtype=np.float32),
        confidence=np.array([0.9, 0.8], dtype=np.float32),
        class_id=np.array([2, 7], dtype=np.int32),
        data={"class_name": np.array(["car", "truck"], dtype=object)},
    )


def test_empty_detections_has_expected_shapes() -> None:
    detections = empty_detections()

    assert detections.xyxy.shape == (0, 4)
    assert detections.confidence is not None
    assert detections.confidence.shape == (0,)
    assert detections.class_id is not None
    assert detections.class_id.shape == (0,)
    assert detections.data["class_name"].tolist() == []


def test_detection_bboxes_converts_xyxy_rows() -> None:
    bboxes = detection_bboxes(_detections())

    assert bboxes[0].model_dump() == {"x1": 1.0, "y1": 2.0, "x2": 11.0, "y2": 22.0}


def test_map_detections_to_global_offsets_without_mutating_input() -> None:
    detections = _detections()

    mapped = map_detections_to_global(detections, (100, 200))

    np.testing.assert_allclose(detections.xyxy[0], [1, 2, 11, 22])
    np.testing.assert_allclose(mapped.xyxy[0], [101, 202, 111, 222])
    assert mapped.data["class_name"].tolist() == ["car", "truck"]


def test_clone_detections_deep_copies_nested_list_data() -> None:
    detections = sv.Detections(
        xyxy=np.array([[1, 2, 11, 22]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([2], dtype=np.int32),
        data={"nested": [[{"tag": "original"}]]},
    )

    cloned = clone_detections(detections)
    cloned.data["nested"][0][0]["tag"] = "changed"

    assert detections.data["nested"][0][0]["tag"] == "original"
    assert cloned.data["nested"][0][0]["tag"] == "changed"


def test_clip_detections_to_shape_clips_and_drops_invalid_rows() -> None:
    detections = sv.Detections(
        xyxy=np.array([[-5, -6, 50, 40], [20, 20, 20, 25]], dtype=np.float32),
        confidence=np.array([0.9, 0.8], dtype=np.float32),
        class_id=np.array([2, 7], dtype=np.int32),
        data={"class_name": np.array(["car", "truck"], dtype=object)},
    )

    clipped = clip_detections_to_shape(detections, (24, 32, 3))

    np.testing.assert_allclose(clipped.xyxy, [[0, 0, 31, 23]])
    assert clipped.confidence is not None
    assert clipped.confidence.tolist() == pytest.approx([0.9])
    assert clipped.class_id is not None
    assert clipped.class_id.tolist() == [2]
    assert clipped.data["class_name"].tolist() == ["car"]


def test_class_names_handles_missing_data() -> None:
    detections = sv.Detections(
        xyxy=np.array([[1, 2, 11, 22]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([2], dtype=np.int32),
    )

    assert class_names(detections) == [""]
