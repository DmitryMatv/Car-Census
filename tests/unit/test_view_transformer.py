import numpy as np
import pytest

from roi.transform import ViewTransformer


SOURCE = [[0, 0], [400, 0], [400, 300], [0, 300]]
TARGET = [[0, 0], [40, 0], [40, 30], [0, 30]]


def test_transform_point_maps_pixels_to_world_scale() -> None:
    transformer = ViewTransformer(SOURCE, TARGET)

    assert transformer.transform_point((200, 150)) == pytest.approx((20.0, 15.0))
    assert transformer.transform_point((0, 0)) == pytest.approx((0.0, 0.0))
    assert transformer.transform_point((400, 300)) == pytest.approx((40.0, 30.0))


def test_transform_points_handles_batches_and_empty_input() -> None:
    transformer = ViewTransformer(SOURCE, TARGET)

    points = transformer.transform_points(
        np.array([[100, 100], [200, 150], [300, 200]], dtype=np.float64)
    )
    assert points.shape == (3, 2)
    assert points[1].tolist() == pytest.approx([20.0, 15.0])

    empty = transformer.transform_points(np.empty((0, 2)))
    assert empty.shape == (0, 2)


def test_distance_between_measures_world_space_displacement() -> None:
    transformer = ViewTransformer(SOURCE, TARGET)

    distance = transformer.distance_between((100, 100), (300, 100))

    assert distance == pytest.approx(20.0)


def test_rejects_mismatched_or_undersized_point_sets() -> None:
    with pytest.raises(ValueError):
        ViewTransformer(SOURCE, TARGET[:3])
    with pytest.raises(ValueError):
        ViewTransformer(SOURCE[:3], TARGET[:3])


def test_rejects_degenerate_collinear_sources() -> None:
    with pytest.raises(ValueError):
        ViewTransformer(
            [[0, 0], [10, 0], [20, 0], [30, 0]],
            [[0, 0], [1, 0], [2, 0], [3, 0]],
        )
