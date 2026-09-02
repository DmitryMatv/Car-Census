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


def _projective_targets(
    source: list[list[float]], matrix: np.ndarray
) -> list[list[float]]:
    homogeneous = np.hstack(
        [np.asarray(source, dtype=np.float64), np.ones((len(source), 1))]
    )
    projected = homogeneous @ matrix.T
    return (projected[:, :2] / projected[:, 2:3]).tolist()


# Genuinely projective mapping (perspective division in x), so a plain
# 4-point affine fit could not satisfy all six pairs.
H_TRUE = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [1e-5, 0.0, 1.0]])
WIDE_SOURCE = [[0, 0], [100, 0], [200, 0], [0, 300], [100, 300], [200, 300]]
WIDE_TARGET = _projective_targets(WIDE_SOURCE, H_TRUE)


def test_accepts_more_than_four_point_pairs() -> None:
    transformer = ViewTransformer(WIDE_SOURCE, WIDE_TARGET)

    assert transformer.transform_point((200, 0)) == pytest.approx(
        WIDE_TARGET[2], abs=1e-5
    )
    assert transformer.transform_point((150, 150)) == pytest.approx(
        _projective_targets([[150, 150]], H_TRUE)[0]
    )


def test_least_squares_pairs_tolerate_field_measurement_noise() -> None:
    noisy = [list(point) for point in WIDE_TARGET]
    noisy[4][1] += 0.05
    noisy[5][0] += 0.05

    transformer = ViewTransformer(WIDE_SOURCE, noisy)

    assert transformer.transform_point((0, 0)) == pytest.approx(WIDE_TARGET[0], abs=0.1)


def test_rejects_least_squares_pair_set_with_blundered_target() -> None:
    blundered = [list(point) for point in WIDE_TARGET]
    blundered[5][0] += 5.0

    with pytest.raises(ValueError, match="Degenerate calibration points"):
        ViewTransformer(WIDE_SOURCE, blundered)


def test_rejects_collinear_sources_with_more_than_four_points() -> None:
    with pytest.raises(ValueError):
        ViewTransformer(
            [[0, 0], [10, 0], [20, 0], [30, 0], [40, 0]],
            [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]],
        )
