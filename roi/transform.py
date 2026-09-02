from __future__ import annotations

import math
from collections.abc import Sequence

import cv2
import numpy as np

from config import AppConfig, CameraProfile


class ViewTransformer:
    """Maps pixel coordinates onto the road plane via a homography.

    Built from at least 4 corresponding point pairs: pixel coordinates on
    the road surface and their real-world positions in meters.
    """

    def __init__(
        self,
        source_points: Sequence[Sequence[float]],
        target_points: Sequence[Sequence[float]],
    ) -> None:
        source = np.asarray(source_points, dtype=np.float32)
        target = np.asarray(target_points, dtype=np.float32)
        if (
            source.ndim != 2
            or target.ndim != 2
            or source.shape != target.shape
            or source.shape[1] != 2
            or source.shape[0] < 4
        ):
            raise ValueError(
                "ViewTransformer requires >= 4 matching (x, y) point pairs, "
                f"got shapes {source.shape} and {target.shape}"
            )
        try:
            if source.shape[0] == 4:
                self._matrix = cv2.getPerspectiveTransform(source, target)
            else:
                # More than four pairs: solve in the least-squares sense so
                # calibrations may cover several depth bands at once. The
                # reprojection guard below still rejects degenerate solutions.
                matrix, _mask = cv2.findHomography(source, target, method=0)
                if matrix is None:
                    raise cv2.error("findHomography returned no solution")
                self._matrix = matrix
        except cv2.error as exc:
            raise ValueError(
                "Failed to compute homography; source points are likely "
                "degenerate (collinear or duplicated)"
            ) from exc
        if not np.all(np.isfinite(self._matrix)):
            raise ValueError(
                "Computed homography is not finite; check calibration points"
            )
        reprojected = cv2.perspectiveTransform(
            source.reshape(-1, 1, 2), self._matrix
        ).reshape(-1, 2)
        if source.shape[0] == 4:
            # Exact solve: residuals are pure floating-point noise.
            tolerance = np.full(
                source.shape[0],
                1e-3 * max(1.0, float(np.abs(target).max())),
            )
        else:
            # Least-squares over field-measured anchors: paint placement
            # tolerance and lens distortion leave honest per-point misfit
            # (observed 0.1-0.25 m across a ~150 m depth range). Allow 20 cm
            # or 1% of the point's depth, whichever is larger; degenerate or
            # grossly mislabeled configurations still fail by orders of
            # magnitude.
            tolerance = np.maximum(0.20, 0.01 * np.abs(target).max(axis=1))
        if not np.all(np.abs(reprojected - target) <= tolerance[:, None]):
            raise ValueError(
                "Degenerate calibration points: homography does not "
                "reproject the given point pairs"
            )

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if array.size == 0:
            return array
        transformed = cv2.perspectiveTransform(array.reshape(-1, 1, 2), self._matrix)
        result = transformed.reshape(-1, 2)
        return np.where(np.isfinite(result), result, np.inf).astype(np.float32)

    def transform_point(self, point: tuple[float, float]) -> tuple[float, float]:
        x, y = self.transform_points(np.array([point], dtype=np.float32))[0]
        return float(x), float(y)

    def distance_between(
        self,
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        first_world = self.transform_point(first)
        second_world = self.transform_point(second)
        if not all(math.isfinite(value) for value in (*first_world, *second_world)):
            return math.inf
        return math.hypot(
            first_world[0] - second_world[0],
            first_world[1] - second_world[1],
        )


def build_view_transformer(
    config: AppConfig, profile: CameraProfile
) -> ViewTransformer | None:
    if not config.tracker.world_reassociation_enabled:
        return None
    if profile.homography is None:
        return None
    return ViewTransformer(
        source_points=profile.homography.source_points,
        target_points=profile.homography.target_points,
    )
