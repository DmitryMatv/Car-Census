from __future__ import annotations

import numpy as np


def normalized_identity(
    make: str | None, model: str | None, generation: str | None
) -> tuple[str, str, str]:
    return (
        "" if make is None else make.strip().casefold(),
        "" if model is None else model.strip().casefold(),
        "" if generation is None else generation.strip().casefold(),
    )


def cosine_distance(left: list[float], right: list[float]) -> float:
    left_array = np.asarray(left, dtype=np.float32)
    right_array = np.asarray(right, dtype=np.float32)
    if left_array.shape != right_array.shape:
        return float("inf")
    left_norm = float(np.linalg.norm(left_array))
    right_norm = float(np.linalg.norm(right_array))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0 if np.array_equal(left_array, right_array) else float("inf")
    similarity = float(np.dot(left_array, right_array) / (left_norm * right_norm))
    return max(0.0, 1.0 - similarity)


def blockwise_cosine_distances(
    embeddings: np.ndarray, start: int, end: int
) -> np.ndarray:
    """Return cosine distances from rows [start, end) to every row, clamped to >= 0.

    Two zero vectors are 0.0 apart; exactly one zero vector is infinitely far.
    """
    norms = np.linalg.norm(embeddings, axis=1)
    nonzero = norms > 0.0
    normalized = np.zeros_like(embeddings)
    normalized[nonzero] = embeddings[nonzero] / norms[nonzero, None]
    distances = np.maximum(0.0, 1.0 - normalized[start:end] @ normalized.T)
    left_zero = ~nonzero[start:end, None]
    right_zero = ~nonzero[None, :]
    distances[left_zero & right_zero] = 0.0
    distances[left_zero ^ right_zero] = np.inf
    return distances
