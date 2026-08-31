"""Selective appearance embeddings (ReID) for the tracking pipeline.

Embeddings are computed for a small subset of detections (per-track cadence
plus rescue candidates) and kept per track in a bounded history. They are
consumed by the world-space rescue layer to confirm or reject identity
takeovers. Embeddings cannot distinguish identical same-model/same-color
cars — the world-space rescue gate covers that blind spot.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np


class AppearanceEmbedder(Protocol):
    """Turns BGR crops into L2-normalized embedding vectors."""

    def embed(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Return an ``(N, D)`` float32 array, one row per crop."""
        ...


class TorchvisionEmbedder:
    """ResNet-18 (ImageNet) feature extractor producing 512-d embeddings.

    Weights are downloaded once on first use and cached by torchvision. The
    model itself is built lazily on the first ``embed`` call, so constructing
    the embedder (e.g. in tests or offline runs that never embed) never
    touches the network. All heavy imports are deferred as well.
    """

    _INPUT_SIZE = 224
    _MEAN = (0.485, 0.456, 0.406)
    _STD = (0.229, 0.224, 0.225)

    def __init__(self, device: str = "auto", batch_size: int = 16) -> None:
        self._device_spec = device
        self._batch_size = max(1, batch_size)
        self._torch: Any = None
        self._device: Any = None
        self._model: Any = None
        self._mean: Any = None
        self._std: Any = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from torch import nn
        from torchvision.models import (  # type: ignore[import-untyped]
            ResNet18_Weights,
            resnet18,
        )

        self._torch = torch
        device = self._device_spec
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self._model = nn.Sequential(*list(backbone.children())[:-2])
        self._model.to(self._device)
        self._model.eval()
        mean = torch.tensor(self._MEAN, dtype=torch.float32)
        std = torch.tensor(self._STD, dtype=torch.float32)
        self._mean = mean.view(3, 1, 1)
        self._std = std.view(3, 1, 1)

    def embed(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        import torch
        from torchvision.transforms import (
            functional as tf,  # type: ignore[import-untyped]
        )

        if not crops:
            return np.empty((0, 512), dtype=np.float32)
        self._ensure_model()
        tensors = []
        for crop in crops:
            image = tf.to_tensor(np.ascontiguousarray(crop[:, :, ::-1]))
            image = tf.resize(image, [self._INPUT_SIZE, self._INPUT_SIZE])
            tensors.append((image - self._mean) / self._std)
        outputs: list[Any] = []
        with torch.no_grad():
            for start in range(0, len(tensors), self._batch_size):
                batch = torch.stack(tensors[start : start + self._batch_size])
                batch = batch.to(self._device)
                features = self._model(batch)
                pooled = features.mean(dim=(2, 3))
                outputs.append(pooled.cpu())
        matrix = torch.cat(outputs).numpy().astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return matrix / norms


class TrackAppearanceMemory:
    """Bounded per-track embedding histories with max-cosine lookup."""

    def __init__(self, history_length: int = 16) -> None:
        self._history_length = max(1, history_length)
        self._by_track: dict[int, deque[np.ndarray]] = {}

    def observe(self, track_id: int, vector: np.ndarray) -> None:
        array = np.asarray(vector, dtype=np.float32).ravel()
        if array.size == 0 or float(np.linalg.norm(array)) == 0.0:
            return
        history = self._by_track.setdefault(
            track_id, deque(maxlen=self._history_length)
        )
        history.append(array)

    def forget(self, track_id: int) -> None:
        self._by_track.pop(track_id, None)

    def known(self, track_id: int) -> bool:
        return bool(self._by_track.get(track_id))

    def similarity(self, track_id: int, vector: np.ndarray) -> float | None:
        """Best cosine similarity against the track's history.

        Returns ``None`` when the track has no stored embeddings, so callers
        can distinguish "no evidence" from "no match".
        """
        history = self._by_track.get(track_id)
        if not history:
            return None
        query = np.asarray(vector, dtype=np.float32).ravel()
        query_norm = float(np.linalg.norm(query))
        if query_norm == 0.0:
            return None
        best = -1.0
        for stored in history:
            stored_norm = float(np.linalg.norm(stored))
            if stored_norm == 0.0:
                continue
            cosine = float(np.dot(query, stored) / (query_norm * stored_norm))
            best = max(best, cosine)
        return best


def build_embedder(config: object) -> AppearanceEmbedder | None:
    """Build the embedder for a ``ReidConfig``-like object; None if disabled.

    Import or model-load failures degrade to ``None`` so the tracking
    pipeline keeps working without appearance evidence.
    """
    enabled = getattr(config, "enabled", False)
    if not enabled:
        return None
    try:
        return TorchvisionEmbedder(
            device=str(getattr(config, "device", "auto")),
            batch_size=int(getattr(config, "batch_size", 16)),
        )
    except Exception:  # pragma: no cover - depends on runtime environment
        return None
