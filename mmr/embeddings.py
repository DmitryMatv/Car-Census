from __future__ import annotations

import base64
import hashlib
import math
import os
from pathlib import Path
from typing import Any, Protocol

import httpx
import orjson


OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
DEFAULT_EMBEDDING_MODEL = "google/gemini-embedding-2"
DEFAULT_EMBEDDING_DIMENSIONS = 768


class EmbeddingUnavailableError(RuntimeError):
    """Raised when an image embedding cannot be obtained."""


class ImageEmbeddingProvider(Protocol):
    model: str
    dimensions: int

    def embed(self, image_bytes: bytes) -> list[float]:
        """Return an embedding for one local JPEG image."""


def embedding_cache_key(
    image_bytes: bytes, model: str, dimensions: int
) -> str:
    image_sha = hashlib.sha256(image_bytes).hexdigest()
    return f"{image_sha}-{hashlib.sha256(model.encode('utf-8')).hexdigest()[:16]}-{dimensions}"


class OpenRouterEmbeddingProvider:
    """OpenRouter-backed image embeddings with a durable local response cache."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_key_env: str = "OPENROUTER_API_KEY",
        model: str = DEFAULT_EMBEDDING_MODEL,
        dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
        cache_dir: Path | None = None,
        timeout: float = 30.0,
        http_client_factory: Any = httpx.Client,
    ) -> None:
        if dimensions <= 0:
            raise ValueError("embedding dimensions must be positive")
        self.api_key = api_key if api_key is not None else os.getenv(api_key_env, "")
        self.model = model
        self.dimensions = dimensions
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.http_client_factory = http_client_factory
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, image_bytes: bytes) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{embedding_cache_key(image_bytes, self.model, self.dimensions)}.json"

    def _load_cached(self, image_bytes: bytes) -> list[float] | None:
        path = self._cache_path(image_bytes)
        if path is None or not path.exists():
            return None
        try:
            payload = orjson.loads(path.read_bytes())
            vector = _parse_embedding(payload, self.dimensions)
        except (OSError, TypeError, ValueError):
            return None
        return vector

    def _write_cached(self, image_bytes: bytes, vector: list[float]) -> None:
        path = self._cache_path(image_bytes)
        if path is None:
            return
        temporary_path = path.with_suffix(".tmp")
        temporary_path.write_bytes(
            orjson.dumps(
                {
                    "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                    "model": self.model,
                    "dimensions": self.dimensions,
                    "data": [{"embedding": vector}],
                },
                option=orjson.OPT_INDENT_2,
            )
        )
        temporary_path.replace(path)

    def embed(self, image_bytes: bytes) -> list[float]:
        cached = self._load_cached(image_bytes)
        if cached is not None:
            return cached
        if not self.api_key:
            raise EmbeddingUnavailableError(
                "Missing OpenRouter API key. Set OPENROUTER_API_KEY before requesting embeddings."
            )

        encoded = base64.b64encode(image_bytes).decode("ascii")
        payload = {
            "model": self.model,
            "input": [
                {
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded}"
                            },
                        }
                    ]
                }
            ],
            "encoding_format": "float",
            "dimensions": self.dimensions,
        }
        try:
            with self.http_client_factory(timeout=self.timeout) as client:
                response = client.post(
                    OPENROUTER_EMBEDDINGS_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                response_payload = response.json()
        except EmbeddingUnavailableError:
            raise
        except Exception as exc:
            raise EmbeddingUnavailableError(
                f"OpenRouter embedding request failed: {exc}"
            ) from exc
        vector = _parse_embedding(response_payload, self.dimensions)
        self._write_cached(image_bytes, vector)
        return vector


def _parse_embedding(payload: Any, dimensions: int) -> list[float]:
    if not isinstance(payload, dict):
        raise ValueError("OpenRouter embedding response must be an object")
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        raise ValueError("OpenRouter embedding response has no data")
    first = data[0]
    if not isinstance(first, dict) or not isinstance(first.get("embedding"), list):
        raise ValueError("OpenRouter embedding response has no embedding vector")
    raw_vector = first["embedding"]
    if len(raw_vector) != dimensions:
        raise ValueError(
            f"Unexpected embedding dimensions: expected {dimensions}, got {len(raw_vector)}"
        )
    vector: list[float] = []
    for value in raw_vector:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Embedding vector contains a non-numeric value")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("Embedding vector contains a non-finite value")
        vector.append(numeric)
    return vector


OpenRouterClient = OpenRouterEmbeddingProvider
