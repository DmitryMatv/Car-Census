from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import pytest

from mmr.embeddings import OpenRouterEmbeddingProvider, embedding_cache_key


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self.payload


class _Client:
    calls = 0
    last_url = ""
    last_headers: dict[str, str] = {}
    last_json: dict[str, Any] = {}

    def __init__(self, timeout: float) -> None:
        _ = timeout

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        _ = exc_type, exc_value, traceback

    def post(
        self, url: str, headers: dict[str, str], json: dict[str, Any]
    ) -> _Response:
        type(self).calls += 1
        type(self).last_url = url
        type(self).last_headers = headers
        type(self).last_json = json
        return _Response({"data": [{"embedding": [0.1, 0.2, 0.3]}]})


def test_openrouter_embedding_request_and_cache(tmp_path: Path) -> None:
    _Client.calls = 0
    provider = OpenRouterEmbeddingProvider(
        api_key="secret",
        api_key_env="OPENROUTER_API_KEY",
        model="google/gemini-embedding-2",
        dimensions=3,
        cache_dir=tmp_path,
        timeout=30.0,
        http_client_factory=_Client,
    )

    vector = provider.embed(b"jpeg-bytes")
    cached_vector = provider.embed(b"jpeg-bytes")

    assert vector == [0.1, 0.2, 0.3]
    assert cached_vector == vector
    assert _Client.calls == 1
    assert _Client.last_headers["Authorization"] == "Bearer secret"
    assert _Client.last_url == "https://openrouter.ai/api/v1/embeddings"
    image_url = _Client.last_json["input"][0]["content"][0]["image_url"]["url"]
    assert image_url.startswith("data:image/jpeg;base64,")
    assert _Client.last_json["dimensions"] == 3
    assert (
        tmp_path
        / f"{embedding_cache_key(b'jpeg-bytes', 'google/gemini-embedding-2', 3)}.json"
    ).exists()


def test_openrouter_embedding_rejects_unexpected_dimensions() -> None:
    class WrongDimensionClient(_Client):
        def post(self, url, headers, json) -> _Response:
            _ = url, headers, json
            return _Response({"data": [{"embedding": [0.1]}]})

    provider = OpenRouterEmbeddingProvider(
        api_key="secret",
        api_key_env="OPENROUTER_API_KEY",
        model="google/gemini-embedding-2",
        dimensions=3,
        timeout=30.0,
        http_client_factory=WrongDimensionClient,
    )

    with pytest.raises(ValueError, match="expected 3, got 1"):
        provider.embed(b"jpeg-bytes")
