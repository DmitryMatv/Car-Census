from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import httpx
import orjson


def hash_request(image_bytes: bytes, request_payload: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(image_bytes)
    digest.update(orjson.dumps(request_payload))
    return digest.hexdigest()


class TrafficEyeCacheClient:
    def __init__(
        self,
        *,
        api_url: str,
        api_key: str,
        timeout: float,
        cache_dir: Path,
        require_api_key: bool = True,
        http_client_factory: Any = httpx.Client,
    ) -> None:
        self.api_url = api_url
        self.api_key = api_key
        self.timeout = timeout
        self.cache_dir = cache_dir
        self.require_api_key = require_api_key
        self.http_client_factory = http_client_factory

    def load_or_request(
        self,
        *,
        image_bytes: bytes,
        filename: str,
        request_payload: dict[str, Any],
        allow_cache: bool = True,
    ) -> tuple[dict[str, Any], str]:
        cache_key = hash_request(image_bytes, request_payload)
        cache_path = self.cache_dir / f"{cache_key}.json"
        if allow_cache and cache_path.exists():
            return orjson.loads(cache_path.read_bytes()), cache_key

        if self.require_api_key and not self.api_key:
            raise RuntimeError(
                "Missing TrafficEye API key. Set the configured API key environment "
                "variable before requesting an uncached classification."
            )

        with self.http_client_factory(timeout=self.timeout) as client:
            response = client.post(
                self.api_url,
                headers={"apikey": self.api_key},
                files={
                    "file": (filename, image_bytes, "image/jpeg"),
                    "request": (
                        None,
                        orjson.dumps(request_payload).decode("utf-8"),
                    ),
                },
            )
            response.raise_for_status()
            payload = response.json()
            cache_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
            return payload, cache_key
