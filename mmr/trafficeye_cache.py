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


def migrate_legacy_response_cache(cache_dir: Path) -> int:
    """Move root-level response JSON files into the organized responses directory."""
    responses_dir = cache_dir / "responses"
    responses_dir.mkdir(parents=True, exist_ok=True)
    migrated = 0
    for legacy_path in sorted(cache_dir.glob("*.json")):
        destination = responses_dir / legacy_path.name
        if destination.exists():
            continue
        legacy_path.replace(destination)
        migrated += 1
    return migrated


class TrafficEyeCacheClient:
    def __init__(
        self,
        *,
        api_url: str,
        api_key: str,
        timeout: float,
        cache_dir: Path,
        legacy_cache_dir: Path | None = None,
        require_api_key: bool = True,
        http_client_factory: Any = httpx.Client,
    ) -> None:
        self.api_url = api_url
        self.api_key = api_key
        self.timeout = timeout
        self.cache_dir = cache_dir
        self.legacy_cache_dir = legacy_cache_dir
        self.require_api_key = require_api_key
        self.http_client_factory = http_client_factory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_paths(self, cache_key: str) -> list[Path]:
        paths = [self.cache_dir / f"{cache_key}.json"]
        if self.legacy_cache_dir is not None:
            legacy_path = self.legacy_cache_dir / f"{cache_key}.json"
            if legacy_path != paths[0]:
                paths.append(legacy_path)
        return paths

    def load_or_request(
        self,
        *,
        image_bytes: bytes,
        filename: str,
        request_payload: dict[str, Any],
        allow_cache: bool = True,
    ) -> tuple[dict[str, Any], str]:
        cache_key = hash_request(image_bytes, request_payload)
        cache_path = self._cache_paths(cache_key)[0]
        if allow_cache:
            for candidate_path in self._cache_paths(cache_key):
                if not candidate_path.exists():
                    continue
                payload_bytes = candidate_path.read_bytes()
                payload = orjson.loads(payload_bytes)
                if candidate_path != cache_path and not cache_path.exists():
                    cache_path.write_bytes(payload_bytes)
                return payload, cache_key

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
