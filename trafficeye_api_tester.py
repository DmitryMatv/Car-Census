#!/usr/bin/env python3
"""Send a car image to TrafficEye's make/model recognition API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError

API_URL = "https://trafficeye.ai/recognition"
API_KEY_ENV = "TRAFFICEYE_API_KEY"
DEFAULT_IMAGE = Path(__file__).resolve().parent / "test.png"
DEFAULT_ENV_FILE = Path(__file__).resolve().parent / ".env"
SUPPORTED_FORMATS = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test TrafficEye make/model recognition with a car image."
    )
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=DEFAULT_IMAGE,
        help=f"car image to upload (default: {DEFAULT_IMAGE})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=45.0,
        help="request timeout in seconds (default: 45)",
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    return args


def inspect_image(image_path: Path) -> tuple[bytes, int, int, str]:
    if not image_path.is_file():
        raise RuntimeError(f"image file does not exist: {image_path}")

    try:
        image_bytes = image_path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"cannot read image: {image_path}: {exc}") from exc

    try:
        with Image.open(image_path) as image:
            width, height = image.size
            image_format = image.format
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise RuntimeError(f"invalid image: {image_path}: {exc}") from exc

    mime_type = SUPPORTED_FORMATS.get(image_format or "")
    if mime_type is None:
        detected = image_format or "unknown"
        raise RuntimeError(
            f"unsupported image format {detected!r}; use a PNG or JPEG image"
        )
    if width <= 0 or height <= 0:
        raise RuntimeError(f"invalid image dimensions: {width}x{height}")

    return image_bytes, width, height, mime_type


def build_request(width: int, height: int) -> dict[str, Any]:
    return {
        "tasks": ["MMR"],
        "mmrPreference": "BOX",
        "combinations": [
            {
                "roadUsers": [
                    {
                        "box": {
                            "position": {
                                "topLeftCol": 0,
                                "topLeftRow": 0,
                                "bottomRightCol": width,
                                "bottomRightRow": height,
                            }
                        }
                    }
                ]
            }
        ],
    }


def recognition_data(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data")
    return data if isinstance(data, dict) else payload


def first_mmr_result(payload: dict[str, Any]) -> dict[str, Any] | None:
    combinations = recognition_data(payload).get("combinations")
    if not isinstance(combinations, list):
        return None

    for combination in combinations:
        if not isinstance(combination, dict):
            continue
        road_users = combination.get("roadUsers")
        if not isinstance(road_users, list):
            continue
        for road_user in road_users:
            if not isinstance(road_user, dict):
                continue
            mmr = road_user.get("mmr")
            if isinstance(mmr, dict):
                return mmr
    return None


def label_and_score(value: Any) -> tuple[str | None, float | None]:
    if isinstance(value, str):
        return value, None
    if not isinstance(value, dict):
        return None, None

    label_value = value.get("value") or value.get("name") or value.get("label")
    label = label_value if isinstance(label_value, str) else None
    score_value = value.get("score")
    score = float(score_value) if isinstance(score_value, (int, float)) else None
    return label, score


def print_summary(payload: dict[str, Any]) -> None:
    print("Recognition summary:")
    mmr = first_mmr_result(payload)
    if mmr is None:
        print("  No MMR result was returned.")
    else:
        for field in ("make", "model", "generation", "variation", "color"):
            label, score = label_and_score(mmr.get(field))
            if label is None:
                continue
            score_text = f" (confidence: {score:.3f})" if score is not None else ""
            print(f"  {field.capitalize()}: {label}{score_text}")

    data = recognition_data(payload)
    for field, title in (("cost", "Cost"), ("credit", "Remaining credit")):
        value = payload.get(field, data.get(field))
        if value is not None:
            print(f"  {title}: {value}")


def print_request_diagnostics(
    *,
    image_path: Path,
    image_bytes: bytes,
    width: int,
    height: int,
    mime_type: str,
    timeout: float,
    request_payload: dict[str, Any],
) -> None:
    print("Request diagnostics:")
    print(f"  Timestamp (UTC): {datetime.now(UTC).isoformat()}")
    print(f"  Method: POST")
    print(f"  URL: {API_URL}")
    print(f"  API key header: sent (value redacted)")
    print(f"  Timeout: {timeout:g} seconds")
    print(f"  Image path: {image_path.resolve()}")
    print(f"  Image MIME type: {mime_type}")
    print(f"  Image dimensions: {width}x{height}")
    print(f"  Image size: {len(image_bytes)} bytes")
    print(f"  Image SHA-256: {hashlib.sha256(image_bytes).hexdigest()}")
    print("  Request JSON:")
    for line in json.dumps(request_payload, indent=2, ensure_ascii=False).splitlines():
        print(f"    {line}")


def print_response_metadata(response: httpx.Response, elapsed: float) -> None:
    print("\nResponse diagnostics:")
    print(f"  Final URL: {response.url}")
    print(f"  HTTP version: {response.http_version}")
    print(f"  HTTP status: {response.status_code} {response.reason_phrase}")
    print(f"  Elapsed time: {elapsed:.2f} seconds")
    print(f"  Response body size: {len(response.content)} bytes")
    print(f"  Response body SHA-256: {hashlib.sha256(response.content).hexdigest()}")
    print("  Response headers:")
    for name, value in response.headers.multi_items():
        print(f"    {name}: {value}")


def print_raw_response(response: httpx.Response) -> None:
    print("\nRaw response body (verbatim, untruncated):")
    print(response.text)


def main() -> int:
    args = parse_args()
    load_dotenv(DEFAULT_ENV_FILE)
    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        print(
            f"Error: {API_KEY_ENV} is not set in {DEFAULT_ENV_FILE} "
            "or the environment.",
            file=sys.stderr,
        )
        return 2

    try:
        image_bytes, width, height, mime_type = inspect_image(args.image)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    request_payload = build_request(width, height)
    print_request_diagnostics(
        image_path=args.image,
        image_bytes=image_bytes,
        width=width,
        height=height,
        mime_type=mime_type,
        timeout=args.timeout,
        request_payload=request_payload,
    )
    started_at = time.perf_counter()
    try:
        with httpx.Client(timeout=args.timeout) as client:
            response = client.post(
                API_URL,
                headers={"apikey": api_key},
                files={
                    "file": (args.image.name, image_bytes, mime_type),
                    "request": (None, json.dumps(request_payload)),
                },
            )
    except httpx.TimeoutException:
        elapsed = time.perf_counter() - started_at
        print(f"Error: request timed out after {elapsed:.2f} seconds", file=sys.stderr)
        return 1
    except httpx.RequestError as exc:
        elapsed = time.perf_counter() - started_at
        print(
            f"Error: could not reach TrafficEye after {elapsed:.2f} seconds: {exc}",
            file=sys.stderr,
        )
        return 1

    elapsed = time.perf_counter() - started_at
    print_response_metadata(response, elapsed)

    if not response.is_success:
        print_raw_response(response)
        print(
            f"Error: TrafficEye returned HTTP {response.status_code}",
            file=sys.stderr,
        )
        return 1

    try:
        payload = response.json()
    except ValueError as exc:
        print_raw_response(response)
        print(f"Error: TrafficEye returned invalid JSON: {exc}", file=sys.stderr)
        return 1
    if not isinstance(payload, dict):
        print_raw_response(response)
        print("Error: TrafficEye returned JSON that is not an object", file=sys.stderr)
        return 1

    print()
    print_summary(payload)
    print("\nComplete parsed JSON response (all fields):")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print_raw_response(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
