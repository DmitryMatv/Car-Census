from __future__ import annotations

from pathlib import Path
from typing import Self

import cv2
import numpy as np

from mmr.trafficeye import TrafficEyeClient


def _write_image(path: Path, quality: int) -> None:
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[:, :] = [80, 140, 200]
    image[20:60, 30:90] = [200, 100, 40]
    ok = cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok


def _payload() -> dict[str, object]:
    return {
        "combinations": [
            {
                "roadUsers": [
                    {
                        "box": {
                            "position": {
                                "topLeftCol": 0,
                                "topLeftRow": 0,
                                "bottomRightCol": 120,
                                "bottomRightRow": 80,
                            }
                        },
                        "mmr": {
                            "make": {"value": "Toyota", "score": 0.95},
                            "model": {"value": "Corolla", "score": 0.90},
                            "generation": {"value": "E210", "score": 0.85},
                            "variation": {"value": "Hybrid", "score": 0.80},
                            "color": {"value": "BLUE", "score": 0.80},
                        },
                    }
                ]
            }
        ]
    }


class _FakeResponse:
    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, object]:
        return _payload()


class _FakeHttpClient:
    calls = 0
    image_shapes: list[tuple[int, int]] = []

    def __init__(self, timeout: float) -> None:
        _ = timeout

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        _ = exc_type, exc_value, traceback

    def post(self, url, headers, files) -> _FakeResponse:
        _ = url, headers, files
        type(self).calls += 1
        image = cv2.imdecode(
            np.frombuffer(files["file"][1], dtype=np.uint8), cv2.IMREAD_COLOR
        )
        assert image is not None
        type(self).image_shapes.append(image.shape[:2])
        return _FakeResponse()


def test_exact_retrieval_hit_does_not_need_api_key_or_http_call(
    default_config, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("mmr.trafficeye.httpx.Client", _FakeHttpClient)
    monkeypatch.setenv("TRAFFICEYE_API_KEY", "test-key")
    _FakeHttpClient.calls = 0
    _FakeHttpClient.image_shapes = []
    image_path = tmp_path / "crop.jpg"
    _write_image(image_path, quality=95)
    cache_dir = tmp_path / "cache"

    first = TrafficEyeClient(config=default_config, cache_dir=cache_dir)
    first_result = first.recognize_vehicle_crop(image_path)

    assert first_result.resolution_method == "external_api"
    assert first_result.evidence_source == "api_confirmed"
    assert _FakeHttpClient.calls == 1
    for path in cache_dir.glob("*.json"):
        path.unlink()

    monkeypatch.delenv("TRAFFICEYE_API_KEY")
    second = TrafficEyeClient(config=default_config, cache_dir=cache_dir)
    second_result = second.recognize_vehicle_crop(image_path)

    assert second_result.resolution_method == "exact_retrieval"
    assert second_result.evidence_source == "api_confirmed"
    assert second_result.make == "Toyota"
    assert second_result.color == "BLUE"
    assert second_result.source_image == image_path
    assert _FakeHttpClient.calls == 1


def test_embedding_shadow_match_still_calls_traffic_eye(
    config_factory, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("mmr.trafficeye.httpx.Client", _FakeHttpClient)
    monkeypatch.setenv("TRAFFICEYE_API_KEY", "test-key")
    _FakeHttpClient.calls = 0
    _FakeHttpClient.image_shapes = []
    first_path = tmp_path / "first.jpg"
    second_path = tmp_path / "second.jpg"
    _write_image(first_path, quality=95)
    _write_image(second_path, quality=90)
    cache_dir = tmp_path / "cache"
    config = config_factory({"mmr": {"retrieval_mode": "shadow"}})

    client = TrafficEyeClient(config=config, cache_dir=cache_dir)
    client.recognize_vehicle_crop(first_path)
    result = client.recognize_vehicle_crop(second_path)

    assert result.resolution_method == "external_api"
    assert _FakeHttpClient.calls == 2
    audit_lines = (cache_dir / "retrieval" / "lookup_audit.jsonl").read_text().splitlines()
    assert '"action":"shadow"' in audit_lines[-1]


def test_embedding_enforce_hit_reuses_identity_without_observation_fields(
    config_factory, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("mmr.trafficeye.httpx.Client", _FakeHttpClient)
    monkeypatch.setenv("TRAFFICEYE_API_KEY", "test-key")
    _FakeHttpClient.calls = 0
    _FakeHttpClient.image_shapes = []
    first_path = tmp_path / "first.jpg"
    second_path = tmp_path / "second.jpg"
    _write_image(first_path, quality=95)
    _write_image(second_path, quality=90)
    cache_dir = tmp_path / "cache"

    shadow_config = config_factory({"mmr": {"retrieval_mode": "shadow"}})
    TrafficEyeClient(config=shadow_config, cache_dir=cache_dir).recognize_vehicle_crop(
        first_path
    )

    monkeypatch.delenv("TRAFFICEYE_API_KEY")
    enforce_config = config_factory({"mmr": {"retrieval_mode": "enforce"}})
    result = TrafficEyeClient(
        config=enforce_config, cache_dir=cache_dir
    ).recognize_vehicle_crop(second_path)

    assert result.resolution_method == "embedding_retrieval"
    assert result.evidence_source == "api_confirmed"
    assert result.make == "Toyota"
    assert result.model == "Corolla"
    assert result.generation == "E210"
    assert result.color is None
    assert result.detection_box is None
    assert _FakeHttpClient.calls == 1


def test_batch_retrieval_removes_exact_hits_before_api_request(
    config_factory, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("mmr.trafficeye.httpx.Client", _FakeHttpClient)
    monkeypatch.setenv("TRAFFICEYE_API_KEY", "test-key")
    _FakeHttpClient.calls = 0
    _FakeHttpClient.image_shapes = []
    first_path = tmp_path / "first.jpg"
    second_path = tmp_path / "second.jpg"
    _write_image(first_path, quality=95)
    _write_image(second_path, quality=70)
    cache_dir = tmp_path / "cache"

    single_config = config_factory({"mmr": {"batch_size": 1}})
    TrafficEyeClient(config=single_config, cache_dir=cache_dir).recognize_vehicle_crop(
        first_path
    )

    batch_config = config_factory(
        {"mmr": {"batch_size": 2, "batch_grid_columns": 2}}
    )
    results = TrafficEyeClient(
        config=batch_config, cache_dir=cache_dir
    ).recognize_vehicle_crops([first_path, second_path])

    assert len(results) == 2
    assert results[0].resolution_method == "exact_retrieval"
    assert results[1].resolution_method == "external_api"
    assert _FakeHttpClient.calls == 2
    assert _FakeHttpClient.image_shapes[-1] == (512, 512)


def test_batch_api_evidence_can_be_reused_by_single_crop_request(
    config_factory, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("mmr.trafficeye.httpx.Client", _FakeHttpClient)
    monkeypatch.setenv("TRAFFICEYE_API_KEY", "test-key")
    _FakeHttpClient.calls = 0
    _FakeHttpClient.image_shapes = []
    image_path = tmp_path / "crop.jpg"
    _write_image(image_path, quality=95)
    cache_dir = tmp_path / "cache"

    batch_config = config_factory({"mmr": {"batch_size": 2}})
    batch_result = TrafficEyeClient(
        config=batch_config, cache_dir=cache_dir
    ).recognize_vehicle_crops([image_path])[0]
    assert batch_result.resolution_method == "external_api"
    assert _FakeHttpClient.calls == 1
    for path in cache_dir.glob("*.json"):
        path.unlink()

    monkeypatch.delenv("TRAFFICEYE_API_KEY")
    single_config = config_factory({"mmr": {"batch_size": 1}})
    single_result = TrafficEyeClient(
        config=single_config, cache_dir=cache_dir
    ).recognize_vehicle_crop(image_path)

    assert single_result.resolution_method == "exact_retrieval"
    assert single_result.make == "Toyota"
    assert _FakeHttpClient.calls == 1
