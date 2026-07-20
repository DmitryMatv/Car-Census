from __future__ import annotations

from typing import Self

import cv2
import numpy as np
import orjson
import pytest

from config import AppConfig
from mmr.trafficeye import (
    TrafficEyeClient,
    parse_mmr_response,
    parse_mmr_results,
    parse_mmr_results_by_combination,
)
from mmr.trafficeye_batch_grid import (
    BatchCell,
    _resize_for_batch_cell,
    batch_request_payload,
    build_batch_image,
    decode_image,
    match_batch_results,
)
from mmr.trafficeye_cache import hash_request
from models import BBox, MMRResult


def test_hash_request_is_stable_for_same_image_and_payload() -> None:
    payload = {"tasks": ["MMR"], "mmrPreference": "BOX"}

    first = hash_request(b"image-bytes", payload)
    second = hash_request(b"image-bytes", payload)

    assert first == second


def test_batch_request_payload_uses_content_boxes(tmp_path) -> None:
    cells = [
        BatchCell(
            index=0,
            source_image=tmp_path / "crop.jpg",
            cell_box=BBox(x1=0, y1=0, x2=100, y2=100),
            content_box=BBox(x1=10, y1=20, x2=90, y2=80),
        )
    ]

    request = batch_request_payload(cells, mmr_preference="BOX")

    assert request["tasks"] == ["MMR"]
    assert request["mmrPreference"] == "BOX"
    assert request["combinations"][0]["roadUsers"][0]["box"]["position"] == {
        "topLeftCol": 10,
        "topLeftRow": 20,
        "bottomRightCol": 90,
        "bottomRightRow": 80,
    }


def test_decode_image_raises_for_invalid_image_bytes(tmp_path) -> None:
    image_path = tmp_path / "bad.jpg"

    with pytest.raises(RuntimeError, match="Could not decode crop for MMR request"):
        decode_image(b"not-an-image", image_path)


def test_traffic_eye_acceptance_requires_make_confidence_and_model(tmp_path) -> None:
    config = AppConfig.model_validate({"mmr": {"accept_model_confidence": 0.30}})
    client = TrafficEyeClient(
        config=config,
        cache_dir=tmp_path / "cache",
        require_api_key=False,
    )

    make_only = client._mark_acceptance(
        MMRResult(make="Toyota", make_confidence=0.90, model_confidence=0.90)
    )
    model_only = client._mark_acceptance(
        MMRResult(model="Corolla", make_confidence=0.90, model_confidence=0.90)
    )
    low_make_confidence = client._mark_acceptance(
        MMRResult(
            make="Toyota",
            model="Corolla",
            make_confidence=0.29,
            model_confidence=0.90,
        )
    )
    low_model_confidence = client._mark_acceptance(
        MMRResult(
            make="Toyota",
            model="Corolla",
            make_confidence=0.90,
            model_confidence=0.29,
        )
    )
    accepted = client._mark_acceptance(
        MMRResult(
            make="Toyota",
            model="Corolla",
            make_confidence=0.30,
            model_confidence=0.30,
        )
    )

    assert make_only.accepted is False
    assert model_only.accepted is False
    assert low_make_confidence.accepted is False
    assert low_model_confidence.accepted is True
    assert accepted.accepted is True


def test_resize_for_batch_cell_uses_lanczos_when_resampling(monkeypatch) -> None:
    image = np.full((10, 20, 3), 100, dtype=np.uint8)
    captured: dict[str, object] = {}

    def fake_resize(image_arg, size, *, interpolation):
        captured["image"] = image_arg
        captured["size"] = size
        captured["interpolation"] = interpolation
        return np.full((size[1], size[0], 3), 50, dtype=np.uint8)

    monkeypatch.setattr(cv2, "resize", fake_resize)

    resized = _resize_for_batch_cell(image, (40, 20))

    assert resized.shape[:2] == (20, 40)
    assert captured["image"] is image
    assert captured["size"] == (40, 20)
    assert captured["interpolation"] == cv2.INTER_LANCZOS4


def test_resize_for_batch_cell_skips_exact_size_resampling(monkeypatch) -> None:
    image = np.full((10, 20, 3), 100, dtype=np.uint8)

    def fail_resize(*args, **kwargs):
        raise AssertionError("exact-size batch crops should not be resampled")

    monkeypatch.setattr(cv2, "resize", fail_resize)

    assert _resize_for_batch_cell(image, (20, 10)) is image


def test_build_batch_image_rejects_empty_image_paths() -> None:
    with pytest.raises(ValueError, match="image_paths cannot be empty"):
        build_batch_image([], columns=1, cell_size_px=100, jpeg_quality=95)


def test_build_batch_image_rejects_nonpositive_columns(tmp_path) -> None:
    image_path = tmp_path / "crop.jpg"

    with pytest.raises(ValueError, match="columns must be positive, got 0"):
        build_batch_image([image_path], columns=0, cell_size_px=100, jpeg_quality=95)


def test_build_batch_image_rejects_nonpositive_cell_size(tmp_path) -> None:
    image_path = tmp_path / "crop.jpg"

    with pytest.raises(ValueError, match="cell_size_px must be positive, got 0"):
        build_batch_image([image_path], columns=1, cell_size_px=0, jpeg_quality=95)


def test_match_batch_results_prefers_combination_order_before_box_fallback(
    tmp_path,
) -> None:
    cells = [
        BatchCell(
            index=0,
            source_image=tmp_path / "first.jpg",
            cell_box=BBox(x1=0, y1=0, x2=100, y2=100),
            content_box=BBox(x1=0, y1=0, x2=100, y2=100),
        ),
        BatchCell(
            index=1,
            source_image=tmp_path / "second.jpg",
            cell_box=BBox(x1=100, y1=0, x2=200, y2=100),
            content_box=BBox(x1=100, y1=0, x2=200, y2=100),
        ),
    ]
    payload = {
        "combinations": [
            {
                "roadUsers": [
                    {
                        "box": {
                            "position": {
                                "topLeftCol": 130,
                                "topLeftRow": 10,
                                "bottomRightCol": 190,
                                "bottomRightRow": 90,
                            }
                        },
                        "mmr": {
                            "make": {"value": "Toyota", "score": 0.9},
                            "model": {"value": "Corolla", "score": 0.8},
                        },
                    }
                ]
            },
            {"roadUsers": []},
        ]
    }

    results = match_batch_results(
        payload, cells, tmp_path / "batch.jpg", lambda result: result
    )

    assert [result.make for result in results] == ["Toyota", None]
    assert [result.source_image for result in results] == [
        tmp_path / "first.jpg",
        tmp_path / "second.jpg",
    ]
    assert results[1].raw["skipped_reason"] == "batch_no_mmr_result"


def test_parse_mmr_response_replaces_underscores_with_spaces() -> None:
    payload = {
        "combinations": [
            {
                "roadUsers": [
                    {
                        "box": {
                            "position": {
                                "topLeftCol": 0,
                                "topLeftRow": 0,
                                "bottomRightCol": 20,
                                "bottomRightRow": 20,
                            }
                        },
                        "mmr": {
                            "make": {"value": "Hyundai", "score": 0.95},
                            "model": {"value": "Santa_Fe", "score": 0.93},
                            "generation": {"value": "Mk_IV (2018)", "score": 0.91},
                            "variation": {"value": "Ultimate", "score": 0.80},
                        },
                    }
                ]
            }
        ]
    }

    result = parse_mmr_response(payload)

    assert result.make == "Hyundai"
    assert result.model == "Santa Fe"
    assert result.generation == "Mk IV (2018)"
    assert result.variation == "Ultimate"


def test_parse_mmr_response_reads_largest_detected_road_user() -> None:
    payload = {
        "combinations": [
            {
                "roadUsers": [
                    {
                        "box": {
                            "position": {
                                "topLeftCol": 0,
                                "topLeftRow": 0,
                                "bottomRightCol": 20,
                                "bottomRightRow": 20,
                                "score": 0.95,
                            }
                        },
                        "mmr": {
                            "make": {"value": "Audi", "score": 0.99},
                            "model": {"value": "A4", "score": 0.98},
                        },
                    },
                    {
                        "box": {
                            "position": {
                                "topLeftCol": 5,
                                "topLeftRow": 6,
                                "bottomRightCol": 105,
                                "bottomRightRow": 86,
                                "score": 0.91,
                            }
                        },
                        "mmr": {
                            "view": {"value": "frontal", "score": 0.70},
                            "view8": {"value": "frontal+right", "score": 0.71},
                            "category": {"value": "CAR", "score": 0.72},
                            "make": {"value": "VW", "score": 0.91},
                            "model": {"value": "Passat", "score": 0.82},
                            "generation": {"value": "Mk VI (2005)", "score": 0.81},
                            "variation": {"value": "Variant", "score": 0.73},
                            "color": {"value": "WHITE", "score": 0.74},
                            "tags": [{"name": "taxi", "value": "no", "score": 0.9}],
                        },
                    },
                ]
            }
        ]
    }
    result = parse_mmr_response(payload)
    assert result.make == "VW"
    assert result.model == "Passat"
    assert result.make_confidence == 0.91
    assert result.model_confidence == 0.82
    assert result.generation == "Mk VI (2005)"
    assert result.generation_confidence == 0.81
    assert result.variation == "Variant"
    assert result.variation_confidence == 0.73
    assert result.color == "WHITE"
    assert result.color_confidence == 0.74
    assert result.category == "CAR"
    assert result.category_confidence == 0.72
    assert result.view == "frontal"
    assert result.view_confidence == 0.70
    assert result.view8 == "frontal+right"
    assert result.view8_confidence == 0.71
    assert result.tags == [{"name": "taxi", "value": "no", "score": 0.9}]
    assert result.detection_box == BBox(x1=5, y1=6, x2=105, y2=86)
    assert result.detection_confidence == 0.91


def test_parse_mmr_response_falls_back_to_mmr_input_box() -> None:
    payload = {
        "combinations": [
            {
                "roadUsers": [
                    {
                        "mmr": {
                            "make": {"value": "Toyota", "score": 0.8},
                            "model": {"value": "Corolla", "score": 0.9},
                            "input": {
                                "box": {
                                    "topLeftCol": 1,
                                    "topLeftRow": 2,
                                    "bottomRightCol": 31,
                                    "bottomRightRow": 42,
                                }
                            },
                        }
                    }
                ]
            }
        ]
    }

    result = parse_mmr_response(payload)

    assert result.make == "Toyota"
    assert result.model == "Corolla"
    assert result.detection_box == BBox(x1=1, y1=2, x2=31, y2=42)


def test_parse_mmr_response_reads_traffic_eye_data_wrapper() -> None:
    payload = {
        "status": 200,
        "data": {
            "combinations": [
                {
                    "roadUsers": [
                        {
                            "box": {
                                "position": {
                                    "topLeftCol": 24,
                                    "topLeftRow": 19.8,
                                    "bottomRightCol": 181.7,
                                    "bottomRightRow": 249.6,
                                    "score": 1,
                                }
                            },
                            "mmr": {
                                "make": {"value": "Ford", "score": 1},
                                "model": {"value": "Kuga", "score": 1},
                                "generation": {
                                    "value": "Mk I (2008)",
                                    "score": 0.9999,
                                },
                                "color": {"value": "BLACK", "score": 0.9998},
                            },
                        }
                    ]
                }
            ]
        },
    }

    result = parse_mmr_response(payload)

    assert result.make == "Ford"
    assert result.model == "Kuga"
    assert result.generation == "Mk I (2008)"
    assert result.color == "BLACK"
    assert result.detection_box == BBox(x1=24, y1=19.8, x2=181.7, y2=249.6)
    assert result.raw == payload


def test_parse_mmr_response_without_mmr_preserves_raw() -> None:
    payload = {"combinations": [{"roadUsers": [{"box": {"position": {}}}]}]}

    result = parse_mmr_response(payload)

    assert result.make is None
    assert result.model is None
    assert result.accepted is False
    assert result.raw == payload


def test_parse_mmr_results_reads_all_detected_road_users() -> None:
    payload = {
        "combinations": [
            {
                "roadUsers": [
                    {
                        "box": {
                            "position": {
                                "topLeftCol": 0,
                                "topLeftRow": 0,
                                "bottomRightCol": 10,
                                "bottomRightRow": 10,
                            }
                        },
                        "mmr": {
                            "make": {"value": "Toyota", "score": 0.9},
                            "model": {"value": "Corolla", "score": 0.8},
                        },
                    },
                    {
                        "box": {
                            "position": {
                                "topLeftCol": 20,
                                "topLeftRow": 0,
                                "bottomRightCol": 30,
                                "bottomRightRow": 10,
                            }
                        },
                        "mmr": {
                            "make": {"value": "Audi", "score": 0.91},
                            "model": {"value": "A4", "score": 0.81},
                        },
                    },
                ]
            }
        ]
    }

    results = parse_mmr_results(payload)

    assert [result.make for result in results] == ["Toyota", "Audi"]
    assert [result.model for result in results] == ["Corolla", "A4"]
    assert results[0].detection_box == BBox(x1=0, y1=0, x2=10, y2=10)
    assert results[1].raw == payload


def test_parse_mmr_results_by_combination_preserves_empty_slots() -> None:
    payload = {
        "combinations": [
            {
                "roadUsers": [
                    {
                        "mmr": {
                            "make": {"value": "Toyota", "score": 0.9},
                            "model": {"value": "Corolla", "score": 0.8},
                        }
                    }
                ]
            },
            {"roadUsers": [{"box": {"position": {}}}]},
            {
                "roadUsers": [
                    {
                        "mmr": {
                            "make": {"value": "Audi", "score": 0.91},
                            "model": {"value": "A4", "score": 0.81},
                        }
                    }
                ]
            },
        ]
    }

    results = parse_mmr_results_by_combination(payload)

    assert [result.make if result is not None else None for result in results] == [
        "Toyota",
        None,
        "Audi",
    ]


def test_parse_mmr_results_by_combination_uses_largest_box_per_slot() -> None:
    payload = {
        "combinations": [
            {
                "roadUsers": [
                    {
                        "box": {
                            "position": {
                                "topLeftCol": 0,
                                "topLeftRow": 0,
                                "bottomRightCol": 20,
                                "bottomRightRow": 20,
                                "score": 0.99,
                            }
                        },
                        "mmr": {
                            "make": {"value": "Audi", "score": 0.99},
                            "model": {"value": "A4", "score": 0.99},
                        },
                    },
                    {
                        "box": {
                            "position": {
                                "topLeftCol": 10,
                                "topLeftRow": 10,
                                "bottomRightCol": 110,
                                "bottomRightRow": 70,
                                "score": 0.70,
                            }
                        },
                        "mmr": {
                            "make": {"value": "Toyota", "score": 0.65},
                            "model": {"value": "Corolla", "score": 0.60},
                        },
                    },
                ]
            }
        ]
    }

    results = parse_mmr_results_by_combination(payload)

    assert len(results) == 1
    result = results[0]
    assert result is not None
    assert result.make == "Toyota"
    assert result.model == "Corolla"
    assert result.detection_box == BBox(x1=10, y1=10, x2=110, y2=70)


def test_traffic_eye_client_requests_box_detection_and_mmr_only(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("TRAFFICEYE_API_KEY", "test-key")
    image_path = tmp_path / "crop.jpg"
    cv2.imwrite(str(image_path), np.full((10, 20, 3), 255, dtype=np.uint8))
    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, object]:
            return {"combinations": []}

    class FakeHttpClient:
        def __init__(self, timeout: float) -> None:
            captured["timeout"] = timeout

        def __enter__(self) -> Self:
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            pass

        def post(
            self, url, headers, files
        ) -> test_traffic_eye_client_requests_box_detection_and_mmr_only.FakeResponse:
            captured["url"] = url
            captured["headers"] = headers
            captured["request"] = orjson.loads(files["request"][1])
            return FakeResponse()

    monkeypatch.setattr("mmr.trafficeye.httpx.Client", FakeHttpClient)

    client = TrafficEyeClient(config=AppConfig(), cache_dir=tmp_path / "cache")
    client.recognize_vehicle_crop(image_path)

    request = captured["request"]
    assert request["tasks"] == ["MMR"]
    assert request["mmrPreference"] == "BOX"
    assert "requestedDetectionTypes" not in request
    assert request["combinations"] == [
        {
            "roadUsers": [
                {
                    "box": {
                        "position": {
                            "topLeftCol": 0,
                            "topLeftRow": 0,
                            "bottomRightCol": 20,
                            "bottomRightRow": 10,
                        }
                    }
                }
            ]
        }
    ]


def test_traffic_eye_client_batches_crops_with_manual_boxes(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("TRAFFICEYE_API_KEY", "test-key")
    first_crop = tmp_path / "first.jpg"
    second_crop = tmp_path / "second.jpg"
    cv2.imwrite(str(first_crop), np.full((50, 80, 3), 50, dtype=np.uint8))
    cv2.imwrite(str(second_crop), np.full((80, 50, 3), 150, dtype=np.uint8))
    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, object]:
            return {
                "combinations": [
                    {
                        "roadUsers": [
                            {
                                "box": {
                                    "position": {
                                        "topLeftCol": 0,
                                        "topLeftRow": 18,
                                        "bottomRightCol": 100,
                                        "bottomRightRow": 82,
                                    }
                                },
                                "mmr": {
                                    "make": {"value": "Toyota", "score": 0.9},
                                    "model": {"value": "Corolla", "score": 0.8},
                                },
                            }
                        ]
                    },
                    {
                        "roadUsers": [
                            {
                                "box": {
                                    "position": {
                                        "topLeftCol": 119,
                                        "topLeftRow": 0,
                                        "bottomRightCol": 181,
                                        "bottomRightRow": 100,
                                    }
                                },
                                "mmr": {
                                    "make": {"value": "Audi", "score": 0.91},
                                    "model": {"value": "A4", "score": 0.81},
                                },
                            }
                        ]
                    },
                ]
            }

    class FakeHttpClient:
        def __init__(self, timeout: float) -> None:
            captured["timeout"] = timeout

        def __enter__(self) -> Self:
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            pass

        def post(
            self, url, headers, files
        ) -> test_traffic_eye_client_batches_crops_with_manual_boxes.FakeResponse:
            captured["url"] = url
            captured["headers"] = headers
            captured["filename"] = files["file"][0]
            captured["request"] = orjson.loads(files["request"][1])
            return FakeResponse()

    monkeypatch.setattr("mmr.trafficeye.httpx.Client", FakeHttpClient)
    config = AppConfig.model_validate(
        {"mmr": {"batch_grid_columns": 2, "batch_cell_size_px": 100}}
    )

    client = TrafficEyeClient(config=config, cache_dir=tmp_path / "cache")
    results = client.recognize_vehicle_crops([first_crop, second_crop])
    debug_images = sorted((tmp_path / "batch_grids").glob("*.jpg"))
    debug_manifests = sorted((tmp_path / "batch_grids").glob("*.json"))

    request = captured["request"]
    assert request["tasks"] == ["MMR"]
    assert "requestedDetectionTypes" not in request
    assert request["mmrPreference"] == "BOX"
    assert len(request["combinations"]) == 2
    assert request["combinations"][0]["roadUsers"][0]["box"]["position"] == {
        "topLeftCol": 0.0,
        "topLeftRow": 19.0,
        "bottomRightCol": 100.0,
        "bottomRightRow": 81.0,
    }
    assert [result.make for result in results] == ["Toyota", "Audi"]
    assert [result.source_image for result in results] == [first_crop, second_crop]
    assert all(result.accepted for result in results)
    assert len(debug_images) == 1
    assert cv2.imread(str(debug_images[0])).shape[:2] == (100, 200)
    assert len(debug_manifests) == 1
    manifest = orjson.loads(debug_manifests[0].read_bytes())
    assert manifest["image"] == debug_images[0].name
    assert [cell["source_image"] for cell in manifest["cells"]] == [
        str(first_crop),
        str(second_crop),
    ]


def test_traffic_eye_client_batch_cell_uses_largest_returned_box(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("TRAFFICEYE_API_KEY", "test-key")
    crop = tmp_path / "crop.jpg"
    cv2.imwrite(str(crop), np.full((80, 80, 3), 100, dtype=np.uint8))

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, object]:
            return {
                "combinations": [
                    {
                        "roadUsers": [
                            {
                                "box": {
                                    "position": {
                                        "topLeftCol": 10,
                                        "topLeftRow": 10,
                                        "bottomRightCol": 30,
                                        "bottomRightRow": 30,
                                        "score": 0.99,
                                    }
                                },
                                "mmr": {
                                    "make": {"value": "Audi", "score": 0.99},
                                    "model": {"value": "A4", "score": 0.99},
                                },
                            },
                            {
                                "box": {
                                    "position": {
                                        "topLeftCol": 5,
                                        "topLeftRow": 5,
                                        "bottomRightCol": 75,
                                        "bottomRightRow": 70,
                                        "score": 0.70,
                                    }
                                },
                                "mmr": {
                                    "make": {"value": "Toyota", "score": 0.65},
                                    "model": {"value": "Corolla", "score": 0.61},
                                },
                            },
                        ]
                    }
                ]
            }

    class FakeHttpClient:
        def __init__(self, timeout: float) -> None:
            pass

        def __enter__(self) -> Self:
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            pass

        def post(
            self, url, headers, files
        ) -> test_traffic_eye_client_batch_cell_uses_largest_returned_box.FakeResponse:
            return FakeResponse()

    monkeypatch.setattr("mmr.trafficeye.httpx.Client", FakeHttpClient)
    config = AppConfig.model_validate(
        {"mmr": {"batch_grid_columns": 1, "batch_cell_size_px": 100}}
    )

    client = TrafficEyeClient(config=config, cache_dir=tmp_path / "cache")
    results = client.recognize_vehicle_crops([crop])

    assert len(results) == 1
    result = results[0]
    assert result.make == "Toyota"
    assert result.model == "Corolla"
    assert result.source_image == crop
    assert result.raw["batch_cell_index"] == 0
    assert result.detection_box == BBox(x1=5, y1=5, x2=75, y2=70)


def test_traffic_eye_client_matches_manual_batch_results_by_combination_order(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("TRAFFICEYE_API_KEY", "test-key")
    crops = []
    for index in range(3):
        crop = tmp_path / f"crop-{index}.jpg"
        cv2.imwrite(str(crop), np.full((50, 80, 3), index, dtype=np.uint8))
        crops.append(crop)

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, object]:
            return {
                "combinations": [
                    {
                        "roadUsers": [
                            {
                                "mmr": {
                                    "make": {"value": "Toyota", "score": 0.9},
                                    "model": {"value": "Corolla", "score": 0.8},
                                }
                            }
                        ]
                    },
                    {"roadUsers": [{"box": {}}]},
                    {
                        "roadUsers": [
                            {
                                "mmr": {
                                    "make": {"value": "Audi", "score": 0.91},
                                    "model": {"value": "A4", "score": 0.81},
                                }
                            }
                        ]
                    },
                ]
            }

    class FakeHttpClient:
        def __init__(self, timeout: float) -> None:
            pass

        def __enter__(self) -> Self:
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            pass

        def post(
            self, url, headers, files
        ) -> test_traffic_eye_client_matches_manual_batch_results_by_combination_order.FakeResponse:
            return FakeResponse()

    monkeypatch.setattr("mmr.trafficeye.httpx.Client", FakeHttpClient)
    config = AppConfig.model_validate(
        {"mmr": {"batch_grid_columns": 3, "batch_cell_size_px": 100}}
    )

    client = TrafficEyeClient(config=config, cache_dir=tmp_path / "cache")
    results = client.recognize_vehicle_crops(crops)

    assert [result.make for result in results] == ["Toyota", None, "Audi"]
    assert results[1].raw["skipped_reason"] == "batch_no_mmr_result"


def test_traffic_eye_client_does_not_reuse_ordered_batch_result_for_empty_slot(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("TRAFFICEYE_API_KEY", "test-key")
    crops = []
    for index in range(2):
        crop = tmp_path / f"crop-{index}.jpg"
        cv2.imwrite(str(crop), np.full((80, 80, 3), index, dtype=np.uint8))
        crops.append(crop)

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, object]:
            return {
                "combinations": [
                    {
                        "roadUsers": [
                            {
                                "box": {
                                    "position": {
                                        "topLeftCol": 75,
                                        "topLeftRow": 0,
                                        "bottomRightCol": 125,
                                        "bottomRightRow": 100,
                                    }
                                },
                                "mmr": {
                                    "make": {"value": "Toyota", "score": 0.9},
                                    "model": {"value": "Corolla", "score": 0.8},
                                },
                            }
                        ]
                    },
                    {"roadUsers": []},
                ]
            }

    class FakeHttpClient:
        def __init__(self, timeout: float) -> None:
            pass

        def __enter__(self) -> Self:
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            pass

        def post(
            self, url, headers, files
        ) -> test_traffic_eye_client_does_not_reuse_ordered_batch_result_for_empty_slot.FakeResponse:
            return FakeResponse()

    monkeypatch.setattr("mmr.trafficeye.httpx.Client", FakeHttpClient)
    config = AppConfig.model_validate(
        {"mmr": {"batch_grid_columns": 2, "batch_cell_size_px": 100}}
    )

    client = TrafficEyeClient(config=config, cache_dir=tmp_path / "cache")
    results = client.recognize_vehicle_crops(crops)

    assert [result.make for result in results] == ["Toyota", None]
    assert results[1].raw["skipped_reason"] == "batch_no_mmr_result"
