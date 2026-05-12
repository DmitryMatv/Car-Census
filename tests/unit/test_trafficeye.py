import cv2
import numpy as np
import orjson

from config import AppConfig
from mmr.trafficeye import TrafficEyeClient, parse_mmr_response
from models import BBox


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

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            pass

        def post(self, url, headers, files):
            captured["url"] = url
            captured["headers"] = headers
            captured["request"] = orjson.loads(files["request"][1])
            return FakeResponse()

    monkeypatch.setattr("mmr.trafficeye.httpx.Client", FakeHttpClient)

    client = TrafficEyeClient(config=AppConfig(), cache_dir=tmp_path / "cache")
    client.recognize_vehicle_crop(image_path)

    request = captured["request"]
    assert request["tasks"] == ["DETECTION", "MMR"]
    assert request["requestedDetectionTypes"] == ["BOX"]
    assert request["mmrPreference"] == "BOX"
    assert "OCR" not in request["tasks"]
    assert "PLATE" not in request["requestedDetectionTypes"]
    assert "combinations" not in request
