from __future__ import annotations

from mmr.trafficeye_batch_grid import normalize_batch_result_for_source_crop
from models import BBox, MMRResult


def test_seed_normalizes_batch_detection_box_to_source_crop() -> None:
    result = MMRResult(
        make="Seat",
        model="Ateca",
        detection_box=BBox(x1=512, y1=1601, x2=1024, y2=1983),
        raw={
            "batch_content_box": {
                "x1": 512,
                "y1": 1601,
                "x2": 1024,
                "y2": 1983,
            }
        },
    )

    normalized = normalize_batch_result_for_source_crop(
        result,
        image_width=256,
        image_height=191,
    )

    assert normalized.detection_box == BBox(x1=0, y1=0, x2=256, y2=191)


def test_seed_batch_normalization_is_idempotent() -> None:
    result = MMRResult(
        make="Seat",
        model="Ateca",
        detection_box=BBox(x1=156, y1=128, x2=356, y2=384),
        raw={
            "batch_content_box": {
                "x1": 56,
                "y1": 0,
                "x2": 456,
                "y2": 512,
            }
        },
    )

    normalized = normalize_batch_result_for_source_crop(
        result,
        image_width=400,
        image_height=512,
    )
    normalized_again = normalize_batch_result_for_source_crop(
        normalized,
        image_width=400,
        image_height=512,
    )

    assert normalized.detection_box == BBox(x1=100, y1=128, x2=300, y2=384)
    assert normalized_again == normalized
