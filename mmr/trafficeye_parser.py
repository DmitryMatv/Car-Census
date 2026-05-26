from __future__ import annotations

from pathlib import Path
from typing import Any

from models import BBox, MMRResult


def _coerce_confidence(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        score = value.get("score") or value.get("confidence")
        if isinstance(score, (int, float)):
            return float(score)
    return None


def _coerce_label(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, dict):
        nested = value.get("value") or value.get("name") or value.get("label")
        if isinstance(nested, str):
            stripped = nested.strip()
            return stripped or None
    return None


def _coerce_box(value: Any) -> BBox | None:
    if not isinstance(value, dict):
        return None
    try:
        box = BBox(
            x1=float(value["topLeftCol"]),
            y1=float(value["topLeftRow"]),
            x2=float(value["bottomRightCol"]),
            y2=float(value["bottomRightRow"]),
        )
    except (KeyError, TypeError, ValueError):
        return None
    if box.width <= 0 or box.height <= 0:
        return None
    return box


def _recognition_payload(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data")
    if isinstance(data, dict) and isinstance(data.get("combinations"), list):
        return data
    return payload


def _road_user_box_position(road_user: dict[str, Any]) -> dict[str, Any] | None:
    box = road_user.get("box")
    if isinstance(box, dict) and isinstance(box.get("position"), dict):
        return box["position"]
    mmr = road_user.get("mmr")
    if not isinstance(mmr, dict):
        return None
    input_data = mmr.get("input")
    if isinstance(input_data, dict) and isinstance(input_data.get("box"), dict):
        return input_data["box"]
    return None


def _iter_road_users(payload: dict[str, Any]) -> list[dict[str, Any]]:
    road_users: list[dict[str, Any]] = []
    combinations = _recognition_payload(payload).get("combinations")
    if not isinstance(combinations, list):
        return road_users
    for combination in combinations:
        if not isinstance(combination, dict):
            continue
        items = combination.get("roadUsers")
        if not isinstance(items, list):
            continue
        road_users.extend(item for item in items if isinstance(item, dict))
    return road_users


def _road_user_box_area_rank(
    road_user: dict[str, Any],
) -> tuple[float, float, float, float]:
    position = _road_user_box_position(road_user)
    box = _coerce_box(position)
    mmr = road_user.get("mmr")
    assert isinstance(mmr, dict)
    return (
        box.area if box is not None else 0.0,
        _coerce_confidence(position) or 0.0,
        _coerce_confidence(mmr.get("model")) or 0.0,
        _coerce_confidence(mmr.get("make")) or 0.0,
    )


def _selected_road_user(payload: dict[str, Any]) -> dict[str, Any] | None:
    candidates = [
        road_user
        for road_user in _iter_road_users(payload)
        if isinstance(road_user.get("mmr"), dict)
    ]
    if not candidates:
        return None

    return max(candidates, key=_road_user_box_area_rank)


def _ranked_road_user(road_users: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        road_user for road_user in road_users if isinstance(road_user.get("mmr"), dict)
    ]
    if not candidates:
        return None

    return max(candidates, key=_road_user_box_area_rank)


def _mmr_result_from_road_user(
    road_user: dict[str, Any],
    raw: dict[str, Any],
    source_image: Path | None = None,
) -> MMRResult:
    mmr = road_user["mmr"]
    position = _road_user_box_position(road_user)
    tags = mmr.get("tags")

    return MMRResult(
        make=_coerce_label(mmr.get("make")),
        model=_coerce_label(mmr.get("model")),
        make_confidence=_coerce_confidence(mmr.get("make")),
        model_confidence=_coerce_confidence(mmr.get("model")),
        category=_coerce_label(mmr.get("category")),
        category_confidence=_coerce_confidence(mmr.get("category")),
        generation=_coerce_label(mmr.get("generation")),
        generation_confidence=_coerce_confidence(mmr.get("generation")),
        variation=_coerce_label(mmr.get("variation")),
        variation_confidence=_coerce_confidence(mmr.get("variation")),
        color=_coerce_label(mmr.get("color")),
        color_confidence=_coerce_confidence(mmr.get("color")),
        view=_coerce_label(mmr.get("view")),
        view_confidence=_coerce_confidence(mmr.get("view")),
        view8=_coerce_label(mmr.get("view8")),
        view8_confidence=_coerce_confidence(mmr.get("view8")),
        tags=[tag for tag in tags if isinstance(tag, dict)]
        if isinstance(tags, list)
        else [],
        detection_box=_coerce_box(position),
        detection_confidence=_coerce_confidence(position),
        raw=raw,
        source_image=source_image,
    )


def parse_mmr_results(
    payload: dict[str, Any], source_image: Path | None = None
) -> list[MMRResult]:
    return [
        _mmr_result_from_road_user(road_user, raw=payload, source_image=source_image)
        for road_user in _iter_road_users(payload)
        if isinstance(road_user.get("mmr"), dict)
    ]


def parse_mmr_results_by_combination(
    payload: dict[str, Any], source_image: Path | None = None
) -> list[MMRResult | None]:
    combinations = _recognition_payload(payload).get("combinations")
    if not isinstance(combinations, list):
        return []

    results: list[MMRResult | None] = []
    for combination in combinations:
        if not isinstance(combination, dict):
            results.append(None)
            continue
        road_users = combination.get("roadUsers")
        if not isinstance(road_users, list):
            results.append(None)
            continue
        road_user = _ranked_road_user(
            [item for item in road_users if isinstance(item, dict)]
        )
        results.append(
            _mmr_result_from_road_user(
                road_user, raw=payload, source_image=source_image
            )
            if road_user is not None
            else None
        )
    return results


def parse_mmr_response(
    payload: dict[str, Any], source_image: Path | None = None
) -> MMRResult:
    road_user = _selected_road_user(payload)
    if road_user is None:
        return MMRResult(raw=payload, source_image=source_image)

    return _mmr_result_from_road_user(road_user, raw=payload, source_image=source_image)
