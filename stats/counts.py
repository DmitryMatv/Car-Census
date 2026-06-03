from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from models import MMRResult

BASE_REPORT_COLUMNS = [
    "vehicle_index",
    "track_id",
    "category",
    "category_confidence",
    "make",
    "model",
    "generation",
    "variation",
    "accepted",
    "make_confidence",
    "model_confidence",
    "generation_confidence",
    "variation_confidence",
    "color",
    "view",
    "view8",
]

_TAG_PREFIX = "tag_"


def _empty_if_none(value: Any) -> Any:
    return "" if value is None else value


def _report_sort_key(item: tuple[int, MMRResult]) -> tuple[int, int, int, int]:
    track_id, label = item
    if label.vehicle_index is not None:
        return (
            0,
            label.vehicle_index,
            label.api_classification_index or 0,
            track_id,
        )
    return (1, 0, track_id, track_id)


def _normalize_tag_name(name: object) -> str:
    if name is None:
        return ""
    normalized = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    return f"{_TAG_PREFIX}{normalized}" if normalized else ""


def _is_affirmative_tag_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"yes", "true"}
    return False


def _tag_confidence(tag: dict[str, Any]) -> Any:
    return _empty_if_none(tag.get("score", tag.get("confidence")))


def _affirmative_tags(label: MMRResult) -> dict[str, Any]:
    tags: dict[str, Any] = {}
    for tag in label.tags:
        column = _normalize_tag_name(tag.get("name"))
        if column and _is_affirmative_tag_value(tag.get("value")):
            tags[column] = _tag_confidence(tag)
    return tags


def _base_report_row(track_id: int, label: MMRResult) -> dict[str, Any]:
    return {
        "vehicle_index": _empty_if_none(label.vehicle_index),
        "track_id": track_id,
        "make": _empty_if_none(label.make),
        "model": _empty_if_none(label.model),
        "category": _empty_if_none(label.category),
        "generation": _empty_if_none(label.generation),
        "variation": _empty_if_none(label.variation),
        "color": _empty_if_none(label.color),
        "view": _empty_if_none(label.view),
        "view8": _empty_if_none(label.view8),
        "accepted": label.accepted,
        "make_confidence": _empty_if_none(label.make_confidence),
        "model_confidence": _empty_if_none(label.model_confidence),
        "category_confidence": _empty_if_none(label.category_confidence),
        "generation_confidence": _empty_if_none(label.generation_confidence),
        "variation_confidence": _empty_if_none(label.variation_confidence),
    }


def build_vehicle_report_rows(
    labels_by_track: dict[int, MMRResult],
) -> list[dict[str, Any]]:
    rows_with_tags: list[tuple[dict[str, Any], dict[str, Any]]] = []
    seen_keys: set[int] = set()

    for track_id, label in sorted(labels_by_track.items(), key=_report_sort_key):
        if label.vehicle_index is None:
            continue
        identity_key = label.vehicle_index
        if identity_key in seen_keys:
            continue
        seen_keys.add(identity_key)
        rows_with_tags.append(
            (_base_report_row(track_id, label), _affirmative_tags(label))
        )

    tag_columns = sorted(
        {column for _row, row_tags in rows_with_tags for column in row_tags}
    )
    rows: list[dict[str, Any]] = []
    for row, row_tags in rows_with_tags:
        for column in tag_columns:
            confidence_column = f"{column}_confidence"
            if column in row_tags:
                row[column] = True
                row[confidence_column] = row_tags[column]
            else:
                row[column] = ""
                row[confidence_column] = ""
        rows.append(row)
    return rows


def report_csv_columns(rows: list[dict[str, Any]]) -> list[str]:
    tag_columns = sorted(
        column
        for row in rows
        for column in row
        if column.startswith(_TAG_PREFIX) and not column.endswith("_confidence")
    )
    ordered_tag_columns: list[str] = []
    for column in dict.fromkeys(tag_columns):
        ordered_tag_columns.extend([column, f"{column}_confidence"])
    return [*BASE_REPORT_COLUMNS, *ordered_tag_columns]


def write_vehicle_report_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = report_csv_columns(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
