from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import orjson

from models import MMRResult


def aggregate_counts(
    labels_by_track: dict[int, MMRResult], counted_track_ids: set[int]
) -> dict[str, object]:
    by_make = Counter()
    by_model = Counter()
    by_make_model = Counter()
    counted_keys: set[tuple[str, int]] = set()

    for track_id in counted_track_ids:
        label = labels_by_track.get(track_id)
        if label is None:
            continue
        counted_key = (
            ("vehicle", label.vehicle_index)
            if label.vehicle_index is not None
            else ("track", track_id)
        )
        if counted_key in counted_keys:
            continue
        counted_keys.add(counted_key)
        make = (label.make or "unknown").strip() or "unknown"
        model = (label.model or "unknown").strip() or "unknown"
        by_make[make] += 1
        by_model[model] += 1
        by_make_model[f"{make} {model}".strip()] += 1

    return {
        "total_counted": len(counted_keys),
        "by_make": dict(by_make),
        "by_model": dict(by_model),
        "by_make_model": dict(by_make_model),
    }


def write_counts_json(path: Path, payload: dict[str, object]) -> None:
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))


def write_counts_csv(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "label", "count"])
        for group in ["by_make", "by_model", "by_make_model"]:
            for label, count in sorted(payload[group].items()):
                writer.writerow([group, label, count])
