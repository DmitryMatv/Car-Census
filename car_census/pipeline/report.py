from __future__ import annotations

from pathlib import Path

import orjson

from car_census.stats.counts import (
    aggregate_counts,
    write_counts_csv,
    write_counts_json,
)
from car_census.storage.run_store import RunStore
from car_census.types import CountEvent, MMRResult, TrackSummary


def _load_labels(path: Path) -> dict[int, MMRResult]:
    if not path.exists():
        return {}
    raw = orjson.loads(path.read_bytes())
    return {
        int(track_id): MMRResult.model_validate(payload)
        for track_id, payload in raw.items()
    }


def _load_counted_track_ids(path: Path) -> set[int]:
    track_ids: set[int] = set()
    if not path.exists():
        return track_ids
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                track_ids.add(CountEvent.model_validate(orjson.loads(line)).track_id)
    return track_ids


def _load_counted_track_ids_from_summaries(path: Path) -> set[int]:
    track_ids: set[int] = set()
    if not path.exists():
        return track_ids
    with path.open("rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            summary = TrackSummary.model_validate(orjson.loads(line))
            if summary.counted:
                track_ids.add(summary.track_id)
    return track_ids


def generate_reports(run_store: RunStore) -> dict[str, object]:
    labels = _load_labels(run_store.labels_path)
    counted_track_ids = _load_counted_track_ids(run_store.count_events_path)
    if not counted_track_ids:
        counted_track_ids = _load_counted_track_ids_from_summaries(
            run_store.tracks_path
        )
    payload = aggregate_counts(
        labels_by_track=labels, counted_track_ids=counted_track_ids
    )
    write_counts_json(run_store.counts_json_path, payload)
    write_counts_csv(run_store.counts_csv_path, payload)
    return payload
