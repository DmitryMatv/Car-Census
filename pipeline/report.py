from __future__ import annotations

from pathlib import Path

import orjson

from stats.counts import build_vehicle_report_rows, write_vehicle_report_csv
from storage.run_store import RunStore
from models import MMRResult


def _load_labels(path: Path) -> dict[int, MMRResult]:
    if not path.exists():
        return {}
    raw = orjson.loads(path.read_bytes())
    return {
        int(track_id): MMRResult.model_validate(payload)
        for track_id, payload in raw.items()
    }


def _remove_legacy_reports(run_store: RunStore) -> None:
    for path in [run_store.counts_json_path, run_store.counts_csv_path]:
        if path.exists():
            path.unlink()


def generate_reports(run_store: RunStore) -> dict[str, object]:
    labels = _load_labels(run_store.labels_path)
    rows = build_vehicle_report_rows(labels)
    _remove_legacy_reports(run_store)
    write_vehicle_report_csv(run_store.report_csv_path, rows)
    return {
        "report_csv": str(run_store.report_csv_path),
        "rows": len(rows),
    }
