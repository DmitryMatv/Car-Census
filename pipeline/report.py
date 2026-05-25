from __future__ import annotations

from stats.counts import build_vehicle_report_rows, write_vehicle_report_csv
from storage.run_store import RunStore


def generate_reports(run_store: RunStore) -> dict[str, object]:
    labels = run_store.labels.read()
    rows = build_vehicle_report_rows(labels)
    write_vehicle_report_csv(run_store.report_csv_path, rows)
    return {
        "report_csv": str(run_store.report_csv_path),
        "rows": len(rows),
    }
