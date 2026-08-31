from __future__ import annotations

import csv

from models import MMRResult
from pipeline.report import generate_reports


class _DummyRunStore:
    def __init__(self, root) -> None:
        self.labels_path = root / "labels.json"
        self.count_events_path = root / "count_events.jsonl"
        self.tracks_path = root / "tracks.jsonl"
        self.report_csv_path = root / "report.csv"
        self.labels = self

    def read(self) -> dict[int, MMRResult]:
        return {
            1: MMRResult(
                make="Toyota",
                model="Corolla",
                accepted=True,
                vehicle_index=1,
                evidence_source="api_confirmed",
                resolution_method="embedding_retrieval",
                retrieval_record_id="record-1",
                retrieval_distance=0.001,
                retrieval_neighbor_count=2,
            )
        }


def test_report_preserves_retrieval_provenance(tmp_path) -> None:
    store = _DummyRunStore(tmp_path)

    generate_reports(store)

    with store.report_csv_path.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))

    assert row["evidence_source"] == "api_confirmed"
    assert row["resolution_method"] == "embedding_retrieval"
    assert row["retrieval_record_id"] == "record-1"
    assert row["retrieval_distance"] == "0.001"
    assert row["retrieval_neighbor_count"] == "2"
