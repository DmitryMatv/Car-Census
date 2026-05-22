import csv

from models import MMRResult
from pipeline.report import generate_reports


class DummyRunStore:
    def __init__(self, root) -> None:
        self.labels_path = root / "labels.json"
        self.count_events_path = root / "count_events.jsonl"
        self.tracks_path = root / "tracks.jsonl"
        self.report_csv_path = root / "report.csv"
        self.labels: dict[int, MMRResult] = {}

    def read_labels(self) -> dict[int, MMRResult]:
        return self.labels


def test_generate_reports_writes_detailed_vehicle_csv(tmp_path) -> None:
    store = DummyRunStore(tmp_path)
    store.labels = {
        7: MMRResult(
            make="Toyota",
            model="Corolla",
            generation="E210 (2018)",
            variation="Hybrid Touring Sports",
            color="white",
            accepted=True,
            vehicle_index=1,
            api_classification_index=1,
            model_confidence=0.91,
            tags=[
                {"name": "taxi", "value": "yes", "score": 0.9},
                {"name": "damaged", "value": "no", "score": 0.8},
            ],
        ),
        8: MMRResult(
            make="Toyota",
            model="Corolla",
            vehicle_index=1,
            api_classification_index=1,
            tags=[{"name": "taxi", "value": "yes"}],
        ),
        9: MMRResult(
            make="Audi",
            model="A4",
            generation="B9",
            variation="Avant",
            vehicle_index=2,
            api_classification_index=2,
            tags=[{"name": "Police Car", "value": True, "score": 0.72}],
        ),
    }
    payload = generate_reports(store)

    assert payload == {
        "report_csv": str(store.report_csv_path),
        "rows": 2,
    }
    assert store.report_csv_path.exists()

    with store.report_csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert rows[0]["vehicle_index"] == "1"
    assert rows[0]["track_id"] == "7"
    assert rows[0]["make"] == "Toyota"
    assert rows[0]["model"] == "Corolla"
    assert rows[0]["generation"] == "E210 (2018)"
    assert rows[0]["variation"] == "Hybrid Touring Sports"
    assert rows[0]["color"] == "white"
    assert rows[0]["accepted"] == "True"
    assert rows[0]["model_confidence"] == "0.91"
    assert "category" not in rows[0]
    assert "view" not in rows[0]
    assert "view8" not in rows[0]
    assert "category_confidence" not in rows[0]
    assert "color_confidence" not in rows[0]
    assert "view_confidence" not in rows[0]
    assert "view8_confidence" not in rows[0]
    assert "detection_confidence" not in rows[0]
    assert "source_image" not in rows[0]
    assert rows[0]["tag_taxi"] == "True"
    assert rows[0]["tag_taxi_confidence"] == "0.9"
    assert rows[0]["tag_police_car"] == ""
    assert rows[0]["tag_police_car_confidence"] == ""
    assert "tag_damaged" not in rows[0]

    assert rows[1]["vehicle_index"] == "2"
    assert rows[1]["track_id"] == "9"
    assert rows[1]["tag_taxi"] == ""
    assert rows[1]["tag_taxi_confidence"] == ""
    assert rows[1]["tag_police_car"] == "True"
    assert rows[1]["tag_police_car_confidence"] == "0.72"
