import csv

from models import MMRResult
from stats.counts import build_vehicle_report_rows, write_vehicle_report_csv


def test_build_vehicle_report_rows_dedupes_shared_vehicle_index() -> None:
    labels = {
        11: MMRResult(make="Audi", model="A4", vehicle_index=7),
        10: MMRResult(
            make="Audi",
            model="A4",
            generation="B9",
            variation="Avant",
            vehicle_index=7,
        ),
        20: MMRResult(make="Toyota", model="Corolla", vehicle_index=8),
    }

    rows = build_vehicle_report_rows(labels)

    assert len(rows) == 2
    assert rows[0]["vehicle_index"] == 7
    assert rows[0]["track_id"] == 10
    assert rows[0]["generation"] == "B9"
    assert rows[1]["vehicle_index"] == 8


def test_build_vehicle_report_rows_adds_only_affirmative_tag_columns() -> None:
    labels = {
        1: MMRResult(
            make="Toyota",
            model="Corolla",
            tags=[
                {"name": "taxi", "value": "yes", "score": 0.91},
                {"name": "damaged", "value": "no"},
            ],
        ),
        2: MMRResult(
            make="Audi",
            model="A4",
            tags=[
                {"name": "Police Car", "value": True},
                {"name": "fleet", "value": 1},
            ],
        ),
    }

    rows = build_vehicle_report_rows(labels)

    assert rows[0]["tag_taxi"] is True
    assert rows[0]["tag_taxi_confidence"] == 0.91
    assert rows[0]["tag_police_car"] == ""
    assert rows[0]["tag_police_car_confidence"] == ""
    assert rows[0]["tag_fleet"] == ""
    assert rows[0]["tag_fleet_confidence"] == ""
    assert "tag_damaged" not in rows[0]
    assert rows[1]["tag_taxi"] == ""
    assert rows[1]["tag_taxi_confidence"] == ""
    assert rows[1]["tag_police_car"] is True
    assert rows[1]["tag_fleet"] is True


def test_write_vehicle_report_csv_uses_detailed_columns(tmp_path) -> None:
    rows = build_vehicle_report_rows(
        {
            1: MMRResult(
                make="Toyota",
                model="Corolla",
                generation="E210",
                variation="Hybrid",
                vehicle_index=1,
                accepted=True,
                tags=[{"name": "taxi", "value": "true", "score": 0.87}],
            )
        }
    )
    path = tmp_path / "report.csv"

    write_vehicle_report_csv(path, rows)

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        csv_rows = list(reader)

    assert reader.fieldnames[:7] == [
        "vehicle_index",
        "track_id",
        "api_classification_index",
        "make",
        "model",
        "generation",
        "variation",
    ]
    assert "color" in reader.fieldnames
    assert "category" not in reader.fieldnames
    assert "view" not in reader.fieldnames
    assert "view8" not in reader.fieldnames
    assert "category_confidence" not in reader.fieldnames
    assert "color_confidence" not in reader.fieldnames
    assert "view_confidence" not in reader.fieldnames
    assert "view8_confidence" not in reader.fieldnames
    assert "detection_confidence" not in reader.fieldnames
    assert "source_image" not in reader.fieldnames
    assert "tag_taxi" in reader.fieldnames
    assert "tag_taxi_confidence" in reader.fieldnames
    assert csv_rows[0]["accepted"] == "True"
    assert csv_rows[0]["tag_taxi"] == "True"
    assert csv_rows[0]["tag_taxi_confidence"] == "0.87"
