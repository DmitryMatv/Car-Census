import orjson

from pipeline.report import generate_reports
from models import MMRResult, TrackSummary


class DummyRunStore:
    def __init__(self, root):
        self.labels_path = root / "labels.json"
        self.count_events_path = root / "count_events.jsonl"
        self.tracks_path = root / "tracks.jsonl"
        self.counts_json_path = root / "counts.json"
        self.counts_csv_path = root / "counts.csv"


def test_generate_reports_falls_back_to_counted_track_summaries(tmp_path) -> None:
    store = DummyRunStore(tmp_path)
    store.labels_path.write_bytes(
        orjson.dumps(
            {
                "7": MMRResult(
                    make="Toyota",
                    model="Corolla",
                    vehicle_index=1,
                    api_classification_index=1,
                ).model_dump(mode="json")
            }
        )
    )
    store.tracks_path.write_bytes(
        orjson.dumps(
            TrackSummary(
                track_id=7,
                vehicle_index=1,
                first_frame_index=1,
                last_frame_index=3,
                frames_seen=3,
                max_box_height_px=100,
                counted=True,
            ).model_dump(mode="json")
        )
        + b"\n"
    )

    payload = generate_reports(store)

    assert payload["total_counted"] == 1
    assert payload["by_make_model"]["Toyota Corolla"] == 1
