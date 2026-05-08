from pathlib import Path

import orjson

from car_census.config import AppConfig
from car_census.pipeline.classify import classify_tracks
from car_census.types import BBox, CropCandidate, MMRResult, TrackSummary


class DummyRunStore:
    def __init__(self, root: Path) -> None:
        self.tracks_path = root / "analysis" / "tracks.jsonl"
        self.labels_path = root / "mmr" / "labels.json"
        self.mmr_cache_dir = root / "mmr" / "cache"
        self.tracks_path.parent.mkdir(parents=True)
        self.labels_path.parent.mkdir(parents=True)
        self.mmr_cache_dir.mkdir(parents=True)

    def write_json(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))


def _candidate(path: Path, track_id: int = 1) -> CropCandidate:
    return CropCandidate(
        track_id=track_id,
        frame_index=1,
        timestamp_seconds=0.1,
        bbox=BBox(x1=0, y1=0, x2=100, y2=100),
        image_path=path,
        sharpness=1.0,
        edge_margin_score=1.0,
        area_score=10_000.0,
        total_score=10_003.0,
    )


def _write_summaries(path: Path, summaries: list[TrackSummary]) -> None:
    path.write_bytes(
        b"".join(
            orjson.dumps(summary.model_dump(mode="json")) + b"\n"
            for summary in summaries
        )
    )


def test_classify_tracks_skips_tracks_below_min_track_frames(
    tmp_path, monkeypatch
) -> None:
    calls: list[Path] = []

    class FakeTrafficEyeClient:
        def __init__(self, config: AppConfig, cache_dir: Path) -> None:
            pass

        def recognize_vehicle_crop(self, image_path: Path) -> MMRResult:
            calls.append(image_path)
            return MMRResult(
                make="Toyota",
                model="Corolla",
                model_confidence=0.9,
                accepted=True,
            )

    monkeypatch.setattr(
        "car_census.pipeline.classify.TrafficEyeClient", FakeTrafficEyeClient
    )

    store = DummyRunStore(tmp_path)
    short_crop = tmp_path / "short.jpg"
    long_crop = tmp_path / "long.jpg"
    _write_summaries(
        store.tracks_path,
        [
            TrackSummary(
                track_id=1,
                first_frame_index=1,
                last_frame_index=4,
                frames_seen=4,
                max_box_height_px=100,
                candidates=[_candidate(short_crop, track_id=1)],
            ),
            TrackSummary(
                track_id=2,
                first_frame_index=1,
                last_frame_index=10,
                frames_seen=10,
                max_box_height_px=100,
                candidates=[_candidate(long_crop, track_id=2)],
            ),
        ],
    )
    config = AppConfig()
    config.analysis.min_track_frames = 10

    labels = classify_tracks(config=config, run_store=store)

    assert calls == [long_crop]
    assert labels[1].make == "unknown"
    assert labels[1].raw["skipped_reason"] == "track_too_short"
    assert labels[2].make == "Toyota"

    payload = orjson.loads(store.labels_path.read_bytes())
    assert payload["1"]["raw"]["frames_seen"] == 4
    assert payload["2"]["make"] == "Toyota"
