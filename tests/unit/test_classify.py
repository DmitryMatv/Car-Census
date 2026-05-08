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


def _candidate(
    path: Path,
    track_id: int = 1,
    vehicle_index: int | None = None,
    total_score: float = 10_003.0,
) -> CropCandidate:
    return CropCandidate(
        track_id=track_id,
        vehicle_index=vehicle_index,
        frame_index=1,
        timestamp_seconds=0.1,
        bbox=BBox(x1=0, y1=0, x2=100, y2=100),
        image_path=path,
        sharpness=1.0,
        edge_margin_score=1.0,
        area_score=10_000.0,
        total_score=total_score,
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
    assert labels[1].api_classification_index == 1
    assert labels[2].make == "Toyota"
    assert labels[2].api_classification_index == 2

    payload = orjson.loads(store.labels_path.read_bytes())
    assert payload["1"]["raw"]["frames_seen"] == 4
    assert payload["1"]["api_classification_index"] == 1
    assert payload["2"]["make"] == "Toyota"
    assert payload["2"]["api_classification_index"] == 2


def test_classify_tracks_assigns_api_classification_index(
    tmp_path, monkeypatch
) -> None:
    calls: list[Path] = []

    class FakeTrafficEyeClient:
        def __init__(self, config: AppConfig, cache_dir: Path) -> None:
            pass

        def recognize_vehicle_crop(self, image_path: Path) -> MMRResult:
            calls.append(image_path)
            if image_path.name == "accepted.jpg":
                return MMRResult(
                    make="Toyota",
                    model="Corolla",
                    model_confidence=0.9,
                    accepted=True,
                )
            return MMRResult(
                make="unknown",
                model="unknown",
                model_confidence=0.1,
                accepted=False,
            )

    monkeypatch.setattr(
        "car_census.pipeline.classify.TrafficEyeClient", FakeTrafficEyeClient
    )

    store = DummyRunStore(tmp_path)
    skipped_crop = tmp_path / "skipped.jpg"
    rejected_crop = tmp_path / "rejected.jpg"
    accepted_crop = tmp_path / "accepted.jpg"
    _write_summaries(
        store.tracks_path,
        [
            TrackSummary(
                track_id=1,
                first_frame_index=1,
                last_frame_index=4,
                frames_seen=4,
                max_box_height_px=100,
                candidates=[_candidate(skipped_crop, track_id=1)],
            ),
            TrackSummary(
                track_id=2,
                first_frame_index=1,
                last_frame_index=10,
                frames_seen=10,
                max_box_height_px=100,
                candidates=[_candidate(rejected_crop, track_id=2)],
            ),
            TrackSummary(
                track_id=3,
                first_frame_index=1,
                last_frame_index=10,
                frames_seen=10,
                max_box_height_px=100,
                candidates=[_candidate(accepted_crop, track_id=3)],
            ),
        ],
    )
    config = AppConfig()
    config.analysis.min_track_frames = 10

    labels = classify_tracks(config=config, run_store=store)

    assert calls == [rejected_crop, accepted_crop]
    assert labels[1].api_classification_index == 1
    assert labels[2].api_classification_index == 2
    assert labels[2].accepted is False
    assert labels[3].api_classification_index == 3
    assert labels[3].accepted is True

    payload = orjson.loads(store.labels_path.read_bytes())
    assert payload["1"]["api_classification_index"] == 1
    assert payload["2"]["api_classification_index"] == 2
    assert payload["3"]["api_classification_index"] == 3


def test_classify_tracks_groups_by_vehicle_index_and_sends_one_best_crop(
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
    weak_crop = tmp_path / "weak.jpg"
    best_crop = tmp_path / "best.jpg"
    other_crop = tmp_path / "other.jpg"
    _write_summaries(
        store.tracks_path,
        [
            TrackSummary(
                track_id=10,
                vehicle_index=7,
                first_frame_index=1,
                last_frame_index=5,
                frames_seen=5,
                max_box_height_px=100,
                candidates=[
                    _candidate(
                        weak_crop,
                        track_id=10,
                        vehicle_index=7,
                        total_score=100,
                    )
                ],
            ),
            TrackSummary(
                track_id=11,
                vehicle_index=7,
                first_frame_index=9,
                last_frame_index=13,
                frames_seen=5,
                max_box_height_px=100,
                candidates=[
                    _candidate(
                        best_crop,
                        track_id=11,
                        vehicle_index=7,
                        total_score=200,
                    )
                ],
            ),
            TrackSummary(
                track_id=20,
                vehicle_index=8,
                first_frame_index=20,
                last_frame_index=29,
                frames_seen=10,
                max_box_height_px=100,
                candidates=[
                    _candidate(
                        other_crop,
                        track_id=20,
                        vehicle_index=8,
                        total_score=150,
                    )
                ],
            ),
        ],
    )
    config = AppConfig()
    config.analysis.min_track_frames = 10

    labels = classify_tracks(config=config, run_store=store)

    assert calls == [best_crop, other_crop]
    assert labels[10].vehicle_index == 7
    assert labels[10].api_classification_index == 7
    assert labels[11].vehicle_index == 7
    assert labels[11].api_classification_index == 7
    assert labels[20].vehicle_index == 8
    assert labels[20].api_classification_index == 8

    payload = orjson.loads(store.labels_path.read_bytes())
    assert payload["10"]["vehicle_index"] == 7
    assert payload["11"]["vehicle_index"] == 7
    assert payload["10"]["api_classification_index"] == 7
    assert payload["11"]["api_classification_index"] == 7


def test_classify_tracks_ignores_candidate_less_unqualified_tracks(
    tmp_path, monkeypatch
) -> None:
    calls: list[Path] = []

    class FakeTrafficEyeClient:
        def __init__(self, config: AppConfig, cache_dir: Path) -> None:
            pass

        def recognize_vehicle_crop(self, image_path: Path) -> MMRResult:
            calls.append(image_path)
            return MMRResult(make="Toyota", model="Corolla", accepted=True)

    monkeypatch.setattr(
        "car_census.pipeline.classify.TrafficEyeClient", FakeTrafficEyeClient
    )

    store = DummyRunStore(tmp_path)
    _write_summaries(
        store.tracks_path,
        [
            TrackSummary(
                track_id=10,
                vehicle_index=None,
                first_frame_index=1,
                last_frame_index=10,
                frames_seen=10,
                max_box_height_px=50,
                candidates=[],
            ),
            TrackSummary(
                track_id=20,
                vehicle_index=1,
                first_frame_index=20,
                last_frame_index=29,
                frames_seen=10,
                max_box_height_px=100,
                candidates=[
                    _candidate(
                        tmp_path / "vehicle.jpg",
                        track_id=20,
                        vehicle_index=1,
                        total_score=150,
                    )
                ],
            ),
        ],
    )
    config = AppConfig()
    config.analysis.min_track_frames = 1

    labels = classify_tracks(config=config, run_store=store)

    assert calls == [tmp_path / "vehicle.jpg"]
    assert set(labels) == {20}
    payload = orjson.loads(store.labels_path.read_bytes())
    assert set(payload) == {"20"}
