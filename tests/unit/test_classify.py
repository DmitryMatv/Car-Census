from pathlib import Path

import orjson

from config import AppConfig
from models import BBox, CropCandidate, MMRResult, TrackSummary
from pipeline.classify import classify_tracks


class DummyRunStore:
    def __init__(self, root: Path) -> None:
        self.crops_dir = root / "crops"
        self.tracks_path = root / "analysis" / "tracks.jsonl"
        self.labels_path = root / "mmr" / "labels.json"
        self.mmr_cache_dir = root / "mmr" / "cache"
        self.crops_dir.mkdir(parents=True)
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


def test_classify_tracks_omits_tracks_below_min_track_frames(
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

    monkeypatch.setattr("pipeline.classify.TrafficEyeClient", FakeTrafficEyeClient)

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
    assert set(labels) == {2}
    assert labels[2].make == "Toyota"
    assert labels[2].api_classification_index == 1

    payload = orjson.loads(store.labels_path.read_bytes())
    assert set(payload) == {"2"}
    assert payload["2"]["make"] == "Toyota"
    assert payload["2"]["api_classification_index"] == 1


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

    monkeypatch.setattr("pipeline.classify.TrafficEyeClient", FakeTrafficEyeClient)

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
    assert set(labels) == {2, 3}
    assert labels[2].api_classification_index == 1
    assert labels[2].accepted is False
    assert labels[3].api_classification_index == 2
    assert labels[3].accepted is True

    payload = orjson.loads(store.labels_path.read_bytes())
    assert set(payload) == {"2", "3"}
    assert payload["2"]["api_classification_index"] == 1
    assert payload["3"]["api_classification_index"] == 2


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

    monkeypatch.setattr("pipeline.classify.TrafficEyeClient", FakeTrafficEyeClient)

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


def test_classify_tracks_sends_best_crops_in_batches(tmp_path, monkeypatch) -> None:
    calls: list[list[Path]] = []

    class FakeTrafficEyeClient:
        def __init__(self, config: AppConfig, cache_dir: Path) -> None:
            pass

        def recognize_vehicle_crop(self, image_path: Path) -> MMRResult:
            calls.append([image_path])
            return MMRResult(
                make=f"Make {image_path.stem}",
                model=f"Model {image_path.stem}",
                model_confidence=0.9,
                accepted=True,
                source_image=image_path,
            )

        def recognize_vehicle_crops(self, image_paths: list[Path]) -> list[MMRResult]:
            calls.append(image_paths)
            return [
                MMRResult(
                    make=f"Make {image_path.stem}",
                    model=f"Model {image_path.stem}",
                    model_confidence=0.9,
                    accepted=True,
                    source_image=image_path,
                )
                for image_path in image_paths
            ]

    monkeypatch.setattr("pipeline.classify.TrafficEyeClient", FakeTrafficEyeClient)

    store = DummyRunStore(tmp_path)
    first_crop = tmp_path / "first.jpg"
    second_crop = tmp_path / "second.jpg"
    third_crop = tmp_path / "third.jpg"
    _write_summaries(
        store.tracks_path,
        [
            TrackSummary(
                track_id=1,
                vehicle_index=1,
                first_frame_index=1,
                last_frame_index=10,
                frames_seen=10,
                max_box_height_px=100,
                candidates=[_candidate(first_crop, track_id=1, vehicle_index=1)],
            ),
            TrackSummary(
                track_id=2,
                vehicle_index=2,
                first_frame_index=11,
                last_frame_index=20,
                frames_seen=10,
                max_box_height_px=100,
                candidates=[_candidate(second_crop, track_id=2, vehicle_index=2)],
            ),
            TrackSummary(
                track_id=3,
                vehicle_index=3,
                first_frame_index=21,
                last_frame_index=30,
                frames_seen=10,
                max_box_height_px=100,
                candidates=[_candidate(third_crop, track_id=3, vehicle_index=3)],
            ),
        ],
    )
    config = AppConfig.model_validate({"mmr": {"batch_size": 2}})
    config.analysis.min_track_frames = 1

    labels = classify_tracks(config=config, run_store=store)

    assert calls == [[first_crop, second_crop], [third_crop]]
    assert labels[1].make == "Make first"
    assert labels[2].make == "Make second"
    assert labels[3].make == "Make third"
    assert labels[1].api_classification_index == 1
    assert labels[2].api_classification_index == 2
    assert labels[3].api_classification_index == 3


def test_classify_tracks_uses_relocated_crop_when_track_path_is_stale(
    tmp_path, monkeypatch
) -> None:
    calls: list[Path] = []

    class FakeTrafficEyeClient:
        def __init__(self, config: AppConfig, cache_dir: Path) -> None:
            pass

        def recognize_vehicle_crop(self, image_path: Path) -> MMRResult:
            calls.append(image_path)
            return MMRResult(make="Toyota", model="Corolla", accepted=True)

    monkeypatch.setattr("pipeline.classify.TrafficEyeClient", FakeTrafficEyeClient)

    store = DummyRunStore(tmp_path)
    stale_crop = tmp_path / "outputs" / "run" / "crops" / "vehicle_000001.jpg"
    relocated_crop = store.crops_dir / "vehicle_000001.jpg"
    relocated_crop.write_bytes(b"fake image")
    _write_summaries(
        store.tracks_path,
        [
            TrackSummary(
                track_id=1,
                vehicle_index=1,
                first_frame_index=1,
                last_frame_index=10,
                frames_seen=10,
                max_box_height_px=100,
                candidates=[_candidate(stale_crop, track_id=1, vehicle_index=1)],
            ),
        ],
    )
    config = AppConfig()
    config.analysis.min_track_frames = 1

    labels = classify_tracks(config=config, run_store=store)

    assert calls == [relocated_crop]
    assert labels[1].make == "Toyota"


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

    monkeypatch.setattr("pipeline.classify.TrafficEyeClient", FakeTrafficEyeClient)

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
