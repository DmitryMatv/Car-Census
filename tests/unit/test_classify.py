from pathlib import Path

import cv2
import numpy as np
import orjson

from config import AppConfig
from models import BBox, CropCandidate, MMRResult, TrackSummary
from pipeline.classify import classify_tracks, write_skipped_classification_batch_grids


class DummyRunStore:
    def __init__(self, root: Path) -> None:
        self.crops_dir = root / "crops"
        self.tracks_path = root / "analysis" / "tracks.jsonl"
        self.labels_path = root / "mmr" / "labels.json"
        self.mmr_cache_dir = root / "mmr" / "cache"
        self.mmr_batch_grids_dir = root / "mmr" / "batch_grids"
        self.tracks = self
        self.labels = self
        self.crops_dir.mkdir(parents=True)
        self.tracks_path.parent.mkdir(parents=True)
        self.labels_path.parent.mkdir(parents=True)
        self.mmr_cache_dir.mkdir(parents=True)
        self.mmr_batch_grids_dir.mkdir(parents=True)

    def write_json(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))

    def iter(self):
        if not self.tracks_path.exists():
            return
        for line in self.tracks_path.read_bytes().splitlines():
            if line.strip():
                yield TrackSummary.model_validate(orjson.loads(line))

    def write(self, labels_by_track: dict[int, MMRResult]) -> None:
        self.labels_path.parent.mkdir(parents=True, exist_ok=True)
        self.labels_path.write_bytes(
            orjson.dumps(
                {
                    str(track_id): result.model_dump(mode="json")
                    for track_id, result in labels_by_track.items()
                },
                option=orjson.OPT_INDENT_2,
            )
        )


def _candidate(
    path: Path,
    track_id: int = 1,
    vehicle_index: int | None = None,
    frame_index: int = 1,
    bbox: BBox | None = None,
    vehicle_bbox: BBox | None = None,
    sharpness: float = 1.0,
    edge_margin_score: float = 1.0,
    area_score: float = 10_000.0,
) -> CropCandidate:
    crop_bbox = bbox or BBox(x1=0, y1=0, x2=100, y2=100)
    return CropCandidate(
        track_id=track_id,
        vehicle_index=vehicle_index,
        frame_index=frame_index,
        timestamp_seconds=frame_index / 10,
        bbox=crop_bbox,
        vehicle_bbox=vehicle_bbox,
        image_path=path,
        sharpness=sharpness,
        edge_margin_score=edge_margin_score,
        area_score=area_score,
    )


def _write_summaries(path: Path, summaries: list[TrackSummary]) -> None:
    path.write_bytes(
        b"".join(
            orjson.dumps(summary.model_dump(mode="json")) + b"\n"
            for summary in summaries
        )
    )


def test_legacy_candidate_and_summary_payloads_still_parse(tmp_path) -> None:
    payload = {
        "track_id": 1,
        "vehicle_index": 1,
        "first_frame_index": 1,
        "last_frame_index": 10,
        "frames_seen": 10,
        "max_box_height_px": 100,
        "candidates": [
            {
                "track_id": 1,
                "vehicle_index": 1,
                "frame_index": 1,
                "timestamp_seconds": 0.1,
                "bbox": {"x1": 0, "y1": 0, "x2": 100, "y2": 100},
                "image_path": str(tmp_path / "vehicle.jpg"),
                "sharpness": 1.0,
                "edge_margin_score": 1.0,
                "area_score": 10_000.0,
                "total_score": 123.0,
            }
        ],
    }

    summary = TrackSummary.model_validate(payload)

    assert summary.min_box_width_px is None
    assert len(summary.candidates) == 1
    assert not hasattr(summary.candidates[0], "total_score")


def test_classify_tracks_omits_tracks_below_min_track_frames(
    default_config, tmp_path, monkeypatch
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
                vehicle_index=99,
                first_frame_index=1,
                last_frame_index=4,
                frames_seen=4,
                max_box_width_px=100,
                candidates=[_candidate(short_crop, track_id=1, vehicle_index=99)],
            ),
            TrackSummary(
                track_id=2,
                vehicle_index=1,
                first_frame_index=1,
                last_frame_index=10,
                frames_seen=10,
                max_box_width_px=100,
                candidates=[_candidate(long_crop, track_id=2, vehicle_index=1)],
            ),
        ],
    )
    config = default_config
    config.analysis.min_track_frames = 10
    config.analysis.crop_target_box_range_ratio = 0.5

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
    default_config, tmp_path, monkeypatch
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
                vehicle_index=99,
                first_frame_index=1,
                last_frame_index=4,
                frames_seen=4,
                max_box_width_px=100,
                candidates=[_candidate(skipped_crop, track_id=1, vehicle_index=99)],
            ),
            TrackSummary(
                track_id=2,
                vehicle_index=1,
                first_frame_index=1,
                last_frame_index=10,
                frames_seen=10,
                max_box_width_px=100,
                candidates=[_candidate(rejected_crop, track_id=2, vehicle_index=1)],
            ),
            TrackSummary(
                track_id=3,
                vehicle_index=2,
                first_frame_index=1,
                last_frame_index=10,
                frames_seen=10,
                max_box_width_px=100,
                candidates=[_candidate(accepted_crop, track_id=3, vehicle_index=2)],
            ),
        ],
    )
    config = default_config
    config.analysis.min_track_frames = 10
    config.analysis.crop_target_box_range_ratio = 0.5

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
    default_config, tmp_path, monkeypatch
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
                min_box_width_px=40,
                max_box_width_px=100,
                candidates=[
                    _candidate(
                        weak_crop,
                        track_id=10,
                        vehicle_index=7,
                        vehicle_bbox=BBox(x1=0, y1=0, x2=80, y2=80),
                    )
                ],
            ),
            TrackSummary(
                track_id=11,
                vehicle_index=7,
                first_frame_index=9,
                last_frame_index=13,
                frames_seen=5,
                min_box_width_px=40,
                max_box_width_px=100,
                candidates=[
                    _candidate(
                        best_crop,
                        track_id=11,
                        vehicle_index=7,
                        vehicle_bbox=BBox(x1=0, y1=0, x2=70, y2=70),
                    )
                ],
            ),
            TrackSummary(
                track_id=20,
                vehicle_index=8,
                first_frame_index=20,
                last_frame_index=29,
                frames_seen=10,
                min_box_width_px=40,
                max_box_width_px=100,
                candidates=[
                    _candidate(
                        other_crop,
                        track_id=20,
                        vehicle_index=8,
                    )
                ],
            ),
        ],
    )
    config = default_config
    config.analysis.min_track_frames = 10
    config.analysis.crop_target_box_range_ratio = 0.5

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


def test_classify_tracks_ranks_candidates_with_canonical_tie_breaks(
    default_config, tmp_path, monkeypatch
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
    crops = {
        name: tmp_path / f"{name}.jpg"
        for name in [
            "scale",
            "sharpness",
            "edge",
            "area",
            "earlier",
            "later",
        ]
    }
    _write_summaries(
        store.tracks_path,
        [
            TrackSummary(
                track_id=1,
                vehicle_index=1,
                first_frame_index=1,
                last_frame_index=10,
                frames_seen=10,
                min_box_width_px=50,
                max_box_width_px=100,
                candidates=[
                    _candidate(
                        crops["scale"],
                        track_id=1,
                        vehicle_index=1,
                        vehicle_bbox=BBox(x1=0, y1=0, x2=75, y2=75),
                    )
                ],
            ),
            TrackSummary(
                track_id=2,
                vehicle_index=2,
                first_frame_index=11,
                last_frame_index=20,
                frames_seen=10,
                min_box_width_px=50,
                max_box_width_px=100,
                candidates=[
                    _candidate(
                        crops["sharpness"],
                        track_id=2,
                        vehicle_index=2,
                        sharpness=2.0,
                    ),
                    _candidate(
                        tmp_path / "dull.jpg",
                        track_id=2,
                        vehicle_index=2,
                        sharpness=1.0,
                    ),
                ],
            ),
            TrackSummary(
                track_id=3,
                vehicle_index=3,
                first_frame_index=21,
                last_frame_index=30,
                frames_seen=10,
                min_box_width_px=100,
                max_box_width_px=100,
                candidates=[
                    _candidate(
                        crops["edge"],
                        track_id=3,
                        vehicle_index=3,
                        edge_margin_score=10.0,
                    ),
                    _candidate(
                        tmp_path / "edge-low.jpg",
                        track_id=3,
                        vehicle_index=3,
                        edge_margin_score=1.0,
                    ),
                ],
            ),
            TrackSummary(
                track_id=4,
                vehicle_index=4,
                first_frame_index=31,
                last_frame_index=40,
                frames_seen=10,
                min_box_width_px=100,
                max_box_width_px=100,
                candidates=[
                    _candidate(
                        crops["area"],
                        track_id=4,
                        vehicle_index=4,
                        area_score=20_000.0,
                    ),
                    _candidate(
                        tmp_path / "area-low.jpg",
                        track_id=4,
                        vehicle_index=4,
                        area_score=10_000.0,
                    ),
                ],
            ),
            TrackSummary(
                track_id=5,
                vehicle_index=5,
                first_frame_index=41,
                last_frame_index=50,
                frames_seen=10,
                min_box_width_px=100,
                max_box_width_px=100,
                candidates=[
                    _candidate(
                        crops["later"],
                        track_id=5,
                        vehicle_index=5,
                        frame_index=2,
                    ),
                    _candidate(
                        crops["earlier"],
                        track_id=5,
                        vehicle_index=5,
                        frame_index=1,
                    ),
                ],
            ),
        ],
    )
    config = default_config
    config.analysis.min_track_frames = 1
    config.analysis.crop_target_box_range_ratio = 0.5

    classify_tracks(config=config, run_store=store)

    assert calls == [
        crops["scale"],
        crops["sharpness"],
        crops["edge"],
        crops["area"],
        crops["earlier"],
    ]


def test_classify_tracks_sends_best_crops_in_batches(
    config_factory, tmp_path, monkeypatch
) -> None:
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
                max_box_width_px=100,
                candidates=[_candidate(first_crop, track_id=1, vehicle_index=1)],
            ),
            TrackSummary(
                track_id=2,
                vehicle_index=2,
                first_frame_index=11,
                last_frame_index=20,
                frames_seen=10,
                max_box_width_px=100,
                candidates=[_candidate(second_crop, track_id=2, vehicle_index=2)],
            ),
            TrackSummary(
                track_id=3,
                vehicle_index=3,
                first_frame_index=21,
                last_frame_index=30,
                frames_seen=10,
                max_box_width_px=100,
                candidates=[_candidate(third_crop, track_id=3, vehicle_index=3)],
            ),
        ],
    )
    config = config_factory({"mmr": {"batch_size": 2}})
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
    default_config, tmp_path, monkeypatch
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
                max_box_width_px=100,
                candidates=[_candidate(stale_crop, track_id=1, vehicle_index=1)],
            ),
        ],
    )
    config = default_config
    config.analysis.min_track_frames = 1

    labels = classify_tracks(config=config, run_store=store)

    assert calls == [relocated_crop]
    assert labels[1].make == "Toyota"


def test_classify_tracks_ignores_candidate_less_unqualified_tracks(
    default_config, tmp_path, monkeypatch
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
                max_box_width_px=50,
                candidates=[],
            ),
            TrackSummary(
                track_id=20,
                vehicle_index=1,
                first_frame_index=20,
                last_frame_index=29,
                frames_seen=10,
                max_box_width_px=100,
                candidates=[
                    _candidate(
                        tmp_path / "vehicle.jpg",
                        track_id=20,
                        vehicle_index=1,
                    )
                ],
            ),
        ],
    )
    config = default_config
    config.analysis.min_track_frames = 1

    labels = classify_tracks(config=config, run_store=store)

    assert calls == [tmp_path / "vehicle.jpg"]
    assert set(labels) == {20}
    payload = orjson.loads(store.labels_path.read_bytes())
    assert set(payload) == {"20"}


def test_write_skipped_classification_batch_grids_writes_best_crop_grids(
    config_factory, tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("TRAFFICEYE_API_KEY", raising=False)
    store = DummyRunStore(tmp_path)
    crops = []
    for index in range(3):
        crop = tmp_path / f"crop-{index}.jpg"
        cv2.imwrite(str(crop), np.full((40, 80, 3), index, dtype=np.uint8))
        crops.append(crop)

    _write_summaries(
        store.tracks_path,
        [
            TrackSummary(
                track_id=1,
                vehicle_index=1,
                first_frame_index=1,
                last_frame_index=10,
                frames_seen=10,
                max_box_width_px=100,
                candidates=[_candidate(crops[0], track_id=1, vehicle_index=1)],
            ),
            TrackSummary(
                track_id=2,
                vehicle_index=2,
                first_frame_index=11,
                last_frame_index=20,
                frames_seen=10,
                max_box_width_px=100,
                candidates=[_candidate(crops[1], track_id=2, vehicle_index=2)],
            ),
            TrackSummary(
                track_id=3,
                vehicle_index=3,
                first_frame_index=21,
                last_frame_index=30,
                frames_seen=10,
                max_box_width_px=100,
                candidates=[_candidate(crops[2], track_id=3, vehicle_index=3)],
            ),
        ],
    )
    config = config_factory(
        {"mmr": {"batch_size": 2, "batch_grid_columns": 2, "batch_cell_size_px": 100}}
    )
    config.analysis.min_track_frames = 1

    grid_paths = write_skipped_classification_batch_grids(
        config=config, run_store=store
    )

    assert len(grid_paths) == 2
    assert sorted(path.suffix for path in grid_paths) == [".jpg", ".jpg"]
    assert sorted(path.parent for path in grid_paths) == [
        store.mmr_batch_grids_dir,
        store.mmr_batch_grids_dir,
    ]
    manifests = [path.with_suffix(".json") for path in grid_paths]
    assert all(path.exists() for path in manifests)
    first_manifest = orjson.loads(manifests[0].read_bytes())
    second_manifest = orjson.loads(manifests[1].read_bytes())
    manifest_sources = [
        [cell["source_image"] for cell in manifest["cells"]]
        for manifest in [first_manifest, second_manifest]
    ]
    assert manifest_sources == [[str(crops[0]), str(crops[1])], [str(crops[2])]]
    assert not store.labels_path.exists()
