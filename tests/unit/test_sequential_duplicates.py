import orjson

from config import AppConfig
from models import BBox, FrameRecord, MMRResult, TrackedObject, TrackSummary
from pipeline.report import generate_reports
from pipeline.sequential_duplicates import deduplicate_classified_tracks
from storage.run_store import RunStore


def _track(
    track_id: int,
    vehicle_index: int,
    timestamp_seconds: float,
    bbox: BBox,
) -> TrackedObject:
    frame_index = int(round(timestamp_seconds * 10))
    return TrackedObject(
        track_id=track_id,
        vehicle_index=vehicle_index,
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
        bbox=bbox,
        confidence=0.9,
        class_id=2,
        class_name="car",
        centroid=bbox.center,
        bottom_center=bbox.bottom_center,
        inside_roi=True,
        counted=True,
    )


def _label(
    vehicle_index: int,
    *,
    make: str = "Toyota",
    model: str = "Corolla",
    generation: str = "E210",
    variation: str = "sedan",
    color: str = "white",
) -> MMRResult:
    return MMRResult(
        category="car",
        make=make,
        model=model,
        generation=generation,
        variation=variation,
        color=color,
        accepted=True,
        vehicle_index=vehicle_index,
        api_classification_index=vehicle_index,
    )


def _summary(track_id: int, vehicle_index: int) -> TrackSummary:
    return TrackSummary(
        track_id=track_id,
        vehicle_index=vehicle_index,
        first_frame_index=0,
        last_frame_index=1,
        frames_seen=2,
        max_box_width_px=40,
    )


def _write_run(
    tmp_path,
    *,
    labels: dict[int, MMRResult] | None = None,
    first_b_bbox: BBox | None = None,
    b_start_time: float = 0.30,
) -> RunStore:
    store = RunStore(tmp_path)
    store.ensure_directories()
    a0 = BBox(x1=100, y1=100, x2=140, y2=130)
    a1 = BBox(x1=110, y1=100, x2=150, y2=130)
    b0 = first_b_bbox or BBox(x1=130, y1=100, x2=170, y2=130)
    b1 = BBox(x1=140, y1=100, x2=180, y2=130)
    store.frames.write_all(
        [
            FrameRecord(
                frame_index=0,
                timestamp_seconds=0.0,
                tracks=[_track(1, 1, 0.0, a0)],
            ),
            FrameRecord(
                frame_index=1,
                timestamp_seconds=0.1,
                tracks=[_track(1, 1, 0.1, a1)],
            ),
            FrameRecord(
                frame_index=int(round(b_start_time * 10)),
                timestamp_seconds=b_start_time,
                tracks=[_track(2, 2, b_start_time, b0)],
            ),
            FrameRecord(
                frame_index=int(round((b_start_time + 0.1) * 10)),
                timestamp_seconds=b_start_time + 0.1,
                tracks=[_track(2, 2, b_start_time + 0.1, b1)],
            ),
        ]
    )
    store.tracks.write_all([_summary(1, 1), _summary(2, 2)])
    store.labels.write(labels or {1: _label(1), 2: _label(2)})
    return store


def _config(config_factory, **tracker_overrides: object) -> AppConfig:
    tracker_config: dict[str, object] = {
        "suppress_sequential_duplicate_tracks": True,
        "sequential_duplicate_require_same_color": True,
        "sequential_duplicate_require_same_generation": True,
        "sequential_duplicate_require_same_variation": True,
    }
    tracker_config.update(tracker_overrides)
    return config_factory({"tracker": tracker_config})


def test_sequential_duplicate_merge_rewrites_labels_frames_and_tracks(
    config_factory, tmp_path
) -> None:
    store = _write_run(tmp_path)

    payload = deduplicate_classified_tracks(_config(config_factory), store)

    labels = store.labels.read()
    frames = store.frames.read_all()
    summaries = store.tracks.read_all()
    assert payload["merged_vehicle_count"] == 1
    assert labels[1].vehicle_index == 1
    assert labels[2].vehicle_index == 1
    assert labels[2].api_classification_index == 1
    assert frames[2].tracks[0].vehicle_index == 1
    assert summaries[1].vehicle_index == 1
    assert (store.analysis_dir / "sequential_duplicates.json").exists()


def test_sequential_duplicate_no_merge_when_identity_differs(
    config_factory, tmp_path
) -> None:
    store = _write_run(tmp_path, labels={1: _label(1), 2: _label(2, make="Audi")})

    deduplicate_classified_tracks(_config(config_factory), store)

    assert store.labels.read()[2].vehicle_index == 2


def test_sequential_duplicate_no_merge_when_color_differs_and_required(
    config_factory,
    tmp_path,
) -> None:
    store = _write_run(tmp_path, labels={1: _label(1), 2: _label(2, color="black")})

    deduplicate_classified_tracks(_config(config_factory), store)

    assert store.labels.read()[2].vehicle_index == 2


def test_sequential_duplicate_allows_color_differs_when_not_required(
    config_factory, tmp_path
) -> None:
    store = _write_run(tmp_path, labels={1: _label(1), 2: _label(2, color="black")})

    deduplicate_classified_tracks(
        _config(config_factory, sequential_duplicate_require_same_color=False), store
    )

    assert store.labels.read()[2].vehicle_index == 1


def test_sequential_duplicate_no_merge_when_generation_differs_and_required(
    config_factory,
    tmp_path,
) -> None:
    store = _write_run(tmp_path, labels={1: _label(1), 2: _label(2, generation="E211")})

    deduplicate_classified_tracks(_config(config_factory), store)

    assert store.labels.read()[2].vehicle_index == 2


def test_sequential_duplicate_allows_generation_differs_when_not_required(
    config_factory,
    tmp_path,
) -> None:
    store = _write_run(tmp_path, labels={1: _label(1), 2: _label(2, generation="E211")})

    deduplicate_classified_tracks(
        _config(config_factory, sequential_duplicate_require_same_generation=False),
        store,
    )

    assert store.labels.read()[2].vehicle_index == 1


def test_sequential_duplicate_no_merge_when_variation_differs_and_required(
    config_factory,
    tmp_path,
) -> None:
    store = _write_run(tmp_path, labels={1: _label(1), 2: _label(2, variation="wagon")})

    deduplicate_classified_tracks(_config(config_factory), store)

    assert store.labels.read()[2].vehicle_index == 2


def test_sequential_duplicate_allows_variation_differs_when_not_required(
    config_factory,
    tmp_path,
) -> None:
    store = _write_run(tmp_path, labels={1: _label(1), 2: _label(2, variation="wagon")})

    deduplicate_classified_tracks(
        _config(config_factory, sequential_duplicate_require_same_variation=False),
        store,
    )

    assert store.labels.read()[2].vehicle_index == 1


def test_sequential_duplicate_no_merge_when_gap_exceeds_limit(
    config_factory, tmp_path
) -> None:
    store = _write_run(tmp_path, b_start_time=0.60)

    deduplicate_classified_tracks(_config(config_factory), store)

    assert store.labels.read()[2].vehicle_index == 2


def test_sequential_duplicate_no_merge_when_prediction_or_size_fails(
    config_factory,
    tmp_path,
) -> None:
    store = _write_run(
        tmp_path,
        first_b_bbox=BBox(x1=240, y1=100, x2=300, y2=160),
    )

    deduplicate_classified_tracks(_config(config_factory), store)

    assert store.labels.read()[2].vehicle_index == 2


def test_sequential_duplicate_transitive_chain_uses_earliest_vehicle_index(
    config_factory,
    tmp_path,
) -> None:
    store = _write_run(tmp_path)
    records = store.frames.read_all()
    records.extend(
        [
            FrameRecord(
                frame_index=5,
                timestamp_seconds=0.5,
                tracks=[
                    _track(
                        3,
                        3,
                        0.5,
                        BBox(x1=160, y1=100, x2=200, y2=130),
                    )
                ],
            ),
            FrameRecord(
                frame_index=6,
                timestamp_seconds=0.6,
                tracks=[
                    _track(
                        3,
                        3,
                        0.6,
                        BBox(x1=170, y1=100, x2=210, y2=130),
                    )
                ],
            ),
        ]
    )
    store.frames.write_all(records)
    store.tracks.write_all([_summary(1, 1), _summary(2, 2), _summary(3, 3)])
    store.labels.write({1: _label(1), 2: _label(2), 3: _label(3)})

    deduplicate_classified_tracks(_config(config_factory), store)

    assert {label.vehicle_index for label in store.labels.read().values()} == {1}


def test_report_generation_emits_one_row_after_merged_labels(
    config_factory, tmp_path
) -> None:
    store = _write_run(tmp_path)
    deduplicate_classified_tracks(_config(config_factory), store)

    payload = generate_reports(store)

    rows = list(orjson.loads(orjson.dumps(payload)) for _ in [None])
    assert rows[0]["rows"] == 1


def test_bridge_observations_injected_in_gap_between_merged_tracks(
    config_factory, tmp_path
) -> None:
    store = _write_run(tmp_path, b_start_time=0.30)
    deduplicate_classified_tracks(_config(config_factory), store)

    records = store.frames.read_all()
    track_1_frames = [
        r.frame_index for r in records if any(t.track_id == 1 for t in r.tracks)
    ]
    track_2_frames = [
        r.frame_index for r in records if any(t.track_id == 2 for t in r.tracks)
    ]
    all_frames = sorted({r.frame_index for r in records})

    track_1_max = max(track_1_frames)
    track_2_min = min(track_2_frames)
    gap_frames = [fi for fi in all_frames if track_1_max < fi < track_2_min]

    for fi in gap_frames:
        record = next(r for r in records if r.frame_index == fi)
        track_1_in_gap = [t for t in record.tracks if t.track_id == 1]
        assert len(track_1_in_gap) == 1, (
            f"Expected bridge observation for track 1 at frame {fi}"
        )
        assert track_1_in_gap[0].vehicle_index == 1
