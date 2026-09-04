from __future__ import annotations

import orjson
import pytest
from config import CameraProfile, HomographyConfig
from models import BBox, FrameRecord, TrackedObject, TrackSummary
from pipeline.link import link_analysis_tracks
from storage.run_store import RunStore

# Test homography: 2000x1200 px -> 200x120 m, so 10 px == 1 m and the world
# position of a bottom center is pixel/10.
PIXELS_PER_METER = 10.0


def _calibrated_profile() -> CameraProfile:
    return CameraProfile(
        camera_id="test",
        polygon={"points": [[0, 0], [2000, 0], [2000, 1200], [0, 1200]]},
        homography=HomographyConfig(
            source_points=[[0, 0], [2000, 0], [2000, 1200], [0, 1200]],
            target_points=[[0.0, 0.0], [200.0, 0.0], [200.0, 120.0], [0.0, 120.0]],
        ),
    )


def _uncalibrated_profile() -> CameraProfile:
    return CameraProfile(
        camera_id="test",
        polygon={"points": [[0, 0], [2000, 0], [2000, 1200], [0, 1200]]},
    )


def _observation(
    track_id: int,
    timestamp_seconds: float,
    x_m: float,
    y_m: float,
    *,
    vehicle_index: int | None = None,
) -> TrackedObject:
    x_px = x_m * PIXELS_PER_METER
    y_px = y_m * PIXELS_PER_METER
    bbox = BBox(x1=x_px - 20, y1=y_px - 40, x2=x_px + 20, y2=y_px)
    return TrackedObject(
        track_id=track_id,
        vehicle_index=vehicle_index,
        frame_index=int(round(timestamp_seconds * 10)),
        timestamp_seconds=timestamp_seconds,
        bbox=bbox,
        confidence=0.9,
        class_id=3,
        class_name="car",
        centroid=bbox.center,
        bottom_center=bbox.bottom_center,
        inside_roi=True,
    )


def _summary(
    track_id: int,
    first_frame_index: int,
    last_frame_index: int,
    frames_seen: int,
    vehicle_index: int | None,
) -> TrackSummary:
    return TrackSummary(
        track_id=track_id,
        vehicle_index=vehicle_index,
        first_frame_index=first_frame_index,
        last_frame_index=last_frame_index,
        frames_seen=frames_seen,
        max_box_width_px=40.0,
    )


def _write_run(
    tmp_path,
    records: list[FrameRecord],
    summaries: list[TrackSummary],
) -> RunStore:
    store = RunStore(tmp_path)
    store.ensure_directories()
    store.frames.write_all(records)
    store.tracks.write_all(summaries)
    return store


def _records_for(
    track_id: int,
    timestamps: list[float],
    *,
    x_m: float,
    y_m: float,
    meters_per_second: float,
    vehicle_index: int | None = None,
) -> list[FrameRecord]:
    """One FrameRecord per timestamp; the vehicle moves at the given world
    speed along +y starting from (x_m, y_m)."""
    records: list[FrameRecord] = []
    for index, timestamp in enumerate(timestamps):
        records.append(
            FrameRecord(
                frame_index=int(round(timestamp * 10)),
                timestamp_seconds=timestamp,
                tracks=[
                    _observation(
                        track_id,
                        timestamp,
                        x_m,
                        y_m + meters_per_second * (timestamp - timestamps[0]),
                        vehicle_index=vehicle_index,
                    )
                ],
            )
        )
    return records


def _grouped_records(record_groups: list[list[FrameRecord]]) -> list[FrameRecord]:
    merged: dict[int, FrameRecord] = {}
    for group in record_groups:
        for record in group:
            existing = merged.get(record.frame_index)
            if existing is None:
                merged[record.frame_index] = record
                continue
            existing.tracks.extend(record.tracks)
    return [merged[frame_index] for frame_index in sorted(merged)]


def _link(tmp_path, config, profile):
    store = RunStore(tmp_path)
    return link_analysis_tracks(config=config, profile=profile, run_store=store), store


def _read_links(store: RunStore) -> dict:
    return orjson.loads(store.links_path.read_bytes())


def test_birth_flicker_chain_merges_into_canonical_vehicle(
    tmp_path, config_factory
) -> None:
    config = config_factory(None)
    profile = _calibrated_profile()
    # Flicker chain: 61 (1 obs), 62 (1 obs), 63 (1 obs), then the mature
    # track 64; the vehicle approaches at 25 m/s along +y in lane x=52 m.
    records = _grouped_records(
        [
            _records_for(61, [1.0], x_m=52.0, y_m=10.0, meters_per_second=25.0),
            _records_for(62, [1.1], x_m=52.0, y_m=12.5, meters_per_second=25.0),
            _records_for(63, [1.2], x_m=52.0, y_m=15.0, meters_per_second=25.0),
            _records_for(
                64,
                [round(1.3 + 0.1 * i, 1) for i in range(30)],
                x_m=52.0,
                y_m=17.5,
                meters_per_second=25.0,
                vehicle_index=5,
            ),
        ]
    )
    summaries = [
        _summary(61, 10, 10, 1, None),
        _summary(62, 11, 11, 1, None),
        _summary(63, 12, 12, 1, None),
        _summary(64, 13, 42, 30, 5),
    ]
    _write_run(tmp_path, records, summaries)
    raw_before = (tmp_path / "analysis" / "tracks.jsonl").read_bytes()

    result, store = _link(tmp_path, config, profile)

    assert result is not None and result.status == "linked"
    assert result.merge_group_count == 1
    assert result.merged_track_count == 4
    linked = {s.track_id: s.vehicle_index for s in store.tracks_effective.read_all()}
    assert linked[61] == 5
    assert linked[62] == 5
    assert linked[63] == 5
    assert linked[64] == 5
    # Raw artifacts untouched.
    assert (tmp_path / "analysis" / "tracks.jsonl").read_bytes() == raw_before
    raw_tracks = {s.track_id: s.vehicle_index for s in store.tracks.read_all()}
    assert raw_tracks == {61: None, 62: None, 63: None, 64: 5}

    payload = _read_links(store)
    assert payload["status"] == "linked"
    assert payload["input_vehicle_index_count"] == 1
    assert payload["output_vehicle_index_count"] == 1
    assert len(payload["merge_groups"]) == 1
    evidence = payload["merge_groups"][0]["evidence"]
    assert len(evidence) == 3
    assert evidence[0]["gap_seconds"] == pytest.approx(0.1)


def test_mid_life_split_merges_and_adopts_earliest_vehicle_index(
    tmp_path, config_factory
) -> None:
    config = config_factory(None)
    profile = _calibrated_profile()
    records = _grouped_records(
        [
            _records_for(
                7,
                [round(10.0 + 0.1 * i, 1) for i in range(6)],
                x_m=30.0,
                y_m=40.0,
                meters_per_second=25.0,
                vehicle_index=2,
            ),
            _records_for(
                8,
                [round(10.7 + 0.1 * i, 1) for i in range(4)],
                x_m=30.0,
                y_m=57.5,
                meters_per_second=25.0,
                vehicle_index=3,
            ),
        ]
    )
    summaries = [
        _summary(7, 100, 105, 6, 2),
        _summary(8, 107, 110, 4, 3),
    ]

    _write_run(tmp_path, records, summaries)
    result, store = _link(tmp_path, config, profile)

    assert result is not None and result.merge_group_count == 1
    linked = {s.track_id: s.vehicle_index for s in store.tracks_effective.read_all()}
    assert linked[7] == 2
    assert linked[8] == 2
    payload = _read_links(store)
    assert payload["merge_groups"][0]["canonical_vehicle_index"] == 2
    assert payload["output_vehicle_index_count"] == 1


def test_platoon_successor_outside_gap_window_is_rejected(
    tmp_path, config_factory
) -> None:
    config = config_factory(None)
    profile = _calibrated_profile()
    # Next car in the same lane arrives 0.5 s later (above max_gap_seconds);
    # at 25 m/s it is 12.5 m behind the predecessor's last position.
    records = _grouped_records(
        [
            _records_for(
                1,
                [round(5.0 + 0.1 * i, 1) for i in range(6)],
                x_m=37.5,
                y_m=100.0,
                meters_per_second=25.0,
                vehicle_index=1,
            ),
            _records_for(
                2,
                [round(5.6 + 0.1 * i, 1) for i in range(6)],
                x_m=37.5,
                y_m=112.5,
                meters_per_second=25.0,
                vehicle_index=2,
            ),
        ]
    )
    summaries = [_summary(1, 50, 55, 6, 1), _summary(2, 56, 61, 6, 2)]

    _write_run(tmp_path, records, summaries)
    result, store = _link(tmp_path, config, profile)

    assert result is not None and result.merge_group_count == 0
    linked = {s.track_id: s.vehicle_index for s in store.tracks_effective.read_all()}
    assert linked == {1: 1, 2: 2}
    assert _read_links(store)["merge_groups"] == []


def test_lateral_offset_gate_rejects_lane_change_pair(tmp_path, config_factory) -> None:
    config = config_factory(None)
    profile = _calibrated_profile()
    records = _grouped_records(
        [
            _records_for(
                1,
                [1.0],
                x_m=37.5,
                y_m=50.0,
                meters_per_second=0.0,
                vehicle_index=1,
            ),
            _records_for(
                2,
                [1.1],
                x_m=43.5,
                y_m=52.0,
                meters_per_second=0.0,
                vehicle_index=2,
            ),
        ]
    )
    summaries = [_summary(1, 10, 10, 1, 1), _summary(2, 11, 11, 1, 2)]

    _write_run(tmp_path, records, summaries)
    result, _ = _link(tmp_path, config, profile)

    assert result is not None and result.merge_group_count == 0


def test_implied_speed_gate_rejects_impossible_handoff(
    tmp_path, config_factory
) -> None:
    config = config_factory(None)
    profile = _calibrated_profile()
    # 8 m displacement across a 0.1 s gap implies 80 m/s.
    records = _grouped_records(
        [
            _records_for(
                1,
                [1.0],
                x_m=37.5,
                y_m=50.0,
                meters_per_second=0.0,
                vehicle_index=1,
            ),
            _records_for(
                2,
                [1.1],
                x_m=37.5,
                y_m=58.0,
                meters_per_second=0.0,
                vehicle_index=2,
            ),
        ]
    )
    summaries = [_summary(1, 10, 10, 1, 1), _summary(2, 11, 11, 1, 2)]

    _write_run(tmp_path, records, summaries)
    result, _ = _link(tmp_path, config, profile)

    assert result is not None and result.merge_group_count == 0


def test_direction_gate_rejects_handoff_against_motion(
    tmp_path, config_factory
) -> None:
    config = config_factory(None)
    profile = _calibrated_profile()
    # Predecessor moves at 25 m/s along +y; the successor appears 2.5 m
    # behind its last position: physically a car it can never be.
    records = _grouped_records(
        [
            _records_for(
                1,
                [0.9, 1.0],
                x_m=37.5,
                y_m=45.0,
                meters_per_second=25.0,
                vehicle_index=1,
            ),
            _records_for(
                2,
                [1.1],
                x_m=37.5,
                y_m=42.5,
                meters_per_second=0.0,
                vehicle_index=2,
            ),
        ]
    )
    summaries = [_summary(1, 9, 10, 2, 1), _summary(2, 11, 11, 1, 2)]

    _write_run(tmp_path, records, summaries)
    result, _ = _link(tmp_path, config, profile)

    assert result is not None and result.merge_group_count == 0


def test_ambiguous_pair_without_mutual_best_stays_unmerged(
    tmp_path, config_factory
) -> None:
    config = config_factory(None)
    profile = _calibrated_profile()
    # Two predecessors end 0.5 m apart; both gate-pass to the same successor,
    # but only the closer one wins the mutual-best contest.
    records = _grouped_records(
        [
            _records_for(
                1,
                [1.0],
                x_m=37.0,
                y_m=50.0,
                meters_per_second=0.0,
                vehicle_index=1,
            ),
            _records_for(
                2,
                [1.0],
                x_m=37.5,
                y_m=50.5,
                meters_per_second=0.0,
                vehicle_index=2,
            ),
            _records_for(
                3,
                [1.1],
                x_m=37.4,
                y_m=52.5,
                meters_per_second=0.0,
                vehicle_index=3,
            ),
        ]
    )
    summaries = [
        _summary(1, 10, 10, 1, 1),
        _summary(2, 10, 10, 1, 2),
        _summary(3, 11, 11, 1, 3),
    ]

    _write_run(tmp_path, records, summaries)
    result, store = _link(tmp_path, config, profile)

    assert result is not None
    assert result.merge_group_count == 1
    assert result.ambiguous_pair_count == 1
    linked = {s.track_id: s.vehicle_index for s in store.tracks_effective.read_all()}
    # The canonical vehicle index is the earlier predecessor's (track 2).
    assert linked[2] == 2
    assert linked[3] == 2
    assert linked[1] == 1
    payload = _read_links(store)
    assert payload["rejected_ambiguous_pairs"][0]["predecessor_track_id"] == 1


def test_without_homography_linking_stays_inert(tmp_path, config_factory) -> None:
    config = config_factory(None)
    records = _records_for(
        1,
        [1.0, 1.1],
        x_m=37.5,
        y_m=50.0,
        meters_per_second=25.0,
        vehicle_index=1,
    )
    summaries = [_summary(1, 10, 11, 2, 1)]
    _write_run(tmp_path, records, summaries)

    result, store = _link(tmp_path, config, _uncalibrated_profile())

    assert result is not None
    assert result.status == "skipped_no_homography"
    assert not store.linked_tracks_path.exists()
    assert _read_links(store)["status"] == "skipped_no_homography"


def test_disabled_linking_writes_nothing(tmp_path, config_factory) -> None:
    config = config_factory({"linking": {"enabled": False}})
    records = _records_for(
        1,
        [1.0, 1.1],
        x_m=37.5,
        y_m=50.0,
        meters_per_second=25.0,
        vehicle_index=1,
    )
    summaries = [_summary(1, 10, 11, 2, 1)]
    _write_run(tmp_path, records, summaries)

    result, store = _link(tmp_path, config, _calibrated_profile())

    assert result is None
    assert not store.linked_tracks_path.exists()
    assert not store.links_path.exists()


def test_tracks_effective_falls_back_to_raw_without_linking(
    tmp_path, config_factory
) -> None:
    records = _records_for(
        1,
        [1.0, 1.1],
        x_m=37.5,
        y_m=50.0,
        meters_per_second=25.0,
        vehicle_index=1,
    )
    summaries = [_summary(1, 10, 11, 2, 1)]
    store = _write_run(tmp_path, records, summaries)

    assert list(store.tracks_effective.read_all()) == summaries


def test_unmergeable_group_without_vehicle_index_stays_none(
    tmp_path, config_factory
) -> None:
    config = config_factory(None)
    profile = _calibrated_profile()
    records = _grouped_records(
        [
            _records_for(1, [1.0], x_m=37.5, y_m=50.0, meters_per_second=25.0),
            _records_for(2, [1.1], x_m=37.5, y_m=52.5, meters_per_second=25.0),
        ]
    )
    summaries = [_summary(1, 10, 10, 1, None), _summary(2, 11, 11, 1, None)]

    _write_run(tmp_path, records, summaries)
    result, store = _link(tmp_path, config, profile)

    assert result is not None and result.merge_group_count == 1
    linked = {s.track_id: s.vehicle_index for s in store.tracks_effective.read_all()}
    assert linked == {1: None, 2: None}
    payload = _read_links(store)
    assert payload["merge_groups"][0]["canonical_vehicle_index"] is None
    assert payload["input_vehicle_index_count"] == 0
