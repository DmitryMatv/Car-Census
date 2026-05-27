from pathlib import Path

import orjson
import pytest

from config import AppConfig, CameraProfile, PolygonZoneConfig, build_full_frame_profile
from models import BBox, FrameRecord, RunManifest, TrackedObject
from pipeline.smooth import smooth_render_tracks
from storage.run_store import RunStore


def _track(
    track_id: int,
    frame_index: int,
    timestamp: float,
    bbox: BBox,
    *,
    confidence: float = 0.9,
    vehicle_index: int | None = None,
    class_id: int | None = 2,
    class_name: str | None = "car",
    counted: bool = False,
    crossed_line: bool = False,
) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        vehicle_index=vehicle_index,
        frame_index=frame_index,
        timestamp_seconds=timestamp,
        bbox=bbox,
        confidence=confidence,
        class_id=class_id,
        class_name=class_name,
        centroid=bbox.center,
        bottom_center=bbox.bottom_center,
        inside_roi=True,
        counted=counted,
        crossed_line=crossed_line,
    )


def _record(frame_index: int, timestamp: float, tracks=None) -> FrameRecord:
    return FrameRecord(
        frame_index=frame_index,
        timestamp_seconds=timestamp,
        tracks=tracks or [],
    )


def _write_jsonl(path: Path, records: list[FrameRecord]) -> bytes:
    payload = b"".join(
        orjson.dumps(record.model_dump(mode="json")) + b"\n" for record in records
    )
    path.write_bytes(payload)
    return payload


def _store(
    tmp_path,
    source_fps: float = 30,
    analysis_fps: float = 10,
    frame_count: int = 0,
) -> RunStore:
    store = RunStore(tmp_path)
    store.ensure_directories()
    store.manifest.write(
        RunManifest(
            run_id="test",
            video_path=tmp_path / "video.mp4",
            camera_id="__full_frame__",
            root_dir=tmp_path,
            source_fps=source_fps,
            analysis_fps=analysis_fps,
            width=200,
            height=100,
            frame_count=frame_count,
        )
    )
    return store


def _read_records(path: Path) -> list[FrameRecord]:
    return [
        FrameRecord.model_validate(orjson.loads(line))
        for line in path.read_bytes().splitlines()
        if line.strip()
    ]


def _records_by_frame(records: list[FrameRecord]) -> dict[int, FrameRecord]:
    return {record.frame_index: record for record in records}


def test_smooth_render_tracks_creates_render_artifact_and_preserves_raw(
    tmp_path,
) -> None:
    store = _store(tmp_path)
    profile = build_full_frame_profile(width=200, height=100)
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(3, 0.1, [_track(1, 3, 0.1, BBox(x1=10, y1=10, x2=30, y2=30))]),
        _record(6, 0.2, [_track(1, 6, 0.2, BBox(x1=20, y1=10, x2=40, y2=30))]),
    ]
    raw_payload = _write_jsonl(store.frames_path, records)

    output_path = smooth_render_tracks(AppConfig(), profile, store)

    assert output_path == store.render_frames_path
    assert store.render_frames_path.exists()
    assert store.frames_path.read_bytes() == raw_payload


def test_smooth_render_tracks_writes_empty_artifact_for_empty_input(tmp_path) -> None:
    store = _store(tmp_path)
    profile = build_full_frame_profile(width=200, height=100)
    _write_jsonl(store.frames_path, [])

    output_path = smooth_render_tracks(AppConfig(), profile, store)

    assert output_path == store.render_frames_path
    assert _read_records(store.render_frames_path) == []


def test_smooth_render_tracks_averages_real_observations_before_interpolation(
    tmp_path,
) -> None:
    store = _store(tmp_path)
    profile = build_full_frame_profile(width=200, height=100)
    config = AppConfig.model_validate(
        {
            "render": {
                "smoothing": {
                    "observed_box_smoothing": "causal_average",
                    "history_length": 2,
                }
            }
        }
    )
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(3, 0.1, [_track(1, 3, 0.1, BBox(x1=20, y1=20, x2=40, y2=40))]),
        _record(6, 0.2, [_track(1, 6, 0.2, BBox(x1=60, y1=30, x2=80, y2=50))]),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(config, profile, store)

    smoothed = _records_by_frame(_read_records(store.render_frames_path))
    assert smoothed[0].tracks[0].bbox == records[0].tracks[0].bbox
    assert smoothed[3].tracks[0].bbox.x1 == pytest.approx(10)
    assert smoothed[3].tracks[0].bbox.y1 == pytest.approx(15)
    assert smoothed[3].tracks[0].bbox.x2 == pytest.approx(30)
    assert smoothed[3].tracks[0].bbox.y2 == pytest.approx(35)
    assert smoothed[6].tracks[0].bbox.x1 == pytest.approx(40)
    assert smoothed[6].tracks[0].bbox.y1 == pytest.approx(25)


def test_smooth_render_tracks_interpolates_between_causal_average_observations(
    tmp_path,
) -> None:
    store = _store(tmp_path, source_fps=30, analysis_fps=10)
    profile = build_full_frame_profile(width=200, height=100)
    config = AppConfig.model_validate(
        {
            "render": {
                "smoothing": {
                    "observed_box_smoothing": "causal_average",
                    "history_length": 2,
                }
            }
        }
    )
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(3, 0.1, [_track(1, 3, 0.1, BBox(x1=30, y1=10, x2=50, y2=30))]),
        _record(6, 0.2, [_track(1, 6, 0.2, BBox(x1=90, y1=10, x2=110, y2=30))]),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(config, profile, store)

    smoothed = _records_by_frame(_read_records(store.render_frames_path))
    assert list(smoothed) == list(range(7))
    assert smoothed[1].tracks[0].bbox.x1 == pytest.approx(5)
    assert smoothed[2].tracks[0].bbox.x1 == pytest.approx(10)
    assert smoothed[3].tracks[0].bbox.x1 == pytest.approx(15)
    assert smoothed[4].tracks[0].bbox.x1 == pytest.approx(30)
    assert smoothed[5].tracks[0].bbox.x1 == pytest.approx(45)
    assert smoothed[6].tracks[0].bbox.x1 == pytest.approx(60)


def test_smooth_render_tracks_does_not_smooth_interpolated_frames(tmp_path) -> None:
    store = _store(tmp_path, source_fps=30, analysis_fps=10)
    profile = build_full_frame_profile(width=200, height=100)
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(3, 0.1, [_track(1, 3, 0.1, BBox(x1=30, y1=10, x2=50, y2=30))]),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(
        AppConfig.model_validate(
            {
                "render": {
                    "smoothing": {
                        "observed_box_smoothing": "causal_average",
                        "history_length": 2,
                    }
                }
            }
        ),
        profile,
        store,
    )

    smoothed = _records_by_frame(_read_records(store.render_frames_path))
    assert smoothed[1].tracks[0].bbox.x1 == pytest.approx(5)


def test_smooth_render_tracks_can_disable_source_frame_interpolation(
    tmp_path,
) -> None:
    store = _store(tmp_path, source_fps=30, analysis_fps=10)
    profile = build_full_frame_profile(width=200, height=100)
    config = AppConfig.model_validate(
        {"render": {"smoothing": {"interpolate_source_frames": False}}}
    )
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(3, 0.1, [_track(1, 3, 0.1, BBox(x1=30, y1=10, x2=50, y2=30))]),
        _record(6, 0.2, [_track(1, 6, 0.2, BBox(x1=60, y1=10, x2=80, y2=30))]),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(config, profile, store)

    assert [
        record.frame_index for record in _read_records(store.render_frames_path)
    ] == [
        0,
        3,
        6,
    ]


def test_smooth_render_tracks_preserves_semantic_fields(tmp_path) -> None:
    store = _store(tmp_path)
    profile = build_full_frame_profile(width=200, height=100)
    records = [
        _record(
            0,
            0.0,
            [
                _track(
                    1,
                    0,
                    0.0,
                    BBox(x1=0, y1=10, x2=20, y2=30),
                    vehicle_index=7,
                    class_id=5,
                    class_name="bus",
                    counted=True,
                    crossed_line=True,
                )
            ],
        )
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(AppConfig(), profile, store)

    track = _read_records(store.render_frames_path)[0].tracks[0]
    assert track.track_id == 1
    assert track.vehicle_index == 7
    assert track.frame_index == 0
    assert track.timestamp_seconds == 0.0
    assert track.class_id == 5
    assert track.class_name == "bus"
    assert track.counted is True
    assert track.crossed_line is True


def test_smooth_render_tracks_recomputes_geometry_on_interpolated_tracks(
    tmp_path,
) -> None:
    store = _store(tmp_path)
    profile = CameraProfile(
        camera_id="test",
        polygon=PolygonZoneConfig(points=[[0, 0], [25, 0], [25, 25], [0, 25]]),
    )
    records = [
        _record(
            0,
            0.0,
            [
                _track(
                    1,
                    0,
                    0.0,
                    BBox(x1=0, y1=10, x2=20, y2=20),
                    crossed_line=True,
                )
            ],
        ),
        _record(
            3,
            0.1,
            [_track(1, 3, 0.1, BBox(x1=30, y1=20, x2=50, y2=40))],
        ),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(
        AppConfig.model_validate({"render": {"smoothing": {"history_length": 1}}}),
        profile,
        store,
    )

    track = _records_by_frame(_read_records(store.render_frames_path))[2].tracks[0]
    assert track.centroid == track.bbox.center
    assert track.bottom_center == track.bbox.bottom_center
    assert track.inside_roi is False
    assert track.crossed_line is False


def test_smooth_render_tracks_does_not_move_crossing_event_to_interpolated_frames(
    tmp_path,
) -> None:
    store = _store(tmp_path)
    profile = build_full_frame_profile(width=200, height=100)
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(
            3,
            0.1,
            [
                _track(
                    1,
                    3,
                    0.1,
                    BBox(x1=30, y1=10, x2=50, y2=30),
                    crossed_line=True,
                )
            ],
        ),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(
        AppConfig.model_validate({"render": {"smoothing": {"history_length": 1}}}),
        profile,
        store,
    )

    smoothed = _records_by_frame(_read_records(store.render_frames_path))
    assert smoothed[1].tracks[0].crossed_line is False
    assert smoothed[2].tracks[0].crossed_line is False
    assert smoothed[3].tracks[0].crossed_line is True


def test_smooth_render_tracks_marks_interpolated_counted_without_crossing_event(
    tmp_path,
) -> None:
    store = _store(tmp_path)
    profile = build_full_frame_profile(width=200, height=100)
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(
            3,
            0.1,
            [
                _track(
                    1,
                    3,
                    0.1,
                    BBox(x1=30, y1=10, x2=50, y2=30),
                    counted=True,
                    crossed_line=True,
                )
            ],
        ),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(
        AppConfig.model_validate({"render": {"smoothing": {"history_length": 1}}}),
        profile,
        store,
    )

    smoothed = _records_by_frame(_read_records(store.render_frames_path))
    assert smoothed[1].tracks[0].counted is True
    assert smoothed[1].tracks[0].crossed_line is False
    assert smoothed[2].tracks[0].counted is True
    assert smoothed[2].tracks[0].crossed_line is False
    assert smoothed[3].tracks[0].counted is True
    assert smoothed[3].tracks[0].crossed_line is True


def test_smooth_render_tracks_does_not_interpolate_absent_tracks(tmp_path) -> None:
    store = _store(tmp_path)
    profile = build_full_frame_profile(width=200, height=100)
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(3, 0.1, []),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(AppConfig(), profile, store)

    smoothed = _records_by_frame(_read_records(store.render_frames_path))
    assert [track.track_id for track in smoothed[0].tracks] == [1]
    assert smoothed[1].tracks == []
    assert 2 not in smoothed
    assert smoothed[3].tracks == []


def test_smooth_render_tracks_bridges_missing_analysis_frame(tmp_path) -> None:
    store = _store(tmp_path, source_fps=30, analysis_fps=10)
    profile = build_full_frame_profile(width=200, height=100)
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(3, 0.1, []),
        _record(6, 0.2, [_track(1, 6, 0.2, BBox(x1=60, y1=10, x2=80, y2=30))]),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(AppConfig(), profile, store)

    smoothed = _records_by_frame(_read_records(store.render_frames_path))
    assert smoothed[3].tracks[0].track_id == 1
    assert smoothed[3].tracks[0].bbox.x1 == pytest.approx(30)
    assert smoothed[3].tracks[0].crossed_line is False
    assert smoothed[1].tracks[0].bbox.x1 == pytest.approx(10)
    assert smoothed[5].tracks[0].bbox.x1 == pytest.approx(50)


def test_smooth_render_tracks_bridges_three_missing_analysis_frames(
    tmp_path,
) -> None:
    store = _store(tmp_path, source_fps=30, analysis_fps=10)
    profile = build_full_frame_profile(width=200, height=100)
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(3, 0.1, []),
        _record(6, 0.2, []),
        _record(9, 0.3, []),
        _record(12, 0.4, [_track(1, 12, 0.4, BBox(x1=120, y1=10, x2=140, y2=30))]),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(AppConfig(), profile, store)

    smoothed = _records_by_frame(_read_records(store.render_frames_path))
    assert smoothed[3].tracks[0].bbox.x1 == pytest.approx(30)
    assert smoothed[6].tracks[0].bbox.x1 == pytest.approx(60)
    assert smoothed[9].tracks[0].bbox.x1 == pytest.approx(90)


def test_smooth_render_tracks_does_not_bridge_four_missing_analysis_frames(
    tmp_path,
) -> None:
    store = _store(tmp_path, source_fps=30, analysis_fps=10)
    profile = build_full_frame_profile(width=200, height=100)
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(3, 0.1, []),
        _record(6, 0.2, []),
        _record(9, 0.3, []),
        _record(12, 0.4, []),
        _record(15, 0.5, [_track(1, 15, 0.5, BBox(x1=150, y1=10, x2=170, y2=30))]),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(AppConfig(), profile, store)

    smoothed = _records_by_frame(_read_records(store.render_frames_path))
    assert smoothed[3].tracks == []
    assert smoothed[6].tracks == []
    assert smoothed[9].tracks == []
    assert smoothed[12].tracks == []


def test_smooth_render_tracks_can_disable_missing_analysis_frame_bridge(
    tmp_path,
) -> None:
    store = _store(tmp_path, source_fps=30, analysis_fps=10)
    profile = build_full_frame_profile(width=200, height=100)
    config = AppConfig.model_validate(
        {"render": {"smoothing": {"bridge_missing_analysis_frames": False}}}
    )
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(3, 0.1, []),
        _record(6, 0.2, [_track(1, 6, 0.2, BBox(x1=60, y1=10, x2=80, y2=30))]),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(config, profile, store)

    smoothed = _records_by_frame(_read_records(store.render_frames_path))
    assert smoothed[3].tracks == []


def test_smooth_render_tracks_bridges_multiple_tracks_independently_and_sorts(
    tmp_path,
) -> None:
    store = _store(tmp_path, source_fps=30, analysis_fps=10)
    profile = build_full_frame_profile(width=200, height=100)
    records = [
        _record(
            0,
            0.0,
            [
                _track(2, 0, 0.0, BBox(x1=100, y1=10, x2=120, y2=30)),
                _track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30)),
            ],
        ),
        _record(3, 0.1, []),
        _record(
            6,
            0.2,
            [
                _track(2, 6, 0.2, BBox(x1=160, y1=10, x2=180, y2=30)),
                _track(1, 6, 0.2, BBox(x1=60, y1=10, x2=80, y2=30)),
            ],
        ),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(AppConfig(), profile, store)

    tracks = _records_by_frame(_read_records(store.render_frames_path))[3].tracks
    assert [track.track_id for track in tracks] == [1, 2]
    assert tracks[0].bbox.x1 == pytest.approx(30)
    assert tracks[1].bbox.x1 == pytest.approx(130)


def test_smooth_render_tracks_drops_smoother_history_when_track_absent(
    tmp_path,
) -> None:
    store = _store(tmp_path)
    profile = build_full_frame_profile(width=200, height=100)
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(3, 0.1, []),
        _record(6, 0.2, []),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(
        AppConfig.model_validate(
            {
                "render": {
                    "smoothing": {
                        "observed_box_smoothing": "causal_average",
                        "history_length": 5,
                    }
                }
            }
        ),
        profile,
        store,
    )

    smoothed = _records_by_frame(_read_records(store.render_frames_path))
    assert [track.track_id for track in smoothed[0].tracks] == [1]
    assert smoothed[1].tracks == []
    assert smoothed[3].tracks == []
    assert smoothed[6].tracks == []
    assert all(
        1 not in {track.track_id for track in record.tracks}
        for frame_index, record in smoothed.items()
        if frame_index > 0
    )


def test_smooth_render_tracks_does_not_tail_extrapolate_after_last_observation(
    tmp_path,
) -> None:
    store = _store(tmp_path, source_fps=30, analysis_fps=10, frame_count=10)
    profile = build_full_frame_profile(width=200, height=100)
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(3, 0.1, [_track(1, 3, 0.1, BBox(x1=30, y1=10, x2=50, y2=30))]),
        _record(6, 0.2, [_track(1, 6, 0.2, BBox(x1=60, y1=10, x2=80, y2=30))]),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(AppConfig(), profile, store)

    assert [
        record.frame_index for record in _read_records(store.render_frames_path)
    ] == list(range(7))


def test_smooth_render_tracks_does_not_bridge_large_gaps_by_default(
    tmp_path,
) -> None:
    store = _store(tmp_path, source_fps=30, analysis_fps=10)
    profile = build_full_frame_profile(width=200, height=100)
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(12, 0.4, [_track(1, 12, 0.4, BBox(x1=120, y1=10, x2=140, y2=30))]),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(AppConfig(), profile, store)

    smoothed = _records_by_frame(_read_records(store.render_frames_path))
    assert list(smoothed) == [0, 1, 12]
    assert smoothed[1].tracks == []


def test_smooth_render_tracks_uses_cadence_fallback_when_gap_config_is_null(
    tmp_path,
) -> None:
    store = _store(tmp_path, source_fps=30, analysis_fps=10)
    profile = build_full_frame_profile(width=200, height=100)
    config = AppConfig.model_validate(
        {
            "render": {
                "smoothing": {
                    "history_length": 1,
                    "max_interpolation_gap_seconds": None,
                }
            }
        }
    )
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(3, 0.1, [_track(1, 3, 0.1, BBox(x1=30, y1=10, x2=50, y2=30))]),
        _record(12, 0.4, [_track(1, 12, 0.4, BBox(x1=120, y1=10, x2=140, y2=30))]),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(config, profile, store)

    smoothed = _records_by_frame(_read_records(store.render_frames_path))
    assert list(smoothed) == [0, 1, 2, 3, 4, 12]
    assert smoothed[1].tracks[0].bbox.x1 == pytest.approx(10)
    assert smoothed[2].tracks[0].bbox.x1 == pytest.approx(20)
    assert smoothed[4].tracks == []


def test_smooth_render_tracks_uses_explicit_max_gap_override(tmp_path) -> None:
    store = _store(tmp_path, source_fps=30, analysis_fps=10)
    profile = build_full_frame_profile(width=200, height=100)
    config = AppConfig.model_validate(
        {
            "render": {
                "smoothing": {
                    "history_length": 1,
                    "max_interpolation_gap_seconds": 0.5,
                }
            }
        }
    )
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(12, 0.4, [_track(1, 12, 0.4, BBox(x1=120, y1=10, x2=140, y2=30))]),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(config, profile, store)

    smoothed = _records_by_frame(_read_records(store.render_frames_path))
    assert list(smoothed) == list(range(13))
    assert smoothed[6].tracks[0].bbox.x1 == pytest.approx(60)


def test_smooth_render_tracks_interpolates_multiple_tracks_independently_and_sorts(
    tmp_path,
) -> None:
    store = _store(tmp_path)
    profile = build_full_frame_profile(width=200, height=100)
    records = [
        _record(
            0,
            0.0,
            [
                _track(2, 0, 0.0, BBox(x1=100, y1=10, x2=120, y2=30)),
                _track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30)),
            ],
        ),
        _record(
            3,
            0.1,
            [
                _track(2, 3, 0.1, BBox(x1=130, y1=10, x2=150, y2=30)),
                _track(1, 3, 0.1, BBox(x1=30, y1=10, x2=50, y2=30)),
            ],
        ),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(
        AppConfig.model_validate({"render": {"smoothing": {"history_length": 1}}}),
        profile,
        store,
    )

    smoothed = _records_by_frame(_read_records(store.render_frames_path))
    assert [track.track_id for track in smoothed[1].tracks] == [1, 2]
    assert smoothed[1].tracks[0].bbox.x1 == pytest.approx(10)
    assert smoothed[1].tracks[1].bbox.x1 == pytest.approx(110)


def test_smooth_render_tracks_history_length_one_preserves_observed_boxes(
    tmp_path,
) -> None:
    store = _store(tmp_path)
    profile = build_full_frame_profile(width=200, height=100)
    config = AppConfig.model_validate({"render": {"smoothing": {"history_length": 1}}})
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(3, 0.1, [_track(1, 3, 0.1, BBox(x1=70, y1=10, x2=90, y2=30))]),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(config, profile, store)

    smoothed = _records_by_frame(_read_records(store.render_frames_path))
    assert smoothed[0].tracks[0].bbox == records[0].tracks[0].bbox
    assert smoothed[3].tracks[0].bbox == records[1].tracks[0].bbox


def test_smooth_render_tracks_default_preserves_observed_boxes_and_interpolates(
    tmp_path,
) -> None:
    store = _store(tmp_path, source_fps=30, analysis_fps=10)
    profile = build_full_frame_profile(width=200, height=100)
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(3, 0.1, [_track(1, 3, 0.1, BBox(x1=30, y1=10, x2=50, y2=30))]),
        _record(6, 0.2, [_track(1, 6, 0.2, BBox(x1=60, y1=10, x2=80, y2=30))]),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(AppConfig(), profile, store)

    smoothed = _records_by_frame(_read_records(store.render_frames_path))
    assert smoothed[0].tracks[0].bbox == records[0].tracks[0].bbox
    assert smoothed[3].tracks[0].bbox == records[1].tracks[0].bbox
    assert smoothed[6].tracks[0].bbox == records[2].tracks[0].bbox
    assert smoothed[1].tracks[0].bbox.x1 == pytest.approx(10)
    assert smoothed[2].tracks[0].bbox.x1 == pytest.approx(20)
    assert smoothed[4].tracks[0].bbox.x1 == pytest.approx(40)
    assert smoothed[5].tracks[0].bbox.x1 == pytest.approx(50)


def test_smooth_render_tracks_history_length_without_causal_average_preserves_observed_boxes(
    tmp_path,
) -> None:
    store = _store(tmp_path, source_fps=30, analysis_fps=10)
    profile = build_full_frame_profile(width=200, height=100)
    config = AppConfig.model_validate({"render": {"smoothing": {"history_length": 5}}})
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(3, 0.1, [_track(1, 3, 0.1, BBox(x1=30, y1=10, x2=50, y2=30))]),
        _record(6, 0.2, [_track(1, 6, 0.2, BBox(x1=60, y1=10, x2=80, y2=30))]),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(config, profile, store)

    smoothed = _records_by_frame(_read_records(store.render_frames_path))
    assert smoothed[0].tracks[0].bbox == records[0].tracks[0].bbox
    assert smoothed[3].tracks[0].bbox == records[1].tracks[0].bbox
    assert smoothed[6].tracks[0].bbox == records[2].tracks[0].bbox
    assert smoothed[1].tracks[0].bbox.x1 == pytest.approx(10)
    assert smoothed[2].tracks[0].bbox.x1 == pytest.approx(20)


def test_smooth_render_tracks_returns_raw_path_when_disabled(tmp_path) -> None:
    store = _store(tmp_path)
    profile = build_full_frame_profile(width=200, height=100)
    _write_jsonl(
        store.frames_path,
        [_record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))])],
    )
    config = AppConfig.model_validate({"render": {"smoothing": {"enabled": False}}})

    output_path = smooth_render_tracks(config, profile, store)

    assert output_path == store.frames_path
    assert not store.render_frames_path.exists()
