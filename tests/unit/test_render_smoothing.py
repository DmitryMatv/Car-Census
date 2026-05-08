import orjson
import pytest

from car_census.config import AppConfig, build_full_frame_profile
from car_census.pipeline.smooth import smooth_render_tracks
from car_census.storage.run_store import RunStore
from car_census.types import BBox, FrameRecord, RunManifest, TrackedObject


def _track(
    track_id: int, frame_index: int, timestamp: float, bbox: BBox
) -> TrackedObject:
    bottom_center = ((bbox.x1 + bbox.x2) / 2.0, bbox.y2)
    return TrackedObject(
        track_id=track_id,
        frame_index=frame_index,
        timestamp_seconds=timestamp,
        bbox=bbox,
        confidence=0.9,
        class_id=2,
        class_name="car",
        centroid=bbox.center,
        bottom_center=bottom_center,
        inside_roi=True,
    )


def _record(frame_index: int, timestamp: float, tracks=None) -> FrameRecord:
    return FrameRecord(
        frame_index=frame_index,
        timestamp_seconds=timestamp,
        tracks=tracks or [],
    )


def _write_jsonl(path, records) -> bytes:
    payload = b"".join(
        orjson.dumps(record.model_dump(mode="json")) + b"\n" for record in records
    )
    path.write_bytes(payload)
    return payload


def _store(
    tmp_path,
    source_fps: float = 10,
    analysis_fps: float = 10,
) -> RunStore:
    store = RunStore(tmp_path)
    store.ensure_directories()
    store.write_manifest(
        RunManifest(
            run_id="test",
            video_path=tmp_path / "video.mp4",
            camera_id="__full_frame__",
            root_dir=tmp_path,
            source_fps=source_fps,
            analysis_fps=analysis_fps,
            width=200,
            height=100,
        )
    )
    return store


def _read_records(path):
    return [
        FrameRecord.model_validate(orjson.loads(line))
        for line in path.read_bytes().splitlines()
        if line.strip()
    ]


def test_smooth_render_tracks_creates_render_artifact_and_preserves_raw(
    tmp_path,
) -> None:
    store = _store(tmp_path)
    profile = build_full_frame_profile(width=200, height=100)
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(1, 0.1, [_track(1, 1, 0.1, BBox(x1=10, y1=10, x2=30, y2=30))]),
        _record(2, 0.2, [_track(1, 2, 0.2, BBox(x1=20, y1=10, x2=40, y2=30))]),
    ]
    raw_payload = _write_jsonl(store.frames_path, records)

    output_path = smooth_render_tracks(AppConfig(), profile, store)

    assert output_path == store.render_frames_path
    assert store.render_frames_path.exists()
    assert store.frames_path.read_bytes() == raw_payload


def test_smooth_render_tracks_reduces_single_frame_jitter(tmp_path) -> None:
    store = _store(tmp_path)
    profile = build_full_frame_profile(width=200, height=100)
    config = AppConfig.model_validate(
        {
            "render": {
                "smoothing": {
                    "window_seconds": 0.5,
                    "max_center_offset_ratio": 10,
                    "max_size_delta_ratio": 10,
                }
            }
        }
    )
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(1, 0.1, [_track(1, 1, 0.1, BBox(x1=18, y1=10, x2=38, y2=30))]),
        _record(2, 0.2, [_track(1, 2, 0.2, BBox(x1=20, y1=10, x2=40, y2=30))]),
        _record(3, 0.3, [_track(1, 3, 0.3, BBox(x1=30, y1=10, x2=50, y2=30))]),
        _record(4, 0.4, [_track(1, 4, 0.4, BBox(x1=40, y1=10, x2=60, y2=30))]),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(config, profile, store)

    smoothed = _read_records(store.render_frames_path)
    raw_center = records[1].tracks[0].bbox.center[0]
    smoothed_center = smoothed[1].tracks[0].bbox.center[0]
    expected_center = 20.0
    assert abs(smoothed_center - expected_center) < abs(raw_center - expected_center)


def test_smooth_render_tracks_interpolates_short_missing_gap(tmp_path) -> None:
    store = _store(tmp_path)
    profile = build_full_frame_profile(width=200, height=100)
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(1, 0.1),
        _record(2, 0.2, [_track(1, 2, 0.2, BBox(x1=20, y1=10, x2=40, y2=30))]),
        _record(3, 0.3, [_track(1, 3, 0.3, BBox(x1=30, y1=10, x2=50, y2=30))]),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(AppConfig(), profile, store)

    smoothed = _read_records(store.render_frames_path)
    assert [track.track_id for track in smoothed[1].tracks] == [1]
    assert smoothed[1].tracks[0].centroid == smoothed[1].tracks[0].bbox.center
    assert smoothed[1].tracks[0].bottom_center[1] == smoothed[1].tracks[0].bbox.y2


def test_smooth_render_tracks_interpolates_between_downsampled_analysis_frames(
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

    smoothed = _read_records(store.render_frames_path)
    assert [record.frame_index for record in smoothed] == list(range(7))
    assert smoothed[1].tracks[0].bbox.x1 == pytest.approx(10)
    assert smoothed[2].tracks[0].bbox.x1 == pytest.approx(20)
    assert smoothed[4].tracks[0].bbox.x1 == pytest.approx(40)
    assert smoothed[5].tracks[0].bbox.x1 == pytest.approx(50)


def test_smooth_render_tracks_can_interpolate_without_keyframe_smoothing(
    tmp_path,
) -> None:
    store = _store(tmp_path, source_fps=30, analysis_fps=10)
    profile = build_full_frame_profile(width=200, height=100)
    config = AppConfig.model_validate(
        {"render": {"smoothing": {"smooth_keyframes": False}}}
    )
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(3, 0.1, [_track(1, 3, 0.1, BBox(x1=33, y1=10, x2=53, y2=30))]),
        _record(6, 0.2, [_track(1, 6, 0.2, BBox(x1=60, y1=10, x2=80, y2=30))]),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(config, profile, store)

    smoothed = _read_records(store.render_frames_path)
    assert smoothed[1].tracks[0].bbox.x1 == pytest.approx(11)
    assert smoothed[2].tracks[0].bbox.x1 == pytest.approx(22)
    assert smoothed[3].tracks[0].bbox == records[1].tracks[0].bbox


def test_smooth_render_tracks_can_disable_interpolation(tmp_path) -> None:
    store = _store(tmp_path, source_fps=30, analysis_fps=10)
    profile = build_full_frame_profile(width=200, height=100)
    config = AppConfig.model_validate({"render": {"smoothing": {"interpolate": False}}})
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(3, 0.1, [_track(1, 3, 0.1, BBox(x1=30, y1=10, x2=50, y2=30))]),
        _record(6, 0.2, [_track(1, 6, 0.2, BBox(x1=60, y1=10, x2=80, y2=30))]),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(config, profile, store)

    smoothed = _read_records(store.render_frames_path)
    assert [record.frame_index for record in smoothed] == [0, 3, 6]


def test_smooth_render_tracks_does_not_interpolate_large_gap(tmp_path) -> None:
    store = _store(tmp_path)
    profile = build_full_frame_profile(width=200, height=100)
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(1, 0.5),
        _record(2, 1.0, [_track(1, 2, 1.0, BBox(x1=20, y1=10, x2=40, y2=30))]),
        _record(3, 1.1, [_track(1, 3, 1.1, BBox(x1=30, y1=10, x2=50, y2=30))]),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(AppConfig(), profile, store)

    smoothed = _read_records(store.render_frames_path)
    assert smoothed[1].tracks == []


def test_smooth_render_tracks_keeps_short_tracks_unsmoothed(tmp_path) -> None:
    store = _store(tmp_path)
    profile = build_full_frame_profile(width=200, height=100)
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(1, 0.1),
        _record(2, 0.2, [_track(1, 2, 0.2, BBox(x1=20, y1=10, x2=40, y2=30))]),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(AppConfig(), profile, store)

    smoothed = _read_records(store.render_frames_path)
    assert smoothed[1].tracks == []
    assert smoothed[0].tracks[0].bbox == records[0].tracks[0].bbox


def test_smooth_render_tracks_clamps_extreme_center_offset(tmp_path) -> None:
    store = _store(tmp_path)
    profile = build_full_frame_profile(width=200, height=100)
    config = AppConfig.model_validate(
        {"render": {"smoothing": {"max_center_offset_ratio": 0.1}}}
    )
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(1, 0.1, [_track(1, 1, 0.1, BBox(x1=80, y1=10, x2=100, y2=30))]),
        _record(2, 0.2, [_track(1, 2, 0.2, BBox(x1=20, y1=10, x2=40, y2=30))]),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(config, profile, store)

    smoothed = _read_records(store.render_frames_path)
    raw_center = records[1].tracks[0].bbox.center[0]
    smoothed_center = smoothed[1].tracks[0].bbox.center[0]
    assert abs(smoothed_center - raw_center) <= 2.000001
