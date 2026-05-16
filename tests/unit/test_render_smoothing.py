from pathlib import Path

import orjson
import pytest

from config import AppConfig, build_full_frame_profile
from models import BBox, FrameRecord, RunManifest, TrackedObject
from pipeline.smooth import smooth_render_tracks
from storage.run_store import RunStore


def _track(
    track_id: int, frame_index: int, timestamp: float, bbox: BBox
) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        frame_index=frame_index,
        timestamp_seconds=timestamp,
        bbox=bbox,
        confidence=0.9,
        class_id=2,
        class_name="car",
        centroid=bbox.center,
        bottom_center=bbox.bottom_center,
        inside_roi=True,
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
    source_fps: float = 10,
    analysis_fps: float = 10,
    frame_count: int = 0,
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
                    "interpolation_method": "linear",
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
    config = AppConfig.model_validate(
        {"render": {"smoothing": {"interpolation_method": "linear"}}}
    )
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(3, 0.1, [_track(1, 3, 0.1, BBox(x1=30, y1=10, x2=50, y2=30))]),
        _record(6, 0.2, [_track(1, 6, 0.2, BBox(x1=60, y1=10, x2=80, y2=30))]),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(config, profile, store)

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
        {
            "render": {
                "smoothing": {
                    "interpolation_method": "linear",
                    "smooth_keyframes": False,
                }
            }
        }
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


def test_polynomial_interpolation_preserves_keyframes(tmp_path) -> None:
    store = _store(tmp_path, source_fps=30, analysis_fps=10)
    profile = build_full_frame_profile(width=200, height=100)
    config = AppConfig.model_validate(
        {
            "render": {
                "smoothing": {
                    "interpolation_method": "polynomial",
                    "polynomial_order": 2,
                }
            }
        }
    )
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(3, 0.1, [_track(1, 3, 0.1, BBox(x1=30, y1=10, x2=50, y2=30))]),
        _record(6, 0.2, [_track(1, 6, 0.2, BBox(x1=60, y1=10, x2=80, y2=30))]),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(config, profile, store)

    smoothed = _read_records(store.render_frames_path)
    assert [record.frame_index for record in smoothed] == list(range(7))
    assert smoothed[0].tracks[0].bbox == records[0].tracks[0].bbox
    assert smoothed[3].tracks[0].bbox == records[1].tracks[0].bbox
    assert smoothed[6].tracks[0].bbox == records[2].tracks[0].bbox


def test_polynomial_interpolation_follows_curved_keyframes(tmp_path) -> None:
    store = _store(tmp_path, source_fps=30, analysis_fps=10)
    profile = build_full_frame_profile(width=200, height=100)
    config = AppConfig.model_validate(
        {
            "render": {
                "smoothing": {
                    "interpolation_method": "polynomial",
                    "polynomial_order": 2,
                    "max_center_offset_ratio": 10,
                }
            }
        }
    )
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(3, 0.1, [_track(1, 3, 0.1, BBox(x1=9, y1=10, x2=29, y2=30))]),
        _record(6, 0.2, [_track(1, 6, 0.2, BBox(x1=36, y1=10, x2=56, y2=30))]),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(config, profile, store)

    smoothed = _read_records(store.render_frames_path)
    assert smoothed[1].tracks[0].bbox.x1 == pytest.approx(1)
    assert smoothed[1].tracks[0].bbox.x1 != pytest.approx(3)


def test_polynomial_order_three_is_accepted_and_preserves_keyframes(tmp_path) -> None:
    store = _store(tmp_path, source_fps=30, analysis_fps=10)
    profile = build_full_frame_profile(width=200, height=100)
    config = AppConfig.model_validate(
        {
            "render": {
                "smoothing": {
                    "interpolation_method": "polynomial",
                    "polynomial_order": 3,
                }
            }
        }
    )
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(3, 0.1, [_track(1, 3, 0.1, BBox(x1=30, y1=10, x2=50, y2=30))]),
        _record(6, 0.2, [_track(1, 6, 0.2, BBox(x1=60, y1=10, x2=80, y2=30))]),
        _record(9, 0.3, [_track(1, 9, 0.3, BBox(x1=90, y1=10, x2=110, y2=30))]),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(config, profile, store)

    smoothed = _read_records(store.render_frames_path)
    assert [record.frame_index for record in smoothed] == list(range(10))
    assert smoothed[0].tracks[0].bbox == records[0].tracks[0].bbox
    assert smoothed[3].tracks[0].bbox == records[1].tracks[0].bbox
    assert smoothed[6].tracks[0].bbox == records[2].tracks[0].bbox
    assert smoothed[9].tracks[0].bbox == records[3].tracks[0].bbox


def test_polynomial_order_three_falls_back_near_track_edges(tmp_path) -> None:
    store = _store(tmp_path, source_fps=30, analysis_fps=10)
    profile = build_full_frame_profile(width=200, height=100)
    config = AppConfig.model_validate(
        {
            "render": {
                "smoothing": {
                    "interpolation_method": "polynomial",
                    "polynomial_order": 3,
                }
            }
        }
    )
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(3, 0.1, [_track(1, 3, 0.1, BBox(x1=30, y1=10, x2=50, y2=30))]),
        _record(6, 0.2, [_track(1, 6, 0.2, BBox(x1=60, y1=10, x2=80, y2=30))]),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(config, profile, store)

    smoothed = _read_records(store.render_frames_path)
    assert smoothed[0].tracks[0].bbox == records[0].tracks[0].bbox
    assert smoothed[3].tracks[0].bbox == records[1].tracks[0].bbox
    assert smoothed[6].tracks[0].bbox == records[2].tracks[0].bbox
    assert smoothed[1].tracks[0].bbox.width > 0
    assert smoothed[1].tracks[0].bbox.height > 0


def test_polynomial_interpolation_does_not_fill_large_gap(tmp_path) -> None:
    store = _store(tmp_path)
    profile = build_full_frame_profile(width=200, height=100)
    config = AppConfig.model_validate(
        {"render": {"smoothing": {"interpolation_method": "polynomial"}}}
    )
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(1, 0.5),
        _record(2, 1.0, [_track(1, 2, 1.0, BBox(x1=20, y1=10, x2=40, y2=30))]),
        _record(3, 1.1, [_track(1, 3, 1.1, BBox(x1=30, y1=10, x2=50, y2=30))]),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(config, profile, store)

    smoothed = _read_records(store.render_frames_path)
    assert smoothed[1].tracks == []


def test_polynomial_interpolation_clamps_overshoot_to_linear_reference(
    tmp_path,
) -> None:
    store = _store(tmp_path, source_fps=30, analysis_fps=10)
    profile = build_full_frame_profile(width=2000, height=100)
    config = AppConfig.model_validate(
        {
            "render": {
                "smoothing": {
                    "interpolation_method": "polynomial",
                    "polynomial_order": 3,
                    "max_center_offset_ratio": 0.1,
                    "reject_short_excursions": False,
                }
            }
        }
    )
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(3, 0.1, [_track(1, 3, 0.1, BBox(x1=990, y1=10, x2=1010, y2=30))]),
        _record(6, 0.2, [_track(1, 6, 0.2, BBox(x1=-1010, y1=10, x2=-990, y2=30))]),
        _record(9, 0.3, [_track(1, 9, 0.3, BBox(x1=0, y1=10, x2=20, y2=30))]),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(config, profile, store)

    smoothed = _read_records(store.render_frames_path)
    linear_center_x = 10 + ((1000 - 10) / 3)
    smoothed_center_x = smoothed[1].tracks[0].bbox.center[0]
    assert abs(smoothed_center_x - linear_center_x) <= 2.000001
    assert smoothed[1].tracks[0].bbox.width > 0
    assert smoothed[1].tracks[0].bbox.height > 0


def test_polynomial_interpolation_does_not_extrapolate_after_track_ends(
    tmp_path,
) -> None:
    store = _store(tmp_path, source_fps=30, analysis_fps=10)
    profile = build_full_frame_profile(width=200, height=100)
    config = AppConfig.model_validate(
        {"render": {"smoothing": {"interpolation_method": "polynomial"}}}
    )
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(3, 0.1, [_track(1, 3, 0.1, BBox(x1=30, y1=10, x2=50, y2=30))]),
        _record(6, 0.2, [_track(1, 6, 0.2, BBox(x1=60, y1=10, x2=80, y2=30))]),
        _record(9, 0.3),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(config, profile, store)

    smoothed = _read_records(store.render_frames_path)
    assert [record.frame_index for record in smoothed] == list(range(10))
    assert [track.track_id for track in smoothed[5].tracks] == [1]
    assert smoothed[6].tracks[0].bbox == records[2].tracks[0].bbox
    assert smoothed[7].tracks == []
    assert smoothed[8].tracks == []
    assert smoothed[9].tracks == []


def test_polynomial_interpolation_clamps_final_gap_and_preserves_last_keyframe(
    tmp_path,
) -> None:
    store = _store(tmp_path, source_fps=30, analysis_fps=10)
    profile = build_full_frame_profile(width=300, height=100)
    config = AppConfig.model_validate(
        {
            "render": {
                "smoothing": {
                    "interpolation_method": "polynomial",
                    "polynomial_order": 3,
                    "max_center_offset_ratio": 0.1,
                }
            }
        }
    )
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=10, y1=10, x2=30, y2=30))]),
        _record(3, 0.1, [_track(1, 3, 0.1, BBox(x1=110, y1=10, x2=130, y2=30))]),
        _record(6, 0.2, [_track(1, 6, 0.2, BBox(x1=20, y1=10, x2=40, y2=30))]),
        _record(9, 0.3, [_track(1, 9, 0.3, BBox(x1=190, y1=10, x2=210, y2=30))]),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(config, profile, store)

    smoothed = _read_records(store.render_frames_path)
    linear_center_x = 30 + ((200 - 30) / 3)
    smoothed_center_x = smoothed[7].tracks[0].bbox.center[0]
    assert abs(smoothed_center_x - linear_center_x) <= 2.000001
    assert smoothed[9].tracks[0].bbox == records[3].tracks[0].bbox


def test_hermite_interpolation_preserves_keyframes(tmp_path) -> None:
    store = _store(tmp_path, source_fps=30, analysis_fps=10)
    profile = build_full_frame_profile(width=200, height=100)
    config = AppConfig.model_validate(
        {"render": {"smoothing": {"interpolation_method": "hermite"}}}
    )
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(3, 0.1, [_track(1, 3, 0.1, BBox(x1=30, y1=10, x2=50, y2=30))]),
        _record(6, 0.2, [_track(1, 6, 0.2, BBox(x1=60, y1=10, x2=80, y2=30))]),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(config, profile, store)

    smoothed = _read_records(store.render_frames_path)
    assert [record.frame_index for record in smoothed] == list(range(7))
    assert smoothed[0].tracks[0].bbox == records[0].tracks[0].bbox
    assert smoothed[3].tracks[0].bbox == records[1].tracks[0].bbox
    assert smoothed[6].tracks[0].bbox == records[2].tracks[0].bbox


def test_hermite_interpolation_does_not_extrapolate_after_track_ends(
    tmp_path,
) -> None:
    store = _store(tmp_path, source_fps=30, analysis_fps=10)
    profile = build_full_frame_profile(width=200, height=100)
    config = AppConfig.model_validate(
        {"render": {"smoothing": {"interpolation_method": "hermite"}}}
    )
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(3, 0.1, [_track(1, 3, 0.1, BBox(x1=30, y1=10, x2=50, y2=30))]),
        _record(6, 0.2, [_track(1, 6, 0.2, BBox(x1=60, y1=10, x2=80, y2=30))]),
        _record(9, 0.3),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(config, profile, store)

    smoothed = _read_records(store.render_frames_path)
    assert [record.frame_index for record in smoothed] == list(range(10))
    assert [track.track_id for track in smoothed[5].tracks] == [1]
    assert smoothed[6].tracks[0].bbox == records[2].tracks[0].bbox
    assert smoothed[7].tracks == []
    assert smoothed[8].tracks == []
    assert smoothed[9].tracks == []


def test_hermite_interpolation_reduces_overshoot_on_turning_motion(
    tmp_path,
) -> None:
    store = _store(tmp_path, source_fps=30, analysis_fps=10)
    profile = build_full_frame_profile(width=200, height=100)
    config = AppConfig.model_validate(
        {
            "render": {
                "smoothing": {
                    "interpolation_method": "hermite",
                    "max_center_offset_ratio": 10,
                    "reject_short_excursions": False,
                }
            }
        }
    )
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(3, 0.1, [_track(1, 3, 0.1, BBox(x1=90, y1=10, x2=110, y2=30))]),
        _record(6, 0.2, [_track(1, 6, 0.2, BBox(x1=10, y1=10, x2=30, y2=30))]),
        _record(9, 0.3, [_track(1, 9, 0.3, BBox(x1=30, y1=10, x2=50, y2=30))]),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(config, profile, store)

    smoothed = _read_records(store.render_frames_path)
    hermite_x1 = smoothed[4].tracks[0].bbox.x1
    assert 10 <= hermite_x1 <= 90


def test_hermite_uses_smooth_curve_not_linear_for_accelerating_motion(
    tmp_path,
) -> None:
    store = _store(tmp_path, source_fps=30, analysis_fps=10)
    profile = build_full_frame_profile(width=200, height=100)
    config = AppConfig.model_validate(
        {
            "render": {
                "smoothing": {
                    "interpolation_method": "hermite",
                    "max_center_offset_ratio": 10,
                }
            }
        }
    )
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(3, 0.1, [_track(1, 3, 0.1, BBox(x1=9, y1=10, x2=29, y2=30))]),
        _record(6, 0.2, [_track(1, 6, 0.2, BBox(x1=36, y1=10, x2=56, y2=30))]),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(config, profile, store)

    smoothed = _read_records(store.render_frames_path)
    assert smoothed[1].tracks[0].bbox.x1 != pytest.approx(3)
    assert smoothed[1].tracks[0].bbox.width > 0
    assert smoothed[1].tracks[0].bbox.height > 0


def test_hermite_interpolation_respects_max_gap_seconds(tmp_path) -> None:
    store = _store(tmp_path)
    profile = build_full_frame_profile(width=200, height=100)
    config = AppConfig.model_validate(
        {"render": {"smoothing": {"interpolation_method": "hermite"}}}
    )
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(1, 0.5),
        _record(2, 1.0, [_track(1, 2, 1.0, BBox(x1=20, y1=10, x2=40, y2=30))]),
        _record(3, 1.1, [_track(1, 3, 1.1, BBox(x1=30, y1=10, x2=50, y2=30))]),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(config, profile, store)

    smoothed = _read_records(store.render_frames_path)
    assert smoothed[1].tracks == []


def test_hermite_interpolation_clamps_box_size_and_center(tmp_path) -> None:
    store = _store(tmp_path, source_fps=30, analysis_fps=10)
    profile = build_full_frame_profile(width=2000, height=2000)
    config = AppConfig.model_validate(
        {
            "render": {
                "smoothing": {
                    "interpolation_method": "hermite",
                    "max_center_offset_ratio": 0.1,
                    "max_size_delta_ratio": 0.1,
                    "reject_short_excursions": False,
                }
            }
        }
    )
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(3, 0.1, [_track(1, 3, 0.1, BBox(x1=990, y1=10, x2=1010, y2=30))]),
        _record(6, 0.2, [_track(1, 6, 0.2, BBox(x1=-1010, y1=10, x2=-990, y2=30))]),
        _record(9, 0.3, [_track(1, 9, 0.3, BBox(x1=0, y1=10, x2=20, y2=30))]),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(config, profile, store)

    smoothed = _read_records(store.render_frames_path)
    linear_center_x = 10 + ((1000 - 10) / 3)
    smoothed_bbox = smoothed[1].tracks[0].bbox
    assert abs(smoothed_bbox.center[0] - linear_center_x) <= 2.000001
    assert smoothed_bbox.width > 0
    assert smoothed_bbox.height > 0


def test_hermite_handles_two_keyframes_as_linear_like_curve(tmp_path) -> None:
    store = _store(tmp_path, source_fps=30, analysis_fps=10)
    profile = build_full_frame_profile(width=200, height=100)
    config = AppConfig.model_validate(
        {
            "render": {
                "smoothing": {
                    "interpolation_method": "hermite",
                    "min_observations": 2,
                }
            }
        }
    )
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(3, 0.1, [_track(1, 3, 0.1, BBox(x1=30, y1=10, x2=50, y2=30))]),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(config, profile, store)

    smoothed = _read_records(store.render_frames_path)
    assert smoothed[1].tracks[0].bbox.x1 == pytest.approx(10)
    assert smoothed[2].tracks[0].bbox.x1 == pytest.approx(20)


def test_smooth_render_tracks_extrapolates_final_partial_interval(tmp_path) -> None:
    store = _store(tmp_path, source_fps=30, analysis_fps=6, frame_count=10)
    profile = build_full_frame_profile(width=200, height=100)
    config = AppConfig.model_validate(
        {"render": {"smoothing": {"min_observations": 2}}}
    )
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(
            5,
            5 / 30,
            [_track(1, 5, 5 / 30, BBox(x1=50, y1=10, x2=70, y2=30))],
        ),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(config, profile, store)

    smoothed = _read_records(store.render_frames_path)
    assert smoothed[-1].frame_index == 9
    assert smoothed[5].tracks[0].bbox.x1 == pytest.approx(50)
    assert smoothed[6].tracks[0].bbox.x1 > smoothed[5].tracks[0].bbox.x1
    assert smoothed[9].tracks[0].bbox.x1 > smoothed[6].tracks[0].bbox.x1


def test_smooth_render_tracks_does_not_extrapolate_track_absent_at_final_keyframe(
    tmp_path,
) -> None:
    store = _store(tmp_path, source_fps=30, analysis_fps=6, frame_count=10)
    profile = build_full_frame_profile(width=200, height=100)
    config = AppConfig.model_validate(
        {"render": {"smoothing": {"min_observations": 2}}}
    )
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(3, 0.1, [_track(1, 3, 0.1, BBox(x1=30, y1=10, x2=50, y2=30))]),
        _record(
            5,
            5 / 30,
            [_track(2, 5, 5 / 30, BBox(x1=50, y1=10, x2=70, y2=30))],
        ),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(config, profile, store)

    smoothed = _read_records(store.render_frames_path)
    assert smoothed[-1].frame_index == 9
    assert [track.track_id for track in smoothed[5].tracks] == [2]
    assert smoothed[9].tracks == []


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


def test_smooth_render_tracks_keeps_unsmoothed_short_track_without_full_frame_scan_assumption(
    tmp_path,
) -> None:
    store = _store(tmp_path, frame_count=4)
    profile = build_full_frame_profile(width=200, height=100)
    config = AppConfig.model_validate(
        {
            "render": {
                "smoothing": {
                    "interpolation_method": "linear",
                    "min_observations": 3,
                }
            }
        }
    )
    short_track_bbox = BBox(x1=100, y1=10, x2=120, y2=30)
    records = [
        _record(
            0,
            0.0,
            [
                _track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30)),
                _track(2, 0, 0.0, short_track_bbox),
            ],
        ),
        _record(1, 1 / 30),
        _record(
            2,
            2 / 30,
            [_track(1, 2, 2 / 30, BBox(x1=20, y1=10, x2=40, y2=30))],
        ),
        _record(
            3,
            3 / 30,
            [_track(1, 3, 3 / 30, BBox(x1=30, y1=10, x2=50, y2=30))],
        ),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(config, profile, store)

    smoothed = _read_records(store.render_frames_path)
    assert [track.track_id for track in smoothed[0].tracks] == [1, 2]
    assert smoothed[0].tracks[1].bbox == short_track_bbox
    assert [track.track_id for track in smoothed[1].tracks] == [1]
    assert [track.track_id for track in smoothed[2].tracks] == [1]
    assert [track.track_id for track in smoothed[3].tracks] == [1]


def test_smooth_render_tracks_rejects_one_frame_excursion(tmp_path) -> None:
    store = _store(tmp_path)
    profile = build_full_frame_profile(width=500, height=100)
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(1, 0.1, [_track(1, 1, 0.1, BBox(x1=10, y1=10, x2=30, y2=30))]),
        _record(2, 0.2, [_track(1, 2, 0.2, BBox(x1=300, y1=10, x2=320, y2=30))]),
        _record(3, 0.3, [_track(1, 3, 0.3, BBox(x1=30, y1=10, x2=50, y2=30))]),
        _record(4, 0.4, [_track(1, 4, 0.4, BBox(x1=40, y1=10, x2=60, y2=30))]),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(AppConfig(), profile, store)

    smoothed = _read_records(store.render_frames_path)
    assert smoothed[2].tracks[0].bbox.x1 == pytest.approx(20)


def test_smooth_render_tracks_rejects_two_frame_excursion(tmp_path) -> None:
    store = _store(tmp_path)
    profile = build_full_frame_profile(width=500, height=100)
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(1, 0.1, [_track(1, 1, 0.1, BBox(x1=10, y1=10, x2=30, y2=30))]),
        _record(2, 0.2, [_track(1, 2, 0.2, BBox(x1=300, y1=10, x2=320, y2=30))]),
        _record(3, 0.3, [_track(1, 3, 0.3, BBox(x1=320, y1=10, x2=340, y2=30))]),
        _record(4, 0.4, [_track(1, 4, 0.4, BBox(x1=40, y1=10, x2=60, y2=30))]),
        _record(5, 0.5, [_track(1, 5, 0.5, BBox(x1=50, y1=10, x2=70, y2=30))]),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(AppConfig(), profile, store)

    smoothed = _read_records(store.render_frames_path)
    assert smoothed[2].tracks[0].bbox.x1 == pytest.approx(20)
    assert smoothed[3].tracks[0].bbox.x1 == pytest.approx(30)


def test_smooth_render_tracks_preserves_sustained_excursion(tmp_path) -> None:
    store = _store(tmp_path)
    profile = build_full_frame_profile(width=500, height=100)
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(1, 0.1, [_track(1, 1, 0.1, BBox(x1=10, y1=10, x2=30, y2=30))]),
        _record(2, 0.2, [_track(1, 2, 0.2, BBox(x1=300, y1=10, x2=320, y2=30))]),
        _record(3, 0.3, [_track(1, 3, 0.3, BBox(x1=320, y1=10, x2=340, y2=30))]),
        _record(4, 0.4, [_track(1, 4, 0.4, BBox(x1=340, y1=10, x2=360, y2=30))]),
        _record(5, 0.5, [_track(1, 5, 0.5, BBox(x1=50, y1=10, x2=70, y2=30))]),
        _record(6, 0.6, [_track(1, 6, 0.6, BBox(x1=60, y1=10, x2=80, y2=30))]),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(AppConfig(), profile, store)

    smoothed = _read_records(store.render_frames_path)
    assert smoothed[2].tracks[0].bbox.x1 == pytest.approx(300)
    assert smoothed[3].tracks[0].bbox.x1 == pytest.approx(320)
    assert smoothed[4].tracks[0].bbox.x1 == pytest.approx(340)


def test_smooth_render_tracks_can_disable_short_excursion_rejection(
    tmp_path,
) -> None:
    store = _store(tmp_path)
    profile = build_full_frame_profile(width=500, height=100)
    config = AppConfig.model_validate(
        {"render": {"smoothing": {"reject_short_excursions": False}}}
    )
    records = [
        _record(0, 0.0, [_track(1, 0, 0.0, BBox(x1=0, y1=10, x2=20, y2=30))]),
        _record(1, 0.1, [_track(1, 1, 0.1, BBox(x1=10, y1=10, x2=30, y2=30))]),
        _record(2, 0.2, [_track(1, 2, 0.2, BBox(x1=300, y1=10, x2=320, y2=30))]),
        _record(3, 0.3, [_track(1, 3, 0.3, BBox(x1=30, y1=10, x2=50, y2=30))]),
        _record(4, 0.4, [_track(1, 4, 0.4, BBox(x1=40, y1=10, x2=60, y2=30))]),
    ]
    _write_jsonl(store.frames_path, records)

    smooth_render_tracks(config, profile, store)

    smoothed = _read_records(store.render_frames_path)
    assert smoothed[2].tracks[0].bbox.x1 == pytest.approx(300)


def test_smooth_render_tracks_clamps_extreme_center_offset(tmp_path) -> None:
    store = _store(tmp_path)
    profile = build_full_frame_profile(width=200, height=100)
    config = AppConfig.model_validate(
        {
            "render": {
                "smoothing": {
                    "max_center_offset_ratio": 0.1,
                    "reject_short_excursions": False,
                }
            }
        }
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
