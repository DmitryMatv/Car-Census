from models import TrackSummary


def test_track_summary_migrates_legacy_height_metric_keys() -> None:
    # Legacy manifests used *_box_height_px to mean width; the modern
    # per-dimension height stats must stay unset for those records.
    payload = {
        "track_id": 1,
        "first_frame_index": 0,
        "last_frame_index": 10,
        "frames_seen": 5,
        "min_box_height_px": 40.0,
        "max_box_height_px": 90.0,
    }

    summary = TrackSummary.model_validate(payload)

    assert summary.min_box_width_px == 40.0
    assert summary.max_box_width_px == 90.0
    assert summary.min_box_height_px is None
    assert summary.max_box_height_px is None


def test_track_summary_keeps_modern_height_stats() -> None:
    payload = {
        "track_id": 1,
        "first_frame_index": 0,
        "last_frame_index": 10,
        "frames_seen": 5,
        "min_box_width_px": 100.0,
        "max_box_width_px": 200.0,
        "min_box_height_px": 50.0,
        "max_box_height_px": 80.0,
    }

    summary = TrackSummary.model_validate(payload)

    assert summary.min_box_width_px == 100.0
    assert summary.max_box_width_px == 200.0
    assert summary.min_box_height_px == 50.0
    assert summary.max_box_height_px == 80.0
