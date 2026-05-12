from pathlib import Path

import cv2
import numpy as np
import orjson
import pytest

from config import AppConfig, build_full_frame_profile
from pipeline import render as render_module
from pipeline.render import (
    format_label_text,
    render_video,
    visible_track_label_text_by_track,
)
import render.annotators as annotator_module
from render.annotators import (
    VideoAnnotator,
    label_box_bounds,
    resolve_label_box_bounds,
)
from models import (
    BBox,
    FrameRecord,
    MMRResult,
    RunManifest,
    TrackedObject,
)
from utils.video import VideoMetadata


class DummyRunStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.frames_path = root / "analysis" / "frames.jsonl"
        self.render_frames_path = root / "analysis" / "render_frames.jsonl"
        self.tracks_path = root / "analysis" / "tracks.jsonl"
        self.labels_path = root / "mmr" / "labels.json"
        self.output_video_path = root / "annotated.mp4"
        self.manifest = RunManifest(
            run_id="test",
            video_path=root / "input.mp4",
            camera_id="__full_frame__",
            root_dir=root,
            source_fps=30.0,
            analysis_fps=10.0,
            width=16,
            height=16,
        )

    def read_manifest(self) -> RunManifest:
        return self.manifest


class DummyWriter:
    def __init__(self) -> None:
        self.frames = []
        self.released = False

    def write(self, frame) -> None:
        self.frames.append(frame.copy())

    def release(self) -> None:
        self.released = True


class DummyAnnotator:
    seen_track_ids = []
    seen_labels_by_track = []

    def __init__(self, config: AppConfig) -> None:
        _ = config

    def annotate(self, frame, profile, tracks, labels_by_track):
        _ = profile
        self.seen_track_ids.append([track.track_id for track in tracks])
        self.seen_labels_by_track.append(dict(labels_by_track))
        return frame


def _track(
    track_id: int,
    frame_index: int,
    timestamp: float,
    vehicle_index: int | None = None,
) -> TrackedObject:
    bbox = BBox(x1=1, y1=2, x2=11, y2=12)
    return TrackedObject(
        track_id=track_id,
        vehicle_index=vehicle_index,
        frame_index=frame_index,
        timestamp_seconds=timestamp,
        bbox=bbox,
        confidence=0.9,
        class_id=2,
        class_name="car",
        centroid=bbox.center,
        bottom_center=((bbox.x1 + bbox.x2) / 2.0, bbox.y2),
        inside_roi=True,
    )


def _render_track(
    bbox: BBox,
    track_id: int = 1,
    frame_index: int = 0,
    timestamp: float = 0.0,
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
        bottom_center=((bbox.x1 + bbox.x2) / 2.0, bbox.y2),
        inside_roi=True,
    )


def _write_frame_records(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        FrameRecord(
            frame_index=0,
            timestamp_seconds=0.0,
            tracks=[_track(1, 0, 0.0)],
        ),
        FrameRecord(
            frame_index=3,
            timestamp_seconds=0.1,
            tracks=[_track(2, 3, 0.1)],
        ),
    ]
    path.write_bytes(
        b"".join(
            orjson.dumps(record.model_dump(mode="json")) + b"\n" for record in records
        )
    )


def test_render_writes_every_source_frame_when_analysis_is_downsampled(
    tmp_path, monkeypatch
) -> None:
    store = DummyRunStore(tmp_path)
    _write_frame_records(store.frames_path)
    writer = DummyWriter()
    DummyAnnotator.seen_track_ids = []
    DummyAnnotator.seen_labels_by_track = []

    monkeypatch.setattr(
        render_module,
        "read_video_metadata",
        lambda video_path: VideoMetadata(width=16, height=16, fps=30.0, frame_count=6),
    )
    monkeypatch.setattr(
        render_module,
        "iter_video_frames",
        lambda video_path, fps: (
            (index, index / 30.0, np.zeros((16, 16, 3), dtype=np.uint8))
            for index in range(6)
        ),
    )
    monkeypatch.setattr(
        render_module,
        "build_video_writer",
        lambda **kwargs: writer,
    )
    monkeypatch.setattr(render_module, "VideoAnnotator", DummyAnnotator)

    render_video(
        config=AppConfig.model_validate({"render": {"smoothing": {"enabled": False}}}),
        profile=build_full_frame_profile(width=16, height=16),
        video_path=store.manifest.video_path,
        run_store=store,
    )

    assert len(writer.frames) == 6
    assert writer.released is True
    assert DummyAnnotator.seen_track_ids == [[1], [1], [1], [2], [2], [2]]


def test_render_uses_configured_video_fps_for_iteration_and_writer(
    tmp_path, monkeypatch
) -> None:
    store = DummyRunStore(tmp_path)
    _write_frame_records(store.frames_path)
    writer = DummyWriter()
    iterated_fps = []
    writer_kwargs = {}

    monkeypatch.setattr(
        render_module,
        "read_video_metadata",
        lambda video_path: VideoMetadata(width=16, height=16, fps=30.0, frame_count=6),
    )

    def fake_iter_video_frames(video_path, fps):
        iterated_fps.append(fps)
        for index in range(3):
            yield index, index / 30.0, np.zeros((16, 16, 3), dtype=np.uint8)

    def fake_build_video_writer(**kwargs):
        writer_kwargs.update(kwargs)
        return writer

    monkeypatch.setattr(render_module, "iter_video_frames", fake_iter_video_frames)
    monkeypatch.setattr(render_module, "build_video_writer", fake_build_video_writer)
    monkeypatch.setattr(render_module, "VideoAnnotator", DummyAnnotator)

    render_video(
        config=AppConfig.model_validate({"render": {"smoothing": {"enabled": False}}}),
        profile=build_full_frame_profile(width=16, height=16),
        video_path=store.manifest.video_path,
        run_store=store,
    )

    assert iterated_fps == [30.0]
    assert writer_kwargs["fps"] == 30.0
    assert len(writer.frames) == 3


def test_render_rejects_non_30_fps_input(tmp_path, monkeypatch) -> None:
    store = DummyRunStore(tmp_path)
    _write_frame_records(store.frames_path)

    monkeypatch.setattr(
        render_module,
        "read_video_metadata",
        lambda video_path: VideoMetadata(width=16, height=16, fps=25.0, frame_count=6),
    )

    with pytest.raises(RuntimeError, match="Input video FPS"):
        render_video(
            config=AppConfig.model_validate(
                {"render": {"smoothing": {"enabled": False}}}
            ),
            profile=build_full_frame_profile(width=16, height=16),
            video_path=store.manifest.video_path,
            run_store=store,
        )


def test_render_formats_api_counter_prefix() -> None:
    assert (
        format_label_text(
            MMRResult(
                make="Toyota",
                model="Corolla",
                vehicle_index=13,
                api_classification_index=3,
            ),
            "unknown",
        )
        == "13 | Toyota Corolla"
    )
    assert (
        format_label_text(MMRResult(api_classification_index=3), "unknown")
        == "3 | unknown"
    )
    assert (
        format_label_text(MMRResult(make="Toyota", model="Corolla"), "unknown")
        == "Toyota Corolla"
    )
    assert format_label_text(MMRResult(), "unknown") == "unknown"


def test_render_builds_numbered_unknown_labels_from_visible_tracks(
    tmp_path,
) -> None:
    frames_path = tmp_path / "frames.jsonl"
    frames_path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        FrameRecord(
            frame_index=0,
            timestamp_seconds=0.0,
            tracks=[_track(4, 0, 0.0), _track(2, 0, 0.0)],
        ),
        FrameRecord(
            frame_index=1,
            timestamp_seconds=0.1,
            tracks=[_track(2, 1, 0.1), _track(9, 1, 0.1)],
        ),
    ]
    frames_path.write_bytes(
        b"".join(
            orjson.dumps(record.model_dump(mode="json")) + b"\n" for record in records
        )
    )

    labels = visible_track_label_text_by_track(frames_path, "UNKNOWN")

    assert labels == {
        4: "1 | UNKNOWN",
        2: "2 | UNKNOWN",
        9: "3 | UNKNOWN",
    }


def test_render_uses_vehicle_index_for_visible_track_labels(
    tmp_path,
) -> None:
    frames_path = tmp_path / "frames.jsonl"
    frames_path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        FrameRecord(
            frame_index=0,
            timestamp_seconds=0.0,
            tracks=[
                _track(4, 0, 0.0, vehicle_index=13),
                _track(2, 0, 0.0, vehicle_index=7),
                _track(9, 0, 0.0, vehicle_index=None),
            ],
        ),
        FrameRecord(
            frame_index=1,
            timestamp_seconds=0.1,
            tracks=[_track(4, 1, 0.1, vehicle_index=13)],
        ),
    ]
    frames_path.write_bytes(
        b"".join(
            orjson.dumps(record.model_dump(mode="json")) + b"\n" for record in records
        )
    )

    labels = visible_track_label_text_by_track(frames_path, "UNKNOWN")

    assert labels == {
        4: "13 | UNKNOWN",
        2: "7 | UNKNOWN",
    }


def test_render_passes_numbered_unknown_labels_when_labels_file_is_missing(
    tmp_path, monkeypatch
) -> None:
    store = DummyRunStore(tmp_path)
    _write_frame_records(store.frames_path)
    writer = DummyWriter()
    DummyAnnotator.seen_track_ids = []
    DummyAnnotator.seen_labels_by_track = []

    monkeypatch.setattr(
        render_module,
        "read_video_metadata",
        lambda video_path: VideoMetadata(width=16, height=16, fps=30.0, frame_count=6),
    )
    monkeypatch.setattr(
        render_module,
        "iter_video_frames",
        lambda video_path, fps: (
            (index, index / 30.0, np.zeros((16, 16, 3), dtype=np.uint8))
            for index in range(6)
        ),
    )
    monkeypatch.setattr(render_module, "build_video_writer", lambda **kwargs: writer)
    monkeypatch.setattr(render_module, "VideoAnnotator", DummyAnnotator)

    render_video(
        config=AppConfig.model_validate({"render": {"smoothing": {"enabled": False}}}),
        profile=build_full_frame_profile(width=16, height=16),
        video_path=store.manifest.video_path,
        run_store=store,
    )

    assert DummyAnnotator.seen_labels_by_track
    assert DummyAnnotator.seen_labels_by_track[0] == {
        1: "1 | UNKNOWN",
        2: "2 | UNKNOWN",
    }


def test_render_hides_tracks_without_vehicle_index_for_new_runs(
    tmp_path, monkeypatch
) -> None:
    store = DummyRunStore(tmp_path)
    store.frames_path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        FrameRecord(
            frame_index=0,
            timestamp_seconds=0.0,
            tracks=[
                _track(1, 0, 0.0, vehicle_index=1),
                _track(2, 0, 0.0, vehicle_index=None),
            ],
        )
    ]
    store.frames_path.write_bytes(
        b"".join(
            orjson.dumps(record.model_dump(mode="json")) + b"\n" for record in records
        )
    )
    writer = DummyWriter()
    DummyAnnotator.seen_track_ids = []
    DummyAnnotator.seen_labels_by_track = []

    monkeypatch.setattr(
        render_module,
        "read_video_metadata",
        lambda video_path: VideoMetadata(width=16, height=16, fps=30.0, frame_count=1),
    )
    monkeypatch.setattr(
        render_module,
        "iter_video_frames",
        lambda video_path, fps: (
            (index, index / 30.0, np.zeros((16, 16, 3), dtype=np.uint8))
            for index in range(1)
        ),
    )
    monkeypatch.setattr(render_module, "build_video_writer", lambda **kwargs: writer)
    monkeypatch.setattr(render_module, "VideoAnnotator", DummyAnnotator)

    render_video(
        config=AppConfig.model_validate({"render": {"smoothing": {"enabled": False}}}),
        profile=build_full_frame_profile(width=16, height=16),
        video_path=store.manifest.video_path,
        run_store=store,
    )

    assert DummyAnnotator.seen_track_ids == [[1]]
    assert DummyAnnotator.seen_labels_by_track[0] == {1: "1 | UNKNOWN"}


def test_video_annotator_places_label_above_and_left_aligned() -> None:
    config = AppConfig.model_validate(
        {
            "render": {
                "label_font_scale": 0.5,
                "label_thickness": 1,
                "label_padding_px": 4,
                "label_gap_px": 6,
            }
        }
    )
    bbox = BBox(x1=60, y1=40, x2=100, y2=80)
    track = TrackedObject(
        track_id=1,
        frame_index=0,
        timestamp_seconds=0.0,
        bbox=bbox,
        confidence=0.9,
        class_id=2,
        class_name="car",
        centroid=bbox.center,
        bottom_center=((bbox.x1 + bbox.x2) / 2.0, bbox.y2),
        inside_roi=True,
    )
    bounds = label_box_bounds(
        frame_shape=(120, 180, 3),
        track=track,
        label="#1 Toyota",
        config=config,
    )
    left, top, _right, bottom, _baseline = bounds

    assert left + config.render.label_padding_px == int(round(track.bbox.x1 + 1))
    assert bottom == int(round(track.bbox.y1 - config.render.label_gap_px))


def test_video_annotator_vertically_centers_label_text() -> None:
    config = AppConfig.model_validate(
        {
            "render": {
                "label_font_scale": 0.6,
                "label_thickness": 1,
                "label_padding_px": 5,
                "label_gap_px": 6,
            }
        }
    )
    bbox = BBox(x1=60, y1=20, x2=100, y2=60)
    track = _render_track(bbox=bbox)
    label = "1 | Toyota"

    layout = resolve_label_box_bounds(
        frame_shape=(120, 180, 3),
        tracks=[track],
        labels=[label],
        config=config,
    )[0]
    (_text_width, text_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        config.render.label_font_scale,
        config.render.label_thickness,
    )
    label_height = layout.bottom - layout.top
    old_origin_y = layout.bottom - config.render.label_padding_px - baseline

    assert layout.text_origin[1] == layout.top + int(
        round((label_height + text_height) / 2.0)
    )
    assert layout.text_origin[1] != old_origin_y


def test_video_annotator_uses_fixed_color_without_class_ids() -> None:
    config = AppConfig()
    bbox = BBox(x1=40, y1=20, x2=90, y2=70)
    track = _render_track(bbox=bbox)
    track.class_id = None
    track.class_name = None
    frame = np.zeros((140, 180, 3), dtype=np.uint8)

    annotated = VideoAnnotator(config).annotate(
        frame=frame,
        profile=build_full_frame_profile(width=180, height=140),
        tracks=[track],
        labels_by_track={1: "#1 Toyota Corolla"},
    )

    corner_pixels = annotated[19:23, 40:73]

    assert annotated.shape == frame.shape
    assert annotated.sum() > 0
    assert np.any(np.all(corner_pixels >= 240, axis=2))


def test_video_annotator_draws_glow_around_corner_box() -> None:
    config = AppConfig.model_validate(
        {
            "render": {
                "corner_thickness": 2,
                "corner_length": 24,
                "glow_radius_px": 5,
                "glow_alpha": 0.9,
                "label_glow_alpha": 0.0,
            }
        }
    )
    bbox = BBox(x1=50, y1=40, x2=100, y2=90)
    track = _render_track(bbox=bbox)
    frame = np.zeros((130, 160, 3), dtype=np.uint8)

    annotated = VideoAnnotator(config).annotate(
        frame=frame,
        profile=build_full_frame_profile(width=160, height=130),
        tracks=[track],
        labels_by_track={1: "1 | Toyota"},
    )

    glow_region = annotated[34:38, 55:65]
    crisp_region = annotated[39:42, 55:65]

    assert glow_region.max() > 0
    assert crisp_region.max() > glow_region.max()


def test_video_annotator_can_draw_glass_label_without_solid_purple() -> None:
    config = AppConfig.model_validate(
        {
            "render": {
                "label_font_scale": 0.6,
                "label_thickness": 1,
                "label_padding_px": 5,
                "label_gap_px": 6,
                "label_bg_alpha": 0.42,
                "label_glow_alpha": 0.0,
            }
        }
    )
    bbox = BBox(x1=40, y1=50, x2=90, y2=85)
    track = _render_track(bbox=bbox)
    frame = np.full((130, 180, 3), 100, dtype=np.uint8)

    annotated = VideoAnnotator(config).annotate(
        frame=frame,
        profile=build_full_frame_profile(width=180, height=130),
        tracks=[track],
        labels_by_track={1: "1 | Toyota"},
    )
    left, top, right, bottom, _baseline = label_box_bounds(
        frame.shape, track, "1 | Toyota", config
    )
    label_area = annotated[top:bottom, left:right]
    backing_pixel = annotated[bottom - 3, right - 3]
    old_purple_bgr = np.array([247, 85, 168], dtype=np.uint8)

    assert not np.any(np.all(label_area == old_purple_bgr, axis=2))
    assert np.all(backing_pixel < 100)
    assert not np.all(backing_pixel == 255)
    assert label_area.max() >= 240


def test_video_annotator_defaults_to_glowing_text_without_background_or_shadow() -> (
    None
):
    glow_config = AppConfig.model_validate(
        {
            "render": {
                "label_font_scale": 0.8,
                "label_thickness": 2,
                "label_padding_px": 5,
                "label_gap_px": 6,
                "glow_enabled": False,
                "label_bg_alpha": 0.0,
                "label_shadow_enabled": False,
                "label_glow_alpha": 0.30,
            }
        }
    )
    no_glow_config = AppConfig.model_validate(
        {
            "render": {
                "label_font_scale": 0.8,
                "label_thickness": 2,
                "label_padding_px": 5,
                "label_gap_px": 6,
                "glow_enabled": False,
                "label_bg_alpha": 0.0,
                "label_shadow_enabled": False,
                "label_glow_alpha": 0.0,
            }
        }
    )
    bbox = BBox(x1=40, y1=20, x2=90, y2=55)
    track = _render_track(bbox=bbox)
    frame = np.zeros((130, 180, 3), dtype=np.uint8)

    with_glow = VideoAnnotator(glow_config).annotate(
        frame=frame,
        profile=build_full_frame_profile(width=180, height=130),
        tracks=[track],
        labels_by_track={1: "1 | Toyota"},
    )
    without_glow = VideoAnnotator(no_glow_config).annotate(
        frame=frame,
        profile=build_full_frame_profile(width=180, height=130),
        tracks=[track],
        labels_by_track={1: "1 | Toyota"},
    )
    left, top, right, bottom, _baseline = label_box_bounds(
        frame.shape, track, "1 | Toyota", glow_config
    )
    background_corner = with_glow[bottom - 2, right - 2]

    assert with_glow.sum() > without_glow.sum()
    assert with_glow.max() == without_glow.max()
    assert np.all(background_corner == 0)


def test_video_annotator_does_not_draw_trace_history_between_frames() -> None:
    config = AppConfig.model_validate(
        {
            "render": {
                "glow_enabled": False,
                "glow_alpha": 0.0,
                "label_glow_alpha": 0.0,
            }
        }
    )
    first_track = _render_track(
        bbox=BBox(x1=20, y1=20, x2=50, y2=50), frame_index=0, timestamp=0.0
    )
    second_track = _render_track(
        bbox=BBox(x1=60, y1=20, x2=90, y2=50), frame_index=1, timestamp=0.1
    )
    annotator = VideoAnnotator(config)
    frame = np.zeros((120, 140, 3), dtype=np.uint8)

    annotator.annotate(
        frame=frame,
        profile=build_full_frame_profile(width=140, height=120),
        tracks=[first_track],
        labels_by_track={1: "1 | Toyota"},
    )
    annotated = annotator.annotate(
        frame=frame,
        profile=build_full_frame_profile(width=140, height=120),
        tracks=[second_track],
        labels_by_track={1: "1 | Toyota"},
    )

    assert annotated[35, 55].max() == 0


def test_video_annotator_does_not_draw_bright_label_border() -> None:
    config = AppConfig.model_validate(
        {
            "render": {
                "label_font_scale": 0.6,
                "label_thickness": 1,
                "label_padding_px": 5,
                "label_gap_px": 6,
                "label_bg_alpha": 0.42,
                "label_shadow_enabled": False,
                "glow_enabled": False,
            }
        }
    )
    bbox = BBox(x1=40, y1=50, x2=90, y2=85)
    track = _render_track(bbox=bbox)
    frame = np.full((130, 180, 3), 100, dtype=np.uint8)

    annotated = VideoAnnotator(config).annotate(
        frame=frame,
        profile=build_full_frame_profile(width=180, height=130),
        tracks=[track],
        labels_by_track={1: "1 | Toyota"},
    )
    left, top, right, bottom, _baseline = label_box_bounds(
        frame.shape, track, "1 | Toyota", config
    )
    border_pixels = np.concatenate(
        [
            annotated[top, left:right],
            annotated[bottom, left:right],
            annotated[top:bottom, left],
            annotated[top:bottom, right],
        ]
    )

    assert border_pixels.max() < 240


def test_video_annotator_draws_subtle_text_shadow_without_white_glow() -> None:
    base_config = {
        "render": {
            "label_font_scale": 0.8,
            "label_thickness": 2,
            "label_padding_px": 5,
            "label_gap_px": 6,
            "label_bg_alpha": 0.0,
            "label_shadow_enabled": True,
            "label_shadow_alpha": 0.55,
            "label_shadow_offset_px": 2,
            "label_shadow_thickness_extra": 2,
            "glow_enabled": False,
            "label_glow_alpha": 0.0,
        }
    }
    shadow_config = AppConfig.model_validate(base_config)
    no_shadow_config = AppConfig.model_validate(
        {
            "render": {
                **base_config["render"],
                "label_shadow_enabled": False,
            }
        }
    )
    bbox = BBox(x1=40, y1=20, x2=90, y2=55)
    track = _render_track(bbox=bbox)
    frame = np.full((130, 180, 3), 100, dtype=np.uint8)

    with_shadow = VideoAnnotator(shadow_config).annotate(
        frame=frame,
        profile=build_full_frame_profile(width=180, height=130),
        tracks=[track],
        labels_by_track={1: "1 | Toyota"},
    )
    without_shadow = VideoAnnotator(no_shadow_config).annotate(
        frame=frame,
        profile=build_full_frame_profile(width=180, height=130),
        tracks=[track],
        labels_by_track={1: "1 | Toyota"},
    )

    assert with_shadow.sum() < without_shadow.sum()
    assert with_shadow.max() == without_shadow.max()


def test_video_annotator_label_drawing_does_not_expose_full_frame_overlay_helpers() -> (
    None
):
    assert not hasattr(annotator_module, "_overlay_layer")
    assert not hasattr(annotator_module, "_overlay_text_layer")


def test_video_annotator_draws_edge_label_shadow_with_roi_clipping() -> None:
    config = AppConfig.model_validate(
        {
            "render": {
                "label_font_scale": 0.8,
                "label_thickness": 2,
                "label_padding_px": 5,
                "label_gap_px": 6,
                "label_shadow_enabled": True,
                "label_shadow_offset_px": 3,
                "label_shadow_thickness_extra": 3,
                "label_smart_position": True,
                "glow_enabled": False,
            }
        }
    )
    bbox = BBox(x1=95, y1=55, x2=119, y2=78)
    track = _render_track(bbox=bbox)
    frame = np.zeros((80, 120, 3), dtype=np.uint8)

    annotated = VideoAnnotator(config).annotate(
        frame=frame,
        profile=build_full_frame_profile(width=120, height=80),
        tracks=[track],
        labels_by_track={1: "1 | Long Unknown Label"},
    )

    assert annotated.shape == frame.shape
    assert annotated.sum() > 0


def test_resolve_label_box_bounds_separates_overlapping_labels() -> None:
    config = AppConfig.model_validate(
        {
            "render": {
                "label_font_scale": 0.5,
                "label_thickness": 1,
                "label_padding_px": 4,
                "label_smart_position": True,
                "label_max_offset_px": 48,
            }
        }
    )
    tracks = [
        _render_track(bbox=BBox(x1=40, y1=20, x2=70, y2=55), track_id=1),
        _render_track(bbox=BBox(x1=44, y1=20, x2=74, y2=55), track_id=2),
    ]

    layouts = resolve_label_box_bounds(
        frame_shape=(130, 180, 3),
        tracks=tracks,
        labels=["A", "B"],
        config=config,
    )

    assert layouts[0].right <= layouts[1].left or layouts[1].right <= layouts[0].left


def test_resolve_label_box_bounds_clamps_labels_inside_frame() -> None:
    config = AppConfig.model_validate(
        {
            "render": {
                "label_font_scale": 0.8,
                "label_thickness": 2,
                "label_padding_px": 5,
                "label_smart_position": True,
            }
        }
    )
    track = _render_track(bbox=BBox(x1=145, y1=95, x2=178, y2=120))

    layout = resolve_label_box_bounds(
        frame_shape=(130, 180, 3),
        tracks=[track],
        labels=["1 | Unknown"],
        config=config,
    )[0]

    assert layout.left >= 0
    assert layout.top >= 0
    assert layout.right <= 180
    assert layout.bottom <= 130


def test_resolve_label_box_bounds_respects_max_offset() -> None:
    anchored_config = AppConfig.model_validate(
        {"render": {"label_font_scale": 0.5, "label_smart_position": False}}
    )
    smart_config = AppConfig.model_validate(
        {
            "render": {
                "label_font_scale": 0.5,
                "label_smart_position": True,
                "label_max_offset_px": 3,
            }
        }
    )
    tracks = [
        _render_track(bbox=BBox(x1=40, y1=20, x2=70, y2=55), track_id=1),
        _render_track(bbox=BBox(x1=44, y1=20, x2=74, y2=55), track_id=2),
    ]

    anchored = resolve_label_box_bounds(
        frame_shape=(130, 180, 3),
        tracks=tracks,
        labels=["A", "B"],
        config=anchored_config,
    )
    smart = resolve_label_box_bounds(
        frame_shape=(130, 180, 3),
        tracks=tracks,
        labels=["A", "B"],
        config=smart_config,
    )

    for anchored_layout, smart_layout in zip(anchored, smart, strict=True):
        assert abs(smart_layout.left - anchored_layout.left) <= 3
        assert abs(smart_layout.top - anchored_layout.top) <= 3
