from pathlib import Path

import numpy as np
import orjson

from car_census.config import AppConfig, build_full_frame_profile
from car_census.pipeline import render as render_module
from car_census.pipeline.render import (
    format_label_text,
    render_video,
    visible_track_label_text_by_track,
)
from car_census.render.annotators import VideoAnnotator, label_box_bounds
from car_census.types import (
    BBox,
    FrameRecord,
    MMRResult,
    RunManifest,
    TrackedObject,
)
from car_census.utils.video import VideoMetadata


class DummyRunStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.frames_path = root / "analysis" / "frames.jsonl"
        self.render_frames_path = root / "analysis" / "render_frames.jsonl"
        self.tracks_path = root / "analysis" / "tracks.jsonl"
        self.labels_path = root / "mmr" / "labels.json"
        self.output_video_path = root / "render" / "annotated.mp4"
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
        "iter_sampled_frames",
        lambda video_path, target_fps: (
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
        "iter_sampled_frames",
        lambda video_path, target_fps: (
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
        "iter_sampled_frames",
        lambda video_path, target_fps: (
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


def test_video_annotator_places_label_below_and_centered() -> None:
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
    bbox = BBox(x1=60, y1=20, x2=100, y2=60)
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
    left, top, right, _bottom, _baseline = bounds
    bbox_center_x = (track.bbox.x1 + track.bbox.x2) / 2.0
    label_center_x = (left + right) / 2.0

    assert top == int(round(track.bbox.y2 + config.render.label_gap_px))
    assert label_center_x == bbox_center_x


def test_video_annotator_uses_fixed_color_without_class_ids() -> None:
    config = AppConfig()
    bbox = BBox(x1=40, y1=20, x2=90, y2=70)
    track = TrackedObject(
        track_id=1,
        frame_index=0,
        timestamp_seconds=0.0,
        bbox=bbox,
        confidence=0.9,
        class_id=None,
        class_name=None,
        centroid=bbox.center,
        bottom_center=((bbox.x1 + bbox.x2) / 2.0, bbox.y2),
        inside_roi=True,
    )
    frame = np.zeros((140, 180, 3), dtype=np.uint8)

    annotated = VideoAnnotator(config).annotate(
        frame=frame,
        profile=build_full_frame_profile(width=180, height=140),
        tracks=[track],
        labels_by_track={1: "#1 Toyota Corolla"},
    )

    assert annotated.shape == frame.shape
    assert annotated.sum() > 0
