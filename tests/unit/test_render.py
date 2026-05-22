from pathlib import Path

import numpy as np
import orjson
import pytest

from config import AppConfig, build_full_frame_profile
from models import BBox, FrameRecord, MMRResult, RunManifest, TrackedObject
from pipeline import render as render_module
from pipeline.render import (
    format_label_text,
    render_video,
    visible_track_label_text_by_track,
)
from render.annotators import VideoAnnotator
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
            width=32,
            height=32,
        )

    def read_labels(self) -> dict[int, MMRResult]:
        if not self.labels_path.exists():
            return {}
        raw = orjson.loads(self.labels_path.read_bytes())
        return {
            int(track_id): MMRResult.model_validate(payload)
            for track_id, payload in raw.items()
        }

    def iter_frame_records(self, *, smoothed: bool = False):
        path = self.render_frames_path if smoothed else self.frames_path
        if not path.exists():
            return
        for line in path.read_bytes().splitlines():
            if line.strip():
                yield FrameRecord.model_validate(orjson.loads(line))

    def read_frame_records(self, *, smoothed: bool = False) -> list[FrameRecord]:
        return list(self.iter_frame_records(smoothed=smoothed))


class DummyWriter:
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []
        self.released = False

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def release(self) -> None:
        self.released = True


class DummyAnnotator:
    seen_track_ids: list[list[int]] = []
    seen_labels_by_track: list[dict[int, str]] = []

    def __init__(self, config: AppConfig) -> None:
        _ = config

    def annotate(
        self,
        frame: np.ndarray,
        tracks: list[TrackedObject],
        labels_by_track: dict[int, str],
    ) -> np.ndarray:
        self.seen_track_ids.append([track.track_id for track in tracks])
        self.seen_labels_by_track.append(dict(labels_by_track))
        return frame


def _track(
    track_id: int,
    frame_index: int = 0,
    vehicle_index: int | None = None,
    bbox: BBox | None = None,
) -> TrackedObject:
    bbox = bbox or BBox(x1=4, y1=5, x2=22, y2=24)
    return TrackedObject(
        track_id=track_id,
        vehicle_index=vehicle_index,
        frame_index=frame_index,
        timestamp_seconds=frame_index / 30.0,
        bbox=bbox,
        confidence=0.9,
        class_id=2,
        class_name="car",
        centroid=bbox.center,
        bottom_center=bbox.bottom_center,
        inside_roi=True,
    )


def _write_records(path: Path, records: list[FrameRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"".join(
            orjson.dumps(record.model_dump(mode="json")) + b"\n" for record in records
        )
    )


def _config(
    min_visible_track_observations: int = 1,
    require_crop_eligible_track: bool = False,
    show_unclassified_tracks: bool = False,
) -> AppConfig:
    return AppConfig.model_validate(
        {
            "render": {
                "min_visible_track_observations": min_visible_track_observations,
                "require_crop_eligible_track": require_crop_eligible_track,
                "show_unclassified_tracks": show_unclassified_tracks,
                "smoothing": {"enabled": False},
            }
        }
    )


def _patch_render_io(monkeypatch: pytest.MonkeyPatch, writer: DummyWriter) -> None:
    monkeypatch.setattr(
        render_module,
        "read_video_metadata",
        lambda video_path: VideoMetadata(width=32, height=32, fps=30.0, frame_count=3),
    )
    monkeypatch.setattr(
        render_module,
        "iter_video_frames",
        lambda video_path, fps: (
            (index, index / 30.0, np.zeros((32, 32, 3), dtype=np.uint8))
            for index in range(3)
        ),
    )
    monkeypatch.setattr(render_module, "build_frame_writer", lambda **kwargs: writer)
    monkeypatch.setattr(render_module, "VideoAnnotator", DummyAnnotator)


def test_video_annotator_returns_same_shape_frame() -> None:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    annotated = VideoAnnotator(AppConfig()).annotate(
        frame,
        tracks=[_track(1)],
        labels_by_track={1: "1 | Toyota Corolla"},
    )

    assert annotated.shape == frame.shape


def test_video_annotator_no_tracks_returns_unchanged_copy() -> None:
    frame = np.full((32, 32, 3), 7, dtype=np.uint8)
    annotated = VideoAnnotator(AppConfig()).annotate(frame, [], {})

    assert np.array_equal(annotated, frame)
    assert annotated is not frame


def test_video_annotator_visible_track_produces_nonzero_pixels() -> None:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    annotated = VideoAnnotator(AppConfig()).annotate(
        frame,
        tracks=[_track(1)],
        labels_by_track={1: "1 | Toyota Corolla"},
    )

    assert int(annotated.max()) > 0


def test_video_annotator_handles_missing_labels_with_unknown_fallback() -> None:
    annotated = VideoAnnotator(AppConfig()).annotate(
        np.zeros((64, 64, 3), dtype=np.uint8),
        tracks=[_track(1)],
        labels_by_track={},
    )
    assert isinstance(annotated, np.ndarray)


def test_video_annotator_accepts_fixed_configured_colors() -> None:
    config = AppConfig.model_validate(
        {
            "render": {
                "box_color": "#00FF00",
                "label_bg_color": "#101820",
                "label_text_color": "#FFFFFF",
            }
        }
    )
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    annotated = VideoAnnotator(config).annotate(
        frame,
        tracks=[_track(1, bbox=BBox(x1=8, y1=12, x2=42, y2=48))],
        labels_by_track={1: "1 | Toyota"},
    )

    assert int(annotated.max()) > 0


def test_video_annotator_does_not_retain_trace_history_between_calls() -> None:
    annotator = VideoAnnotator(AppConfig())
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    first = annotator.annotate(frame, [_track(1)], {1: "1 | Toyota"})
    second = annotator.annotate(frame, [], {})

    assert int(first.max()) > 0
    assert np.array_equal(second, frame)


def test_format_label_text_uses_single_line_vehicle_index_only() -> None:
    assert (
        format_label_text(
            MMRResult(
                make="Toyota",
                model="Corolla",
                generation="E210",
                variation="Hybrid",
                vehicle_index=4,
                api_classification_index=99,
            ),
            "unknown",
        )
        == "4 | Toyota Corolla | E210 | Hybrid"
    )
    assert format_label_text(MMRResult(api_classification_index=3), "unknown") == (
        "unknown"
    )
    assert format_label_text(MMRResult(make="Toyota", model="Corolla"), "unknown") == (
        "Toyota Corolla"
    )


def test_visible_track_label_text_only_uses_vehicle_indexed_tracks(tmp_path) -> None:
    frames_path = tmp_path / "frames.jsonl"
    records = [
        FrameRecord(
            frame_index=0,
            timestamp_seconds=0.0,
            tracks=[_track(1), _track(2, vehicle_index=7)],
        )
    ]
    _write_records(
        frames_path,
        records,
    )

    assert visible_track_label_text_by_track(records, "unknown") == {2: "7 | unknown"}


def test_render_filters_by_visibility_count(tmp_path, monkeypatch) -> None:
    store = DummyRunStore(tmp_path)
    _write_records(
        store.frames_path,
        [
            FrameRecord(
                frame_index=0,
                timestamp_seconds=0.0,
                tracks=[_track(1, 0), _track(2, 0)],
            ),
            FrameRecord(
                frame_index=1,
                timestamp_seconds=1 / 30.0,
                tracks=[_track(1, 1)],
            ),
        ],
    )
    store.labels_path.parent.mkdir(parents=True, exist_ok=True)
    store.labels_path.write_bytes(
        orjson.dumps(
            {
                "1": MMRResult(vehicle_index=1).model_dump(mode="json"),
                "2": MMRResult(vehicle_index=2).model_dump(mode="json"),
            }
        )
    )
    writer = DummyWriter()
    DummyAnnotator.seen_track_ids = []
    DummyAnnotator.seen_labels_by_track = []
    _patch_render_io(monkeypatch, writer)

    render_video(
        config=_config(min_visible_track_observations=2),
        profile=build_full_frame_profile(width=32, height=32),
        video_path=store.manifest.video_path,
        run_store=store,
    )

    assert [1] in DummyAnnotator.seen_track_ids
    assert [2] not in DummyAnnotator.seen_track_ids
    assert writer.released is True


def test_render_respects_require_crop_eligible_track(tmp_path, monkeypatch) -> None:
    store = DummyRunStore(tmp_path)
    _write_records(
        store.frames_path,
        [
            FrameRecord(
                frame_index=0,
                timestamp_seconds=0.0,
                tracks=[
                    _track(1, 0, vehicle_index=1),
                    _track(2, 0, vehicle_index=None),
                ],
            )
        ],
    )
    store.labels_path.parent.mkdir(parents=True, exist_ok=True)
    store.labels_path.write_bytes(
        orjson.dumps(
            {
                "1": MMRResult(vehicle_index=1).model_dump(mode="json"),
                "2": MMRResult(vehicle_index=2).model_dump(mode="json"),
            }
        )
    )
    writer = DummyWriter()
    DummyAnnotator.seen_track_ids = []
    _patch_render_io(monkeypatch, writer)

    render_video(
        config=_config(require_crop_eligible_track=True),
        profile=build_full_frame_profile(width=32, height=32),
        video_path=store.manifest.video_path,
        run_store=store,
    )

    assert [1] in DummyAnnotator.seen_track_ids
    assert [2] not in DummyAnnotator.seen_track_ids


def test_render_allows_unclassified_annotations_for_vehicle_indexed_tracks(
    tmp_path, monkeypatch
) -> None:
    store = DummyRunStore(tmp_path)
    _write_records(
        store.frames_path,
        [
            FrameRecord(
                frame_index=0,
                timestamp_seconds=0.0,
                tracks=[
                    _track(1, 0, vehicle_index=8),
                    _track(2, 0, vehicle_index=None),
                ],
            )
        ],
    )
    writer = DummyWriter()
    DummyAnnotator.seen_track_ids = []
    DummyAnnotator.seen_labels_by_track = []
    _patch_render_io(monkeypatch, writer)

    render_video(
        config=_config(),
        profile=build_full_frame_profile(width=32, height=32),
        video_path=store.manifest.video_path,
        run_store=store,
        allow_unclassified_annotations=True,
    )

    assert [1] in DummyAnnotator.seen_track_ids
    assert [2] not in DummyAnnotator.seen_track_ids
    assert DummyAnnotator.seen_labels_by_track[-1] == {1: "8 | UNKNOWN"}
