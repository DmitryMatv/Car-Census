from pathlib import Path

import numpy as np
import orjson

from car_census.config import AppConfig, build_full_frame_profile
from car_census.pipeline import render as render_module
from car_census.pipeline.render import render_video
from car_census.types import BBox, FrameRecord, RunManifest, TrackedObject
from car_census.utils.video import VideoMetadata


class DummyRunStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.frames_path = root / "analysis" / "frames.jsonl"
        self.render_frames_path = root / "analysis" / "render_frames.jsonl"
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

    def __init__(self, config: AppConfig) -> None:
        _ = config

    def annotate(self, frame, profile, tracks, labels_by_track):
        _ = profile
        _ = labels_by_track
        self.seen_track_ids.append([track.track_id for track in tracks])
        return frame


def _track(track_id: int, frame_index: int, timestamp: float) -> TrackedObject:
    bbox = BBox(x1=1, y1=2, x2=11, y2=12)
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
