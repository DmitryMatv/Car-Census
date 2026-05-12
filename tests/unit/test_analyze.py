from pathlib import Path

import cv2
import numpy as np
import orjson
import supervision as sv

from config import AppConfig, CameraProfile, PolygonZoneConfig
from pipeline import analyze as analyze_module
from pipeline.analyze import (
    MutableTrackState,
    _render_bbox_for_track,
    analyze_video,
    _save_candidate,
)
from pipeline.vehicles import staged_track_crop_dir
from storage.run_store import RunStore
from models import BBox, Detection, FrameRecord
from utils.video import VideoMetadata


class DummyRunStore:
    def __init__(self, root: Path) -> None:
        self.crops_dir = root / "crops"
        self.crops_dir.mkdir(parents=True)


def test_save_candidate_stages_crop_under_track_identity(tmp_path) -> None:
    store = DummyRunStore(tmp_path)
    state = MutableTrackState(
        track_id=42,
        first_frame_index=1,
        last_frame_index=1,
    )
    frame = np.full((20, 20, 3), 255, dtype=np.uint8)

    _save_candidate(
        store=store,
        track_state=state,
        frame=frame,
        bbox=BBox(x1=2, y1=3, x2=12, y2=13),
        frame_index=5,
        timestamp_seconds=0.5,
        config=AppConfig(),
    )

    assert len(state.candidates) == 1
    assert state.candidates[0].track_id == 42
    assert state.candidates[0].vehicle_index is None
    assert state.candidates[0].image_path.parent == staged_track_crop_dir(
        store.crops_dir, 42
    )
    assert state.candidates[0].image_path.exists()
    assert not (store.crops_dir / "track_000042").exists()


def test_save_candidate_pads_crop_for_classification(tmp_path) -> None:
    store = DummyRunStore(tmp_path)
    state = MutableTrackState(
        track_id=42,
        first_frame_index=1,
        last_frame_index=1,
    )
    frame = np.full((30, 30, 3), 255, dtype=np.uint8)
    config = AppConfig.model_validate(
        {"analysis": {"crop_padding_ratio": 0.1, "crop_padding_px": 2}}
    )

    _save_candidate(
        store=store,
        track_state=state,
        frame=frame,
        bbox=BBox(x1=10, y1=10, x2=20, y2=20),
        frame_index=5,
        timestamp_seconds=0.5,
        config=config,
    )

    assert len(state.candidates) == 1
    assert state.candidates[0].bbox == BBox(x1=7, y1=7, x2=23, y2=23)
    saved_crop = cv2.imread(str(state.candidates[0].image_path))
    assert saved_crop.shape[:2] == (16, 16)


def test_render_bbox_uses_same_padding_as_crop_candidates() -> None:
    config = AppConfig.model_validate(
        {"analysis": {"crop_padding_ratio": 0.1, "crop_padding_px": 2}}
    )

    bbox = _render_bbox_for_track(
        BBox(x1=10, y1=10, x2=20, y2=20),
        (30, 30, 3),
        config,
    )

    assert bbox == BBox(x1=7, y1=7, x2=23, y2=23)


class FakeDetector:
    def __init__(self, detections: list[Detection]) -> None:
        self.detections = detections

    def detect(self, frame) -> list[Detection]:
        _ = frame
        return self.detections


class FakeTrackerAdapter:
    def __init__(self, tracks: sv.Detections) -> None:
        self.tracks = tracks
        self.received_detections: list[list[Detection]] = []
        self.frame_rate: float | None = None

    def update(self, detections: list[Detection], frame) -> sv.Detections:
        _ = frame
        self.received_detections.append(detections)
        return self.tracks


def _empty_tracks() -> sv.Detections:
    return sv.Detections(
        xyxy=np.empty((0, 4), dtype=np.float32),
        confidence=np.empty((0,), dtype=np.float32),
        class_id=np.empty((0,), dtype=np.int32),
        tracker_id=np.empty((0,), dtype=np.int32),
        data={"class_name": np.empty((0,), dtype=object)},
    )


def _single_track(bbox: BBox) -> sv.Detections:
    return sv.Detections(
        xyxy=np.array([[bbox.x1, bbox.y1, bbox.x2, bbox.y2]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([2], dtype=np.int32),
        tracker_id=np.array([7], dtype=np.int32),
        data={"class_name": np.array(["car"], dtype=object)},
    )


def _profile_with_slanted_polygon() -> CameraProfile:
    return CameraProfile(
        camera_id="slanted",
        polygon=PolygonZoneConfig(points=[[10, 10], [90, 30], [90, 90], [10, 90]]),
    )


def _prepare_analyze_test(
    tmp_path,
    monkeypatch,
    detector: FakeDetector,
    tracker: FakeTrackerAdapter,
) -> RunStore:
    store = RunStore(tmp_path / "run")
    store.ensure_directories()
    frame = np.full((100, 100, 3), 255, dtype=np.uint8)

    monkeypatch.setattr(
        analyze_module,
        "read_video_metadata",
        lambda video_path: VideoMetadata(
            width=100, height=100, fps=30.0, frame_count=1
        ),
    )
    monkeypatch.setattr(
        analyze_module,
        "iter_sampled_frames",
        lambda video_path, source_fps, target_fps: iter([(0, 0.0, frame)]),
    )
    monkeypatch.setattr(
        analyze_module, "create_detector", lambda config, project_root: detector
    )
    monkeypatch.setattr(
        analyze_module,
        "BotSortAdapter",
        lambda config, frame_rate: (
            setattr(tracker, "frame_rate", frame_rate) or tracker
        ),
    )
    return store


def _read_frame_records(path: Path) -> list[FrameRecord]:
    return [
        FrameRecord.model_validate(orjson.loads(line))
        for line in path.read_bytes().splitlines()
        if line.strip()
    ]


def test_analyze_suppresses_tracker_output_touching_polygon_edge(
    tmp_path, monkeypatch
) -> None:
    detector = FakeDetector([])
    tracker = FakeTrackerAdapter(_single_track(BBox(x1=48, y1=18, x2=56, y2=28)))
    store = _prepare_analyze_test(tmp_path, monkeypatch, detector, tracker)

    analyze_video(
        project_root=tmp_path,
        config=AppConfig(),
        profile=_profile_with_slanted_polygon(),
        video_path=tmp_path / "input.mp4",
        run_store=store,
    )

    records = _read_frame_records(store.frames_path)
    manifest = store.read_manifest()
    assert len(records) == 1
    assert records[0].tracks == []
    assert tracker.frame_rate == 10.0
    assert manifest.source_fps == 30.0
    assert manifest.analysis_fps == 10.0


def test_analyze_filters_detection_touching_polygon_edge_before_tracker(
    tmp_path, monkeypatch
) -> None:
    detector = FakeDetector(
        [
            Detection(
                bbox=BBox(x1=38, y1=8, x2=46, y2=18),
                confidence=0.9,
                class_id=2,
                class_name="car",
            )
        ]
    )
    tracker = FakeTrackerAdapter(_empty_tracks())
    store = _prepare_analyze_test(tmp_path, monkeypatch, detector, tracker)

    analyze_video(
        project_root=tmp_path,
        config=AppConfig(),
        profile=_profile_with_slanted_polygon(),
        video_path=tmp_path / "input.mp4",
        run_store=store,
    )

    assert tracker.received_detections == [[]]
