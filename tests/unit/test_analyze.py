from collections.abc import Collection
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import orjson
import supervision as sv

from config import AppConfig, CameraProfile, PolygonZoneConfig
from models import BBox, Detection, FrameRecord
from pipeline import analyze as analyze_module
from pipeline.analyze import (
    MutableTrackState,
    _render_bbox_for_track,
    _save_candidate,
    analyze_video,
)
from pipeline.vehicles import staged_track_crop_dir
from storage.run_store import RunStore
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


def test_save_candidate_retains_only_crop_closest_to_target_scale(tmp_path) -> None:
    store = DummyRunStore(tmp_path)
    state = MutableTrackState(
        track_id=42,
        first_frame_index=1,
        last_frame_index=1,
        min_box_height_px=40,
        max_box_height_px=140,
    )
    frame = np.full((200, 200, 3), 255, dtype=np.uint8)
    config = AppConfig.model_validate(
        {
            "analysis": {
                "crop_padding_ratio": 0,
                "crop_target_box_range_ratio": 0.7,
                "crop_min_spacing_seconds": 0,
            }
        }
    )

    _save_candidate(
        store=store,
        track_state=state,
        frame=frame,
        bbox=BBox(x1=10, y1=10, x2=150, y2=150),
        frame_index=5,
        timestamp_seconds=0.5,
        config=config,
    )
    first_path = state.candidates[0].image_path

    _save_candidate(
        store=store,
        track_state=state,
        frame=frame,
        bbox=BBox(x1=20, y1=20, x2=130, y2=130),
        frame_index=6,
        timestamp_seconds=0.6,
        config=config,
    )

    assert len(state.candidates) == 1
    assert state.candidates[0].frame_index == 6
    assert state.candidates[0].vehicle_bbox == BBox(x1=20, y1=20, x2=130, y2=130)
    assert state.candidates[0].vehicle_bbox.height == 110
    assert state.candidates[0].image_path.exists()
    assert not first_path.exists()


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
    def __init__(
        self,
        detections: list[Detection],
        diagnostics: dict[str, object] | None = None,
    ) -> None:
        self.detections = detections
        self.diagnostics = diagnostics or {}
        self.received_batch_sizes: list[int] = []

    def detect(self, frame) -> list[Detection]:
        _ = frame
        return self.detections

    def detect_batch(self, frames) -> list[list[Detection]]:
        self.received_batch_sizes.append(len(frames))
        return [self.detect(frame) for frame in frames]

    def detection_diagnostics(self) -> dict[str, object]:
        return self.diagnostics


class SequenceFakeDetector(FakeDetector):
    def __init__(self, detections_by_frame: list[list[Detection]]) -> None:
        super().__init__([])
        self.detections_by_frame = detections_by_frame
        self.frame_index = 0

    def detect(self, frame) -> list[Detection]:
        _ = frame
        if self.frame_index >= len(self.detections_by_frame):
            return []
        detections = self.detections_by_frame[self.frame_index]
        self.frame_index += 1
        return detections


class FakeTrackerAdapter:
    def __init__(self, tracks: sv.Detections) -> None:
        self.tracks = tracks
        self.received_detections: list[list[Detection]] = []
        self.frame_rate: float | None = None
        self.drop_calls: list[set[int]] = []

    def update(self, detections: list[Detection], frame) -> sv.Detections:
        _ = frame
        self.received_detections.append(detections)
        return self.tracks

    def drop_tracks(self, track_ids: Collection[int]) -> None:
        self.drop_calls.append(set(track_ids))


class SequenceFakeTrackerAdapter(FakeTrackerAdapter):
    def __init__(self, tracks_by_frame: list[sv.Detections]) -> None:
        super().__init__(_empty_tracks())
        self.tracks_by_frame = tracks_by_frame

    def update(self, detections: list[Detection], frame) -> sv.Detections:
        _ = frame
        self.received_detections.append(detections)
        index = len(self.received_detections) - 1
        if index >= len(self.tracks_by_frame):
            return _empty_tracks()
        return self.tracks_by_frame[index]


class DetectionDrivenTrackerAdapter(FakeTrackerAdapter):
    def __init__(self, track_ids_by_update: list[int]) -> None:
        super().__init__(_empty_tracks())
        self.track_ids_by_update = track_ids_by_update
        self.dropped_track_ids: set[int] = set()

    def update(self, detections: list[Detection], frame) -> sv.Detections:
        _ = frame
        self.received_detections.append(detections)
        index = len(self.received_detections) - 1
        if not detections or index >= len(self.track_ids_by_update):
            return _empty_tracks()
        track_id = self.track_ids_by_update[index]
        if track_id in self.dropped_track_ids:
            return _empty_tracks()
        detection = detections[0]
        return sv.Detections(
            xyxy=np.array(
                [
                    [
                        detection.bbox.x1,
                        detection.bbox.y1,
                        detection.bbox.x2,
                        detection.bbox.y2,
                    ]
                ],
                dtype=np.float32,
            ),
            confidence=np.array([detection.confidence], dtype=np.float32),
            class_id=np.array(
                [detection.class_id if detection.class_id is not None else -1],
                dtype=np.int32,
            ),
            tracker_id=np.array([track_id], dtype=np.int32),
            data={"class_name": np.array([detection.class_name or ""], dtype=object)},
        )

    def drop_tracks(self, track_ids: Collection[int]) -> None:
        super().drop_tracks(track_ids)
        self.dropped_track_ids.update(track_ids)


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
    return _prepare_analyze_frames_test(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        detector=detector,
        tracker=tracker,
        frames=[np.full((100, 100, 3), 255, dtype=np.uint8)],
    )


def _prepare_analyze_frames_test(
    tmp_path,
    monkeypatch,
    detector: FakeDetector,
    tracker: FakeTrackerAdapter,
    frames: list[np.ndarray],
) -> RunStore:
    store = RunStore(tmp_path / "run")
    store.ensure_directories()
    frame_count = len(frames)

    monkeypatch.setattr(
        analyze_module,
        "read_video_metadata",
        lambda video_path: VideoMetadata(
            width=100, height=100, fps=30.0, frame_count=frame_count
        ),
    )
    monkeypatch.setattr(
        analyze_module,
        "iter_sampled_frames",
        lambda video_path, source_fps, target_fps: iter(
            [
                (frame_index, frame_index / 30.0, frame)
                for frame_index, frame in enumerate(frames)
            ]
        ),
    )
    monkeypatch.setattr(
        analyze_module, "create_detector", lambda config, project_root: detector
    )

    def create_tracker(config, frame_rate):
        _ = config
        tracker.frame_rate = frame_rate
        return tracker

    monkeypatch.setattr(analyze_module, "BotSortAdapter", create_tracker)
    return store


def _read_frame_records(path: Path) -> list[FrameRecord]:
    return [
        FrameRecord.model_validate(orjson.loads(line))
        for line in path.read_bytes().splitlines()
        if line.strip()
    ]


def _read_detection_stats(store: RunStore) -> dict[str, Any]:
    return orjson.loads(store.detection_stats_path.read_bytes())


def test_analyze_ignores_unconfirmed_tracker_ids(tmp_path, monkeypatch) -> None:
    detector = FakeDetector([])
    tracked = sv.Detections(
        xyxy=np.array([[10, 10, 20, 20], [30, 30, 50, 50]], dtype=np.float32),
        confidence=np.array([0.9, 0.8], dtype=np.float32),
        class_id=np.array([2, 2], dtype=np.int32),
        tracker_id=np.array([-1, 7], dtype=np.int32),
        data={"class_name": np.array(["car", "car"], dtype=object)},
    )
    tracker = FakeTrackerAdapter(tracked)
    store = _prepare_analyze_test(tmp_path, monkeypatch, detector, tracker)
    config = AppConfig.model_validate({"tracker": {"ignore_edge_touches": False}})

    analyze_video(
        project_root=tmp_path,
        config=config,
        profile=CameraProfile(
            camera_id="full",
            polygon=PolygonZoneConfig(points=[[0, 0], [99, 0], [99, 99], [0, 99]]),
        ),
        video_path=tmp_path / "input.mp4",
        run_store=store,
    )

    records = _read_frame_records(store.frames_path)
    assert [track.track_id for track in records[0].tracks] == [7]


def test_analyze_batches_detection_but_updates_tracker_in_frame_order(
    tmp_path, monkeypatch
) -> None:
    frames = [np.full((100, 100, 3), value, dtype=np.uint8) for value in [40, 80, 120]]
    detector = FakeDetector(
        [
            Detection(
                bbox=BBox(x1=20, y1=20, x2=80, y2=80),
                confidence=0.9,
                class_id=2,
                class_name="car",
            )
        ]
    )

    class OrderedTracker(FakeTrackerAdapter):
        def update(self, detections: list[Detection], frame) -> sv.Detections:
            self.received_detections.append(detections)
            frame_value = int(frame[0, 0, 0])
            track_id = {40: 1, 80: 2, 120: 3}[frame_value]
            return sv.Detections(
                xyxy=np.array([[20, 20, 80, 80]], dtype=np.float32),
                confidence=np.array([0.9], dtype=np.float32),
                class_id=np.array([2], dtype=np.int32),
                tracker_id=np.array([track_id], dtype=np.int32),
                data={"class_name": np.array(["car"], dtype=object)},
            )

    tracker = OrderedTracker(_empty_tracks())
    store = RunStore(tmp_path / "run")
    store.ensure_directories()
    monkeypatch.setattr(
        analyze_module,
        "read_video_metadata",
        lambda video_path: VideoMetadata(
            width=100, height=100, fps=30.0, frame_count=3
        ),
    )
    monkeypatch.setattr(
        analyze_module,
        "iter_sampled_frames",
        lambda video_path, source_fps, target_fps: iter(
            [
                (0, 0.0, frames[0]),
                (1, 1 / 30, frames[1]),
                (2, 2 / 30, frames[2]),
            ]
        ),
    )
    monkeypatch.setattr(
        analyze_module, "create_detector", lambda config, project_root: detector
    )

    def create_tracker(config, frame_rate):
        _ = config
        tracker.frame_rate = frame_rate
        return tracker

    monkeypatch.setattr(analyze_module, "BotSortAdapter", create_tracker)

    analyze_video(
        project_root=tmp_path,
        config=AppConfig.model_validate(
            {
                "analysis": {"batch_size": 2, "min_track_frames": 1},
                "tracker": {"ignore_edge_touches": False},
            }
        ),
        profile=CameraProfile(
            camera_id="full",
            polygon=PolygonZoneConfig(points=[[0, 0], [99, 0], [99, 99], [0, 99]]),
        ),
        video_path=tmp_path / "input.mp4",
        run_store=store,
    )

    records = _read_frame_records(store.frames_path)
    assert detector.received_batch_sizes == [2, 1]
    assert [record.frame_index for record in records] == [0, 1, 2]
    assert [record.tracks[0].track_id for record in records] == [1, 2, 3]
    assert len(tracker.received_detections) == 3


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


def test_analyze_passes_edge_detection_to_tracker_then_kills_source_edge_track(
    tmp_path, monkeypatch
) -> None:
    detector = SequenceFakeDetector(
        [
            [
                Detection(
                    bbox=BBox(x1=20, y1=20, x2=60, y2=60),
                    confidence=0.9,
                    class_id=2,
                    class_name="car",
                )
            ],
            [
                Detection(
                    bbox=BBox(x1=0, y1=20, x2=60, y2=60),
                    confidence=0.9,
                    class_id=2,
                    class_name="car",
                )
            ],
            [
                Detection(
                    bbox=BBox(x1=30, y1=20, x2=70, y2=60),
                    confidence=0.9,
                    class_id=2,
                    class_name="car",
                )
            ],
        ]
    )
    tracker = DetectionDrivenTrackerAdapter([7, 7, 7])
    store = _prepare_analyze_frames_test(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        detector=detector,
        tracker=tracker,
        frames=[np.full((100, 100, 3), 255, dtype=np.uint8) for _ in range(3)],
    )

    analyze_video(
        project_root=tmp_path,
        config=AppConfig.model_validate({"analysis": {"min_track_frames": 1}}),
        profile=CameraProfile(
            camera_id="full",
            polygon=PolygonZoneConfig(points=[[0, 0], [99, 0], [99, 99], [0, 99]]),
        ),
        video_path=tmp_path / "input.mp4",
        run_store=store,
    )

    records = _read_frame_records(store.frames_path)
    assert [[track.track_id for track in record.tracks] for record in records] == [
        [7],
        [],
        [],
    ]
    assert tracker.drop_calls == [{7}]
    assert [detection.bbox for detection in tracker.received_detections[1]] == [
        BBox(x1=0, y1=20, x2=60, y2=60)
    ]


def test_analyze_passes_edge_detection_to_tracker_then_kills_roi_crop_edge_track(
    tmp_path, monkeypatch
) -> None:
    detector = SequenceFakeDetector(
        [
            [
                Detection(
                    bbox=BBox(x1=20, y1=20, x2=50, y2=50),
                    confidence=0.9,
                    class_id=2,
                    class_name="car",
                )
            ],
            [
                Detection(
                    bbox=BBox(x1=0, y1=20, x2=50, y2=50),
                    confidence=0.9,
                    class_id=2,
                    class_name="car",
                )
            ],
            [
                Detection(
                    bbox=BBox(x1=25, y1=25, x2=55, y2=55),
                    confidence=0.9,
                    class_id=2,
                    class_name="car",
                )
            ],
        ]
    )
    tracker = DetectionDrivenTrackerAdapter([7, 7, 7])
    store = _prepare_analyze_frames_test(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        detector=detector,
        tracker=tracker,
        frames=[np.full((100, 100, 3), 255, dtype=np.uint8) for _ in range(3)],
    )

    analyze_video(
        project_root=tmp_path,
        config=AppConfig.model_validate({"analysis": {"min_track_frames": 1}}),
        profile=CameraProfile(
            camera_id="inner-square",
            polygon=PolygonZoneConfig(points=[[10, 10], [90, 10], [90, 90], [10, 90]]),
        ),
        video_path=tmp_path / "input.mp4",
        run_store=store,
    )

    records = _read_frame_records(store.frames_path)
    assert [[track.track_id for track in record.tracks] for record in records] == [
        [7],
        [],
        [],
    ]
    assert tracker.drop_calls == [{7}]
    assert [detection.bbox for detection in tracker.received_detections[1]] == [
        BBox(x1=10, y1=30, x2=60, y2=60)
    ]


def test_analyze_passes_edge_detection_to_tracker_then_kills_polygon_edge_track(
    tmp_path, monkeypatch
) -> None:
    detector = SequenceFakeDetector(
        [
            [
                Detection(
                    bbox=BBox(x1=40, y1=30, x2=50, y2=40),
                    confidence=0.9,
                    class_id=2,
                    class_name="car",
                )
            ],
            [
                Detection(
                    bbox=BBox(x1=38, y1=8, x2=46, y2=18),
                    confidence=0.9,
                    class_id=2,
                    class_name="car",
                )
            ],
            [
                Detection(
                    bbox=BBox(x1=50, y1=30, x2=60, y2=40),
                    confidence=0.9,
                    class_id=2,
                    class_name="car",
                )
            ],
        ]
    )
    tracker = DetectionDrivenTrackerAdapter([7, 7, 7])
    store = _prepare_analyze_frames_test(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        detector=detector,
        tracker=tracker,
        frames=[np.full((100, 100, 3), 255, dtype=np.uint8) for _ in range(3)],
    )

    analyze_video(
        project_root=tmp_path,
        config=AppConfig.model_validate({"analysis": {"min_track_frames": 1}}),
        profile=_profile_with_slanted_polygon(),
        video_path=tmp_path / "input.mp4",
        run_store=store,
    )

    records = _read_frame_records(store.frames_path)
    assert [[track.track_id for track in record.tracks] for record in records] == [
        [7],
        [],
        [],
    ]
    assert tracker.drop_calls == [{7}]
    assert [detection.bbox for detection in tracker.received_detections[1]] == [
        BBox(x1=48, y1=18, x2=56, y2=28)
    ]


def test_analyze_allows_new_track_id_after_edge_kill(tmp_path, monkeypatch) -> None:
    detector = SequenceFakeDetector(
        [
            [
                Detection(
                    bbox=BBox(x1=20, y1=20, x2=60, y2=60),
                    confidence=0.9,
                    class_id=2,
                    class_name="car",
                )
            ],
            [
                Detection(
                    bbox=BBox(x1=0, y1=20, x2=60, y2=60),
                    confidence=0.9,
                    class_id=2,
                    class_name="car",
                )
            ],
            [
                Detection(
                    bbox=BBox(x1=30, y1=20, x2=70, y2=60),
                    confidence=0.9,
                    class_id=2,
                    class_name="car",
                )
            ],
        ]
    )
    tracker = DetectionDrivenTrackerAdapter([7, 7, 8])
    store = _prepare_analyze_frames_test(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        detector=detector,
        tracker=tracker,
        frames=[np.full((100, 100, 3), 255, dtype=np.uint8) for _ in range(3)],
    )

    analyze_video(
        project_root=tmp_path,
        config=AppConfig.model_validate({"analysis": {"min_track_frames": 1}}),
        profile=CameraProfile(
            camera_id="full",
            polygon=PolygonZoneConfig(points=[[0, 0], [99, 0], [99, 99], [0, 99]]),
        ),
        video_path=tmp_path / "input.mp4",
        run_store=store,
    )

    records = _read_frame_records(store.frames_path)
    assert [[track.track_id for track in record.tracks] for record in records] == [
        [7],
        [],
        [8],
    ]
    assert tracker.drop_calls == [{7}]


def test_edge_killed_track_does_not_save_crop_on_kill_frame(
    tmp_path, monkeypatch
) -> None:
    detector = SequenceFakeDetector(
        [
            [
                Detection(
                    bbox=BBox(x1=20, y1=20, x2=70, y2=70),
                    confidence=0.9,
                    class_id=2,
                    class_name="car",
                )
            ],
            [
                Detection(
                    bbox=BBox(x1=0, y1=0, x2=99, y2=99),
                    confidence=0.9,
                    class_id=2,
                    class_name="car",
                )
            ],
        ]
    )
    tracker = DetectionDrivenTrackerAdapter([7, 7])
    store = _prepare_analyze_frames_test(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        detector=detector,
        tracker=tracker,
        frames=[np.full((100, 100, 3), 255, dtype=np.uint8) for _ in range(2)],
    )
    config = AppConfig.model_validate(
        {
            "analysis": {
                "min_track_frames": 1,
                "min_box_height_px": 1,
                "crop_padding_ratio": 0,
                "crop_padding_px": 0,
                "crop_min_spacing_seconds": 0,
            }
        }
    )

    analyze_video(
        project_root=tmp_path,
        config=config,
        profile=CameraProfile(
            camera_id="full",
            polygon=PolygonZoneConfig(points=[[0, 0], [99, 0], [99, 99], [0, 99]]),
        ),
        video_path=tmp_path / "input.mp4",
        run_store=store,
    )

    summaries = [
        orjson.loads(line)
        for line in store.tracks_path.read_bytes().splitlines()
        if line.strip()
    ]
    assert len(summaries) == 1
    assert [candidate["frame_index"] for candidate in summaries[0]["candidates"]] == [0]
    assert tracker.drop_calls == [{7}]


def test_analyze_no_longer_filters_edge_detections_before_tracker(
    tmp_path, monkeypatch
) -> None:
    detector = SequenceFakeDetector(
        [
            [
                Detection(
                    bbox=BBox(x1=38, y1=8, x2=46, y2=18),
                    confidence=0.9,
                    class_id=2,
                    class_name="car",
                )
            ]
        ]
    )
    tracker = DetectionDrivenTrackerAdapter([7])
    store = _prepare_analyze_test(tmp_path, monkeypatch, detector, tracker)

    analyze_video(
        project_root=tmp_path,
        config=AppConfig(),
        profile=_profile_with_slanted_polygon(),
        video_path=tmp_path / "input.mp4",
        run_store=store,
    )

    records = _read_frame_records(store.frames_path)
    assert records[0].tracks == []
    assert [detection.bbox for detection in tracker.received_detections[0]] == [
        BBox(x1=48, y1=18, x2=56, y2=28)
    ]
    assert tracker.drop_calls == [{7}]


def test_analyze_writes_detector_and_tracker_diagnostics(tmp_path, monkeypatch) -> None:
    detector = FakeDetector(
        [
            Detection(
                bbox=BBox(x1=20, y1=20, x2=80, y2=80),
                confidence=0.9,
                class_id=2,
                class_name="car",
            )
        ],
        diagnostics={
            "counts": {
                "raw_candidate_rows": 5,
                "detections_after_confidence_filtering": 3,
                "detections_after_class_filtering": 1,
            },
            "confidence_values": [0.35, 0.9],
        },
    )
    tracker = DetectionDrivenTrackerAdapter([7])
    store = _prepare_analyze_test(tmp_path, monkeypatch, detector, tracker)

    analyze_video(
        project_root=tmp_path,
        config=AppConfig.model_validate(
            {
                "analysis": {"min_track_frames": 1, "min_box_height_px": 1},
                "tracker": {"ignore_edge_touches": False},
                "render": {"min_visible_track_observations": 1},
            }
        ),
        profile=CameraProfile(
            camera_id="full",
            polygon=PolygonZoneConfig(points=[[0, 0], [99, 0], [99, 99], [0, 99]]),
        ),
        video_path=tmp_path / "input.mp4",
        run_store=store,
    )

    stats = _read_detection_stats(store)
    assert stats["total_sampled_frames"] == 1
    assert stats["raw_detections_before_class_filtering"] == 5
    assert stats["detections_after_confidence_filtering"] == 3
    assert stats["detections_after_class_filtering"] == 1
    assert stats["detections_passed_to_tracker"] == 1
    assert stats["tracker_outputs"] == 1
    assert stats["tracks_discarded_edge_contact"] == 0
    assert stats["tracks_discarded_min_track_frames"] == 0
    assert stats["tracks_without_crop_candidates"] == 0
    assert stats["tracks_hidden_from_render_due_to_crop_eligibility"] == 0
    assert sum(bucket["count"] for bucket in stats["confidence_histogram"]) == 2
    assert sum(bucket["count"] for bucket in stats["box_height_histogram"]) == 1


def test_analyze_diagnostics_count_edge_short_and_crop_ineligible_tracks(
    tmp_path, monkeypatch
) -> None:
    detector = SequenceFakeDetector(
        [
            [
                Detection(
                    bbox=BBox(x1=20, y1=20, x2=80, y2=80),
                    confidence=0.9,
                    class_id=2,
                    class_name="car",
                )
            ],
            [
                Detection(
                    bbox=BBox(x1=15, y1=15, x2=75, y2=75),
                    confidence=0.9,
                    class_id=2,
                    class_name="car",
                )
            ],
            [
                Detection(
                    bbox=BBox(x1=0, y1=20, x2=60, y2=60),
                    confidence=0.9,
                    class_id=2,
                    class_name="car",
                )
            ],
        ]
    )
    tracker = DetectionDrivenTrackerAdapter([7, 8, 8])
    store = _prepare_analyze_frames_test(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        detector=detector,
        tracker=tracker,
        frames=[np.full((100, 100, 3), 255, dtype=np.uint8) for _ in range(3)],
    )

    analyze_video(
        project_root=tmp_path,
        config=AppConfig.model_validate(
            {
                "analysis": {"min_track_frames": 2, "min_box_height_px": 100},
                "render": {
                    "require_crop_eligible_track": True,
                    "min_visible_track_observations": 1,
                },
            }
        ),
        profile=CameraProfile(
            camera_id="full",
            polygon=PolygonZoneConfig(points=[[0, 0], [99, 0], [99, 99], [0, 99]]),
        ),
        video_path=tmp_path / "input.mp4",
        run_store=store,
    )

    stats = _read_detection_stats(store)
    assert stats["tracks_discarded_edge_contact"] == 1
    assert stats["tracks_discarded_min_track_frames"] == 2
    assert stats["tracks_without_crop_candidates"] == 2
    assert stats["tracks_hidden_from_render_due_to_crop_eligibility"] == 2


def test_analyze_uses_bottom_center_for_roi_membership(tmp_path, monkeypatch) -> None:
    detector = FakeDetector([])
    tracker = FakeTrackerAdapter(_single_track(BBox(x1=40, y1=20, x2=60, y2=80)))
    store = _prepare_analyze_test(tmp_path, monkeypatch, detector, tracker)
    config = AppConfig.model_validate(
        {
            "analysis": {"crop_padding_ratio": 0, "crop_padding_px": 0},
            "tracker": {"ignore_edge_touches": False},
        }
    )

    analyze_video(
        project_root=tmp_path,
        config=config,
        profile=CameraProfile(
            camera_id="lower-band",
            polygon=PolygonZoneConfig(points=[[0, 60], [99, 60], [99, 99], [0, 99]]),
        ),
        video_path=tmp_path / "input.mp4",
        run_store=store,
    )

    records = _read_frame_records(store.frames_path)
    assert records[0].tracks[0].centroid == (50.0, 50.0)
    assert records[0].tracks[0].bottom_center == (50.0, 80.0)
    assert records[0].tracks[0].inside_roi is True


def test_analyze_discards_crops_and_vehicle_index_for_short_tracks(
    tmp_path, monkeypatch
) -> None:
    detector = FakeDetector([])
    tracker = FakeTrackerAdapter(_single_track(BBox(x1=20, y1=20, x2=80, y2=80)))
    store = _prepare_analyze_test(tmp_path, monkeypatch, detector, tracker)
    config = AppConfig.model_validate(
        {"analysis": {"min_track_frames": 2, "min_box_height_px": 1}}
    )

    analyze_video(
        project_root=tmp_path,
        config=config,
        profile=CameraProfile(
            camera_id="full",
            polygon=PolygonZoneConfig(points=[[0, 0], [99, 0], [99, 99], [0, 99]]),
        ),
        video_path=tmp_path / "input.mp4",
        run_store=store,
    )

    records = _read_frame_records(store.frames_path)
    summaries = [
        orjson.loads(line)
        for line in store.tracks_path.read_bytes().splitlines()
        if line.strip()
    ]
    crop_files = [
        path
        for path in store.crops_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".jpg"
    ]
    assert records[0].tracks[0].vehicle_index is None
    assert summaries[0]["vehicle_index"] is None
    assert summaries[0]["candidates"] == []
    assert crop_files == []
