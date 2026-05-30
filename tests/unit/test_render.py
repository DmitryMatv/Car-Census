from pathlib import Path
from typing import cast

import numpy as np
import orjson
import pytest

from config import AppConfig, build_full_frame_profile
from models import (
    BBox,
    FrameRecord,
    MMRResult,
    RunManifest,
    TrackedObject,
    TrackSummary,
)
from pipeline import render as render_module
from pipeline.render import (
    _output_fps,
    _resolve_render_frames_path,
    format_label_text,
    render_video,
    visible_track_ids_for_render,
    visible_track_label_text_by_track,
)
from render.annotators import VideoAnnotator
from storage.run_store import RunStore
from utils.video import VideoMetadata


class DummyTracksFile:
    def __init__(self, store: "DummyRunStore") -> None:
        self.store = store

    def read_all(self) -> list[TrackSummary]:
        return list(self.store.track_summaries)


class DummyRunStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.frames_path = root / "analysis" / "frames.jsonl"
        self.render_frames_path = root / "analysis" / "render_frames.jsonl"
        self.tracks_path = root / "analysis" / "tracks.jsonl"
        self.labels_path = root / "mmr" / "labels.json"
        self.output_video_path = root / "annotated.mp4"
        self.labels = self
        self.frames = self
        self.tracks = DummyTracksFile(self)
        self.track_summaries: list[TrackSummary] = []
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

    def read(self) -> dict[int, MMRResult]:
        if not self.labels_path.exists():
            return {}
        raw = orjson.loads(self.labels_path.read_bytes())
        return {
            int(track_id): MMRResult.model_validate(payload)
            for track_id, payload in raw.items()
        }

    def iter(self, *, smoothed: bool = False):
        path = self.render_frames_path if smoothed else self.frames_path
        if not path.exists():
            return
        for line in path.read_bytes().splitlines():
            if line.strip():
                yield FrameRecord.model_validate(orjson.loads(line))

    def read_all(self, *, smoothed: bool = False) -> list[FrameRecord]:
        return list(self.iter(smoothed=smoothed))


class DummyWriter:
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []
        self.released = False

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def release(self) -> None:
        self.released = True


def _as_run_store(store: DummyRunStore) -> RunStore:
    return cast(RunStore, store)


class DummyAnnotator:
    seen_track_ids: list[list[int]] = []
    seen_labels_by_track: list[dict[int, str]] = []
    seen_counter_values: list[int | None] = []

    def __init__(self, config: AppConfig) -> None:
        _ = config

    def annotate(
        self,
        frame: np.ndarray,
        tracks: list[TrackedObject],
        labels_by_track: dict[int, str],
        counter_value: int | None = None,
    ) -> np.ndarray:
        self.seen_track_ids.append([track.track_id for track in tracks])
        self.seen_labels_by_track.append(dict(labels_by_track))
        self.seen_counter_values.append(counter_value)
        return frame


def _track(
    track_id: int,
    frame_index: int = 0,
    vehicle_index: int | None = None,
    bbox: BBox | None = None,
    counted: bool = False,
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
        counted=counted,
    )


def _summary(
    track_id: int,
    max_box_height_px: float,
    vehicle_index: int | None = None,
) -> TrackSummary:
    return TrackSummary(
        track_id=track_id,
        vehicle_index=vehicle_index,
        first_frame_index=0,
        last_frame_index=0,
        frames_seen=1,
        min_box_height_px=max_box_height_px,
        max_box_height_px=max_box_height_px,
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
    smoothing_enabled: bool = False,
) -> AppConfig:
    return AppConfig.model_validate(
        {
            "render": {
                "min_visible_track_observations": min_visible_track_observations,
                "require_crop_eligible_track": require_crop_eligible_track,
                "show_unclassified_tracks": show_unclassified_tracks,
                "smoothing": {"enabled": smoothing_enabled},
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


def test_video_annotator_draws_transparent_rectangle_box() -> None:
    config = AppConfig.model_validate(
        {
            "render": {
                "box_color": "#00FF00",
                "box_alpha": 0.5,
                "box_thickness": 2,
                "counter_enabled": False,
            }
        }
    )
    frame = np.full((120, 120, 3), 255, dtype=np.uint8)

    annotated = VideoAnnotator(config).annotate(
        frame,
        tracks=[_track(1, bbox=BBox(x1=20, y1=20, x2=80, y2=60))],
        labels_by_track={1: "Toyota Corolla"},
    )

    assert np.any(annotated[20:23, 20:81] != frame[20:23, 20:81])


def test_video_annotator_can_disable_rectangle_box() -> None:
    config = AppConfig.model_validate(
        {
            "render": {
                "box_enabled": False,
                "box_color": "#00FF00",
                "counter_enabled": False,
            }
        }
    )
    frame = np.full((140, 140, 3), 255, dtype=np.uint8)

    annotated = VideoAnnotator(config).annotate(
        frame,
        tracks=[_track(1, bbox=BBox(x1=20, y1=20, x2=80, y2=60))],
        labels_by_track={1: "Toyota Corolla"},
    )

    assert np.array_equal(annotated[20:61, 20:81], frame[20:61, 20:81])


def test_video_annotator_draws_counter() -> None:
    config = AppConfig.model_validate(
        {
            "render": {
                "label_bg_color": "#000000",
                "label_text_color": "#FFFFFF",
                "counter_enabled": True,
            }
        }
    )
    frame = np.full((80, 120, 3), 255, dtype=np.uint8)

    annotated = VideoAnnotator(config).annotate(
        frame,
        tracks=[],
        labels_by_track={},
        counter_value=6,
    )

    assert annotated.shape == frame.shape
    assert np.any(annotated[:30, :80] != frame[:30, :80])


def test_video_annotator_clamps_label_below_track_at_bottom_edge() -> None:
    config = AppConfig.model_validate(
        {
            "render": {
                "label_bg_color": "#000000",
                "label_gap_px": 4,
                "label_padding_px": 4,
            }
        }
    )
    frame = np.full((80, 120, 3), 255, dtype=np.uint8)

    annotated = VideoAnnotator(config).annotate(
        frame,
        tracks=[_track(1, bbox=BBox(x1=10, y1=40, x2=70, y2=70))],
        labels_by_track={1: "VW Passat\nMk VI (2010)"},
    )

    label_pixels = np.where(np.any(annotated < 255, axis=2))
    assert int(label_pixels[0].min()) >= 74


def test_video_annotator_does_not_retain_trace_history_between_calls() -> None:
    annotator = VideoAnnotator(AppConfig())
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    first = annotator.annotate(frame, [_track(1)], {1: "1 | Toyota"})
    second = annotator.annotate(frame, [], {})

    assert int(first.max()) > 0
    assert np.array_equal(second, frame)


def test_format_label_text_uses_clean_multiline_label() -> None:
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
        == "Toyota Corolla\nE210\nHybrid"
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

    assert visible_track_label_text_by_track(records, "unknown") == {2: "unknown"}


def test_output_fps_caps_configured_render_fps_at_source_fps() -> None:
    config = AppConfig.model_validate(
        {"video": {"fps": 30.0}, "render": {"output_fps": 60.0}}
    )

    assert _output_fps(config) == 30.0


def test_visible_track_ids_applies_crop_eligibility_when_required() -> None:
    records = [
        FrameRecord(
            frame_index=0,
            timestamp_seconds=0.0,
            tracks=[
                _track(1, vehicle_index=1),
                _track(2, vehicle_index=None),
            ],
        )
    ]

    assert visible_track_ids_for_render(
        _config(require_crop_eligible_track=True),
        records,
    ) == {1}


def test_visible_track_ids_skips_crop_eligibility_when_unclassified_tracks_show() -> (
    None
):
    records = [
        FrameRecord(
            frame_index=0,
            timestamp_seconds=0.0,
            tracks=[
                _track(1, vehicle_index=1),
                _track(2, vehicle_index=None),
            ],
        )
    ]

    assert visible_track_ids_for_render(
        _config(
            require_crop_eligible_track=True,
            show_unclassified_tracks=True,
        ),
        records,
    ) == {1, 2}


def test_visible_track_ids_applies_min_box_height_when_summaries_exist() -> None:
    records = [
        FrameRecord(
            frame_index=0,
            timestamp_seconds=0.0,
            tracks=[
                _track(1, vehicle_index=1),
                _track(2, vehicle_index=2),
            ],
        )
    ]
    summaries = [
        _summary(1, max_box_height_px=160),
        _summary(2, max_box_height_px=159),
    ]

    assert visible_track_ids_for_render(
        AppConfig.model_validate(
            {
                "analysis": {"min_box_height_px": 160},
                "render": {"min_visible_track_observations": 1},
            }
        ),
        records,
        track_summaries=summaries,
    ) == {1}


def test_visible_track_ids_keeps_old_behavior_when_summaries_missing() -> None:
    records = [
        FrameRecord(
            frame_index=0,
            timestamp_seconds=0.0,
            tracks=[
                _track(1, vehicle_index=1),
                _track(2, vehicle_index=2),
            ],
        )
    ]

    assert visible_track_ids_for_render(_config(), records) == {1, 2}


def test_resolve_render_frames_path_returns_raw_path_when_smoothing_disabled(
    tmp_path,
) -> None:
    store = DummyRunStore(tmp_path)

    assert (
        _resolve_render_frames_path(
            _config(smoothing_enabled=False),
            build_full_frame_profile(width=32, height=32),
            _as_run_store(store),
            smooth_render_tracks=None,
        )
        == store.frames_path
    )


def test_resolve_render_frames_path_calls_smoother_when_smoothing_enabled(
    tmp_path,
) -> None:
    store = DummyRunStore(tmp_path)
    calls = 0

    def fake_smoother(config, profile, run_store):
        nonlocal calls
        _ = config, profile
        calls += 1
        assert run_store is store
        return store.render_frames_path

    assert (
        _resolve_render_frames_path(
            _config(smoothing_enabled=True),
            build_full_frame_profile(width=32, height=32),
            _as_run_store(store),
            smooth_render_tracks=fake_smoother,
        )
        == store.render_frames_path
    )
    assert calls == 1


def test_resolve_render_frames_path_requires_smoother_when_smoothing_enabled(
    tmp_path,
) -> None:
    store = DummyRunStore(tmp_path)

    with pytest.raises(ValueError, match="no smoothing stage was provided"):
        _resolve_render_frames_path(
            _config(smoothing_enabled=True),
            build_full_frame_profile(width=32, height=32),
            _as_run_store(store),
            smooth_render_tracks=None,
        )


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
    DummyAnnotator.seen_counter_values = []
    _patch_render_io(monkeypatch, writer)

    render_video(
        config=_config(min_visible_track_observations=2),
        profile=build_full_frame_profile(width=32, height=32),
        video_path=store.manifest.video_path,
        run_store=_as_run_store(store),
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
        run_store=_as_run_store(store),
    )

    assert [1] in DummyAnnotator.seen_track_ids
    assert [2] not in DummyAnnotator.seen_track_ids


def test_render_hides_labeled_track_below_min_box_height(tmp_path, monkeypatch) -> None:
    store = DummyRunStore(tmp_path)
    store.track_summaries = [
        _summary(1, max_box_height_px=120, vehicle_index=1),
        _summary(2, max_box_height_px=180, vehicle_index=2),
    ]
    _write_records(
        store.frames_path,
        [
            FrameRecord(
                frame_index=0,
                timestamp_seconds=0.0,
                tracks=[
                    _track(1, 0, vehicle_index=1),
                    _track(2, 0, vehicle_index=2),
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
        config=AppConfig.model_validate(
            {
                "analysis": {"min_box_height_px": 160},
                "render": {
                    "min_visible_track_observations": 1,
                    "smoothing": {"enabled": False},
                },
            }
        ),
        profile=build_full_frame_profile(width=32, height=32),
        video_path=store.manifest.video_path,
        run_store=_as_run_store(store),
    )

    assert [2] in DummyAnnotator.seen_track_ids
    assert [1] not in DummyAnnotator.seen_track_ids


def test_render_min_box_height_applies_when_showing_unclassified_tracks(
    tmp_path, monkeypatch
) -> None:
    store = DummyRunStore(tmp_path)
    store.track_summaries = [
        _summary(1, max_box_height_px=120, vehicle_index=1),
        _summary(2, max_box_height_px=180, vehicle_index=2),
    ]
    _write_records(
        store.frames_path,
        [
            FrameRecord(
                frame_index=0,
                timestamp_seconds=0.0,
                tracks=[
                    _track(1, 0, vehicle_index=1),
                    _track(2, 0, vehicle_index=2),
                ],
            )
        ],
    )
    writer = DummyWriter()
    DummyAnnotator.seen_track_ids = []
    _patch_render_io(monkeypatch, writer)

    render_video(
        config=AppConfig.model_validate(
            {
                "analysis": {"min_box_height_px": 160},
                "render": {
                    "min_visible_track_observations": 1,
                    "show_unclassified_tracks": True,
                    "smoothing": {"enabled": False},
                },
            }
        ),
        profile=build_full_frame_profile(width=32, height=32),
        video_path=store.manifest.video_path,
        run_store=_as_run_store(store),
    )

    assert [2] in DummyAnnotator.seen_track_ids
    assert [1] not in DummyAnnotator.seen_track_ids


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
        run_store=_as_run_store(store),
        allow_unclassified_annotations=True,
    )

    assert [1] in DummyAnnotator.seen_track_ids
    assert [2] not in DummyAnnotator.seen_track_ids
    assert DummyAnnotator.seen_labels_by_track[-1] == {1: "UNKNOWN"}


def test_render_passes_live_count_to_annotator(tmp_path, monkeypatch) -> None:
    store = DummyRunStore(tmp_path)
    _write_records(
        store.frames_path,
        [
            FrameRecord(
                frame_index=0,
                timestamp_seconds=0.0,
                tracks=[_track(1, 0, vehicle_index=1, counted=True)],
            ),
            FrameRecord(
                frame_index=1,
                timestamp_seconds=1 / 30.0,
                tracks=[
                    _track(1, 1, vehicle_index=1, counted=True),
                    _track(2, 1, vehicle_index=2, counted=True),
                ],
            ),
        ],
    )
    store.labels_path.parent.mkdir(parents=True, exist_ok=True)
    store.labels_path.write_bytes(
        orjson.dumps(
            {
                "1": MMRResult(make="Toyota", vehicle_index=1).model_dump(mode="json"),
                "2": MMRResult(make="VW", vehicle_index=2).model_dump(mode="json"),
            }
        )
    )
    writer = DummyWriter()
    DummyAnnotator.seen_track_ids = []
    DummyAnnotator.seen_labels_by_track = []
    DummyAnnotator.seen_counter_values = []
    _patch_render_io(monkeypatch, writer)

    render_video(
        config=_config(),
        profile=build_full_frame_profile(width=32, height=32),
        video_path=store.manifest.video_path,
        run_store=_as_run_store(store),
    )

    assert DummyAnnotator.seen_counter_values == [1, 2, 2]


def test_render_uses_injected_smoother_when_smoothing_is_enabled(
    tmp_path, monkeypatch
) -> None:
    store = DummyRunStore(tmp_path)
    _write_records(
        store.frames_path,
        [
            FrameRecord(
                frame_index=0,
                timestamp_seconds=0.0,
                tracks=[_track(1, 0, vehicle_index=1)],
            )
        ],
    )
    _write_records(
        store.render_frames_path,
        [
            FrameRecord(
                frame_index=0,
                timestamp_seconds=0.0,
                tracks=[_track(1, 0, vehicle_index=1)],
            )
        ],
    )
    writer = DummyWriter()
    _patch_render_io(monkeypatch, writer)
    calls = 0

    def fake_smoother(config, profile, run_store):
        nonlocal calls
        _ = config, profile
        calls += 1
        assert run_store is store
        return store.render_frames_path

    render_video(
        config=_config(smoothing_enabled=True),
        profile=build_full_frame_profile(width=32, height=32),
        video_path=store.manifest.video_path,
        run_store=_as_run_store(store),
        allow_unclassified_annotations=True,
        smooth_render_tracks=fake_smoother,
    )

    assert calls == 1


def test_render_requires_smoother_when_smoothing_is_enabled(
    tmp_path, monkeypatch
) -> None:
    store = DummyRunStore(tmp_path)
    writer = DummyWriter()
    _patch_render_io(monkeypatch, writer)

    with pytest.raises(ValueError, match="no smoothing stage was provided"):
        render_video(
            config=_config(smoothing_enabled=True),
            profile=build_full_frame_profile(width=32, height=32),
            video_path=store.manifest.video_path,
            run_store=_as_run_store(store),
        )
