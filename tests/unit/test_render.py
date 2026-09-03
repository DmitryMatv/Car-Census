import queue
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
import orjson
import pytest

from config import AppConfig, build_full_frame_profile
from mmr.powertrain_catalog import PowertrainClass, VehicleIdentity
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
    _label_text_and_colors_by_track,
    _make_statistics_rows,
    _output_fps,
    _resolve_render_frames_path,
    format_label_text,
    render_video,
    visible_track_ids_for_render,
    visible_track_label_text_by_track,
)
from render import annotators as annotators_module
from render.annotators import MakeStatisticRow, VideoAnnotator
from storage.run_store import RunStore
from utils.video import FrameWriter, VideoMetadata


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
        self.tracks_effective = self.tracks
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


class _CountingWriter:
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []
        self.release_calls = 0

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def release(self) -> None:
        self.release_calls += 1


class _WriteFailingWriter(_CountingWriter):
    def write(self, frame: np.ndarray) -> None:
        _ = frame
        raise RuntimeError("write failed")


class _ReleaseFailingWriter(_CountingWriter):
    def release(self) -> None:
        self.release_calls += 1
        raise RuntimeError("finalize failed")


def _as_run_store(store: DummyRunStore) -> RunStore:
    return cast(RunStore, store)


class DummyAnnotator:
    seen_track_ids: list[list[int]] = []
    seen_labels_by_track: list[dict[int, str]] = []
    seen_label_text_colors_by_track: list[dict[int, str]] = []
    seen_counter_values: list[int | None] = []
    seen_make_statistics_rows: list[list[MakeStatisticRow]] = []

    def __init__(self, config: AppConfig) -> None:
        _ = config

    def annotate(
        self,
        frame: np.ndarray,
        tracks: list[TrackedObject],
        labels_by_track: dict[int, str],
        counter_value: int | None = None,
        label_text_colors_by_track: dict[int, str] | None = None,
        make_statistics_rows: Sequence[MakeStatisticRow] = (),
    ) -> np.ndarray:
        self.seen_track_ids.append([track.track_id for track in tracks])
        self.seen_labels_by_track.append(dict(labels_by_track))
        self.seen_label_text_colors_by_track.append(
            dict(label_text_colors_by_track or {})
        )
        self.seen_counter_values.append(counter_value)
        self.seen_make_statistics_rows.append(list(make_statistics_rows))
        return frame


def _reset_dummy_annotator() -> None:
    DummyAnnotator.seen_track_ids = []
    DummyAnnotator.seen_labels_by_track = []
    DummyAnnotator.seen_label_text_colors_by_track = []
    DummyAnnotator.seen_counter_values = []
    DummyAnnotator.seen_make_statistics_rows = []


def _run_threaded_render_pipeline(
    default_config,
    writer: FrameWriter,
    *,
    frame_count: int,
) -> None:
    frames = iter(
        (
            index,
            index / 30.0,
            np.full((4, 4, 3), index, dtype=np.uint8),
        )
        for index in range(frame_count)
    )
    render_module._render_annotated_frames_pipeline(
        frame_iter=frames,
        record_iter=iter(()),
        annotator=DummyAnnotator(default_config),
        writer=writer,
        label_text={},
        label_text_colors={},
        accepted_labels_by_track={},
        origin_country_by_make={},
        visible_track_ids=set(),
        num_workers=1,
    )


def test_threaded_render_pipeline_writes_all_frames_and_releases_once(
    default_config,
) -> None:
    writer = _CountingWriter()

    _run_threaded_render_pipeline(default_config, writer, frame_count=3)

    assert [int(frame[0, 0, 0]) for frame in writer.frames] == [0, 1, 2]
    assert writer.release_calls == 1


def test_threaded_render_pipeline_propagates_write_failure(default_config) -> None:
    writer = _WriteFailingWriter()

    with pytest.raises(RuntimeError, match="write failed"):
        _run_threaded_render_pipeline(default_config, writer, frame_count=1)

    assert writer.release_calls == 1


def test_threaded_render_pipeline_propagates_finalize_failure(default_config) -> None:
    writer = _ReleaseFailingWriter()

    with pytest.raises(RuntimeError, match="finalize failed"):
        _run_threaded_render_pipeline(default_config, writer, frame_count=0)

    assert writer.release_calls == 1


def test_queue_writer_release_does_not_block_cancelled_full_queue() -> None:
    encode_q: queue.Queue[np.ndarray | object] = queue.Queue(maxsize=1)
    encode_q.put(np.zeros((1, 1, 3), dtype=np.uint8))
    cancel = threading.Event()
    cancel.set()
    writer = render_module._QueueFrameWriter(encode_q, cancel)
    release_thread = threading.Thread(target=writer.release, daemon=True)

    release_thread.start()
    release_thread.join(timeout=1.0)

    assert not release_thread.is_alive()
    assert encode_q.qsize() == 1


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
    max_box_width_px: float,
    vehicle_index: int | None = None,
    max_box_height_px: float | None = None,
) -> TrackSummary:
    return TrackSummary(
        track_id=track_id,
        vehicle_index=vehicle_index,
        first_frame_index=0,
        last_frame_index=0,
        frames_seen=1,
        min_box_width_px=max_box_width_px,
        max_box_width_px=max_box_width_px,
        max_box_height_px=max_box_height_px,
    )


def _write_records(path: Path, records: list[FrameRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"".join(
            orjson.dumps(record.model_dump(mode="json")) + b"\n" for record in records
        )
    )


def _render_config(
    config_factory,
    min_visible_track_observations: int = 1,
    require_crop_eligible_track: bool = False,
    show_unclassified_tracks: bool = False,
    smoothing_enabled: bool = False,
) -> AppConfig:
    return config_factory(
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
    monkeypatch.setattr(
        render_module,
        "load_default_make_country_catalog",
        lambda: {"Toyota": "🇯🇵", "VW": "🇩🇪"},
    )
    monkeypatch.setattr(
        render_module,
        "load_default_powertrain_catalog",
        lambda: {
            VehicleIdentity("Tesla", "Model 3", "Mk I (2018)"): PowertrainClass.BEV,
            VehicleIdentity(
                "Hyundai", "Kona", "Mk I (2017) ~ Mk I EV (2019)"
            ): PowertrainClass.MIXED,
        },
    )


def test_video_annotator_returns_same_shape_frame(default_config) -> None:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    annotated = VideoAnnotator(default_config).annotate(
        frame,
        tracks=[_track(1)],
        labels_by_track={1: "1 | Toyota Corolla"},
    )

    assert annotated.shape == frame.shape


def test_video_annotator_no_tracks_returns_unchanged_copy(default_config) -> None:
    frame = np.full((32, 32, 3), 7, dtype=np.uint8)
    annotated = VideoAnnotator(default_config).annotate(frame, [], {})

    assert np.array_equal(annotated, frame)
    assert annotated is not frame


def test_video_annotator_visible_track_produces_nonzero_pixels(default_config) -> None:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    annotated = VideoAnnotator(default_config).annotate(
        frame,
        tracks=[_track(1)],
        labels_by_track={1: "1 | Toyota Corolla"},
    )

    assert int(annotated.max()) > 0


def test_video_annotator_handles_missing_labels_with_unknown_fallback(
    default_config,
) -> None:
    annotated = VideoAnnotator(default_config).annotate(
        np.zeros((64, 64, 3), dtype=np.uint8),
        tracks=[_track(1)],
        labels_by_track={},
    )
    assert isinstance(annotated, np.ndarray)


def test_video_annotator_accepts_fixed_configured_colors(config_factory) -> None:
    config = config_factory(
        {
            "render": {
                "box_enabled": True,
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


def test_video_annotator_draws_transparent_rectangle_box(config_factory) -> None:
    config = config_factory(
        {
            "render": {
                "box_enabled": True,
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


def test_video_annotator_can_disable_rectangle_box(config_factory) -> None:
    config = config_factory(
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


def test_video_annotator_draws_counter(config_factory) -> None:
    config = config_factory(
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


def test_video_annotator_draws_make_statistics(
    config_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drawn_text: list[str] = []

    def capture_put_text(
        img,
        text,
        org,
        font_face,
        font_scale,
        color,
        thickness,
        line_type,
    ):
        _ = org, font_face, font_scale, color, thickness, line_type
        drawn_text.append(text)
        return img

    def render_test_flag(flag: str, target_height: int) -> np.ndarray:
        _ = flag
        image = np.zeros((target_height, target_height, 4), dtype=np.uint8)
        image[:, :, 0] = 255
        image[:, :, 3] = 255
        return image

    monkeypatch.setattr(annotators_module.cv2, "putText", capture_put_text)
    monkeypatch.setattr(annotators_module, "_render_flag_emoji", render_test_flag)

    config = config_factory(
        {
            "render": {
                "counter_enabled": True,
                "label_bg_color": "#000000",
                "label_font_scale": 0.5,
                "label_text_color": "#FFFFFF",
            }
        }
    )
    frame = np.full((160, 360, 3), 255, dtype=np.uint8)

    annotated = VideoAnnotator(config).annotate(
        frame,
        tracks=[],
        labels_by_track={},
        counter_value=9,
        make_statistics_rows=[
            MakeStatisticRow(make="Toyota", origin_flag="🇯🇵", count=3, progress=1.0)
        ],
    )

    assert "Toyota" in drawn_text
    assert "3" in drawn_text
    assert "9" in drawn_text
    assert annotated.shape == frame.shape
    assert np.any(annotated[:80, :240] != frame[:80, :240])


def test_video_annotator_make_statistics_animation_smoke(
    config_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def render_test_flag(flag: str, target_height: int) -> np.ndarray:
        _ = flag
        image = np.zeros((target_height, target_height, 4), dtype=np.uint8)
        image[:, :, 1] = 255
        image[:, :, 3] = 255
        return image

    monkeypatch.setattr(annotators_module, "_render_flag_emoji", render_test_flag)

    config = config_factory(
        {"render": {"counter_enabled": True, "label_font_scale": 0.5}}
    )
    annotator = VideoAnnotator(config)
    frame = np.full((180, 420, 3), 255, dtype=np.uint8)

    first = annotator.annotate(
        frame,
        tracks=[],
        labels_by_track={},
        make_statistics_rows=[
            MakeStatisticRow(make="Toyota", origin_flag="🇯🇵", count=2, progress=1.0),
            MakeStatisticRow(make="VW", origin_flag="🇩🇪", count=1, progress=0.5),
        ],
    )
    second = annotator.annotate(
        frame,
        tracks=[],
        labels_by_track={},
        make_statistics_rows=[
            MakeStatisticRow(make="VW", origin_flag="🇩🇪", count=3, progress=1.0),
            MakeStatisticRow(make="Toyota", origin_flag="🇯🇵", count=2, progress=2 / 3),
        ],
    )

    assert first.shape == frame.shape
    assert second.shape == frame.shape
    assert np.any(first != second)


def test_split_flag_emojis_splits_pairs_and_skips_invalid_tail() -> None:
    assert annotators_module._split_flag_emojis("🇩🇪🇨🇭") == ("🇩🇪", "🇨🇭")
    assert annotators_module._split_flag_emojis("🇯🇵") == ("🇯🇵",)
    assert annotators_module._split_flag_emojis("🇩🇪x") == ("🇩🇪",)
    assert annotators_module._split_flag_emojis("xx") == ()
    assert annotators_module._split_flag_emojis("") == ()


def test_video_annotator_make_statistics_renders_multi_flag_origin(
    config_factory,
) -> None:
    config = config_factory(
        {"render": {"counter_enabled": True, "label_font_scale": 0.5}}
    )
    frame = np.full((180, 420, 3), 255, dtype=np.uint8)

    annotated = VideoAnnotator(config).annotate(
        frame,
        tracks=[],
        labels_by_track={},
        counter_value=1,
        make_statistics_rows=[
            MakeStatisticRow(make="Smart", origin_flag="🇩🇪🇨🇭", count=1, progress=1.0),
        ],
    )

    assert annotated.shape == frame.shape
    assert np.any(annotated != frame)


def test_video_annotator_counter_uses_balanced_visible_vertical_padding(
    config_factory,
) -> None:
    config = config_factory(
        {
            "render": {
                "label_bg_alpha": 1.0,
                "label_bg_color": "#000000",
                "label_padding_px": 6,
                "label_text_color": "#00FF00",
                "counter_enabled": True,
            }
        }
    )
    frame = np.full((100, 240, 3), 255, dtype=np.uint8)

    annotated = VideoAnnotator(config).annotate(
        frame,
        tracks=[],
        labels_by_track={},
        counter_value=682,
    )

    background_pixels = np.where(np.any(annotated < 255, axis=2))
    text_pixels = np.where(
        (annotated[:, :, 1] > 0)
        & (annotated[:, :, 1] > annotated[:, :, 0])
        & (annotated[:, :, 1] > annotated[:, :, 2])
    )
    top_padding = int(text_pixels[0].min()) - int(background_pixels[0].min())
    bottom_padding = int(background_pixels[0].max()) - int(text_pixels[0].max())

    assert abs(top_padding - bottom_padding) <= 1


def test_video_annotator_applies_track_color_to_all_label_lines_not_counter(
    default_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drawn_text_and_colors: list[tuple[str, tuple[int, int, int]]] = []

    def capture_put_text(
        img,
        text,
        org,
        font_face,
        font_scale,
        color,
        thickness,
        line_type,
    ):
        _ = org, font_face, font_scale, thickness, line_type
        drawn_text_and_colors.append((text, color))
        return img

    monkeypatch.setattr(annotators_module.cv2, "putText", capture_put_text)

    VideoAnnotator(default_config).annotate(
        np.zeros((240, 360, 3), dtype=np.uint8),
        tracks=[_track(1, bbox=BBox(x1=20, y1=20, x2=160, y2=80))],
        labels_by_track={1: "🇩🇪 VW Golf\nMk VIII\nElectric"},
        counter_value=1,
        label_text_colors_by_track={1: "#00BFFF"},
    )

    assert drawn_text_and_colors == [
        ("VW Golf", (255, 191, 0)),
        ("Mk VIII", (255, 191, 0)),
        ("Electric", (255, 191, 0)),
        ("1", (255, 255, 255)),
    ]


def test_video_annotator_clamps_label_below_track_at_bottom_edge(
    config_factory,
) -> None:
    config = config_factory(
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
    assert int(label_pixels[0].min()) >= 71


def test_draw_track_label_clips_past_right_edge_without_shifting_left(
    config_factory,
) -> None:
    config = config_factory(
        {
            "render": {
                "box_enabled": False,
                "counter_enabled": False,
                "label_bg_color": "#000000",
                "label_font_scale": 0.8,
                "label_scale_reference_box_width_px": 90,
            }
        }
    )
    frame = np.full((90, 120, 3), 255, dtype=np.uint8)

    annotated = VideoAnnotator(config).annotate(
        frame,
        tracks=[_track(1, bbox=BBox(x1=90, y1=10, x2=180, y2=50))],
        labels_by_track={1: "Renault Master Extra Long Label"},
    )

    assert np.array_equal(annotated[:, :90], frame[:, :90])
    assert np.any(annotated[55:80, 90:120] != frame[55:80, 90:120])


def test_video_annotator_always_passes_full_label_for_tiny_box(
    default_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels: list[str] = []

    def capture_label(frame, track, label, config, **kwargs):
        _ = frame, track, config, kwargs
        labels.append(label)

    monkeypatch.setattr(annotators_module, "_draw_track_label", capture_label)

    VideoAnnotator(default_config).annotate(
        np.zeros((120, 160, 3), dtype=np.uint8),
        tracks=[_track(1, bbox=BBox(x1=10, y1=10, x2=30, y2=50))],
        labels_by_track={1: "Toyota Corolla\nE210\nHybrid"},
    )

    assert labels == ["Toyota Corolla\nE210\nHybrid"]


def test_label_scale_factor_uses_box_width_proportion(config_factory) -> None:
    config = config_factory({"render": {"label_scale_reference_box_width_px": 90}})

    assert annotators_module._label_scale_factor(
        _track(1, bbox=BBox(x1=10, y1=10, x2=55, y2=50)),
        config,
    ) == pytest.approx(0.5)
    assert annotators_module._label_scale_factor(
        _track(1, bbox=BBox(x1=10, y1=10, x2=100, y2=50)),
        config,
    ) == pytest.approx(1.0)
    assert annotators_module._label_scale_factor(
        _track(1, bbox=BBox(x1=10, y1=10, x2=190, y2=50)),
        config,
    ) == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("box_width", "expected_scale"),
    [
        (45, 0.25),
        (180, 1.0),
    ],
)
def test_draw_track_label_scales_simplex_font_by_box_width(
    config_factory,
    monkeypatch: pytest.MonkeyPatch,
    box_width: int,
    expected_scale: float,
) -> None:
    scales: list[float] = []
    font_faces: list[int] = []
    thicknesses: list[int] = []

    def capture_put_text(
        img,
        text,
        org,
        font_face,
        font_scale,
        color,
        thickness,
        line_type,
    ):
        _ = text, org, color, line_type
        font_faces.append(font_face)
        scales.append(font_scale)
        thicknesses.append(thickness)
        return img

    monkeypatch.setattr(annotators_module.cv2, "putText", capture_put_text)
    config = config_factory(
        {
            "render": {
                "label_font_scale": 0.5,
                "label_thickness": 2,
                "label_scale_reference_box_width_px": 90,
            }
        }
    )

    annotators_module._draw_track_label(
        np.zeros((120, 160, 3), dtype=np.uint8),
        _track(1, bbox=BBox(x1=10, y1=10, x2=10 + box_width, y2=50)),
        "Toyota Corolla",
        config,
    )

    assert font_faces == [annotators_module.cv2.FONT_HERSHEY_DUPLEX]
    assert scales == [pytest.approx(expected_scale)]
    assert thicknesses == [2]


def test_flag_emoji_raster_scales_with_label_height() -> None:
    small = annotators_module._render_flag_emoji("🇩🇪", 12)
    large = annotators_module._render_flag_emoji("🇩🇪", 36)

    assert small.shape[0] == 12
    assert large.shape[0] == 36
    assert large.shape[1] == pytest.approx(small.shape[1] * 3, abs=2)
    assert np.count_nonzero(large[:, :, 3]) > 0
    assert len(np.unique(large[large[:, :, 3] > 0][:, :3], axis=0)) > 10


def test_draw_track_label_renders_color_flag_and_plain_opencv_text(
    default_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drawn_text: list[str] = []

    def capture_put_text(
        img,
        text,
        org,
        font_face,
        font_scale,
        color,
        thickness,
        line_type,
    ):
        _ = org, font_face, font_scale, color, thickness, line_type
        drawn_text.append(text)
        return img

    monkeypatch.setattr(annotators_module.cv2, "putText", capture_put_text)
    frame = np.zeros((120, 180, 3), dtype=np.uint8)

    annotators_module._draw_track_label(
        frame,
        _track(1, bbox=BBox(x1=10, y1=10, x2=100, y2=50)),
        "🇩🇪 VW Golf",
        default_config,
    )

    blue, green, red = (frame[:, :, index] for index in range(3))
    assert drawn_text == ["VW Golf"]
    assert np.any((red > 100) & (green < 100) & (blue < 100))
    assert np.any((red > 100) & (green > 100) & (blue < 100))


def test_draw_track_label_renders_multiple_leading_flags(
    default_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drawn_text: list[str] = []
    rendered_flags: list[str] = []

    def capture_put_text(
        img,
        text,
        org,
        font_face,
        font_scale,
        color,
        thickness,
        line_type,
    ):
        _ = org, font_face, font_scale, color, thickness, line_type
        drawn_text.append(text)
        return img

    def render_test_flag(flag: str, target_height: int) -> np.ndarray:
        rendered_flags.append(flag)
        return np.full((target_height, target_height, 4), 255, dtype=np.uint8)

    monkeypatch.setattr(annotators_module.cv2, "putText", capture_put_text)
    monkeypatch.setattr(annotators_module, "_render_flag_emoji", render_test_flag)

    annotators_module._draw_track_label(
        np.zeros((120, 180, 3), dtype=np.uint8),
        _track(1, bbox=BBox(x1=10, y1=10, x2=100, y2=50)),
        "🇩🇪🇨🇭 Smart",
        default_config,
    )

    assert drawn_text == ["Smart"]
    assert rendered_flags == ["🇩🇪", "🇨🇭", "🇩🇪", "🇨🇭"]


def test_draw_track_label_renders_trailing_tag_emojis_outside_opencv_text(
    default_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drawn_text: list[str] = []
    rendered_emojis: list[str] = []

    def capture_put_text(
        img,
        text,
        org,
        font_face,
        font_scale,
        color,
        thickness,
        line_type,
    ):
        _ = org, font_face, font_scale, color, thickness, line_type
        drawn_text.append(text)
        return img

    def render_test_emoji(emoji: str, target_height: int) -> np.ndarray:
        rendered_emojis.append(emoji)
        return np.full((target_height, target_height, 4), 255, dtype=np.uint8)

    monkeypatch.setattr(annotators_module.cv2, "putText", capture_put_text)
    monkeypatch.setattr(annotators_module, "_render_color_emoji", render_test_emoji)

    annotators_module._draw_track_label(
        np.zeros((120, 180, 3), dtype=np.uint8),
        _track(1, bbox=BBox(x1=10, y1=10, x2=100, y2=50)),
        "VW Golf 🚓 🚑",
        default_config,
    )

    assert drawn_text == ["VW Golf"]
    assert rendered_emojis == ["🚓", "🚑", "🚓", "🚑"]


def test_draw_track_label_uses_configured_scaled_flag_gap(
    config_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text_origins: list[tuple[int, int]] = []

    def capture_put_text(
        img,
        text,
        org,
        font_face,
        font_scale,
        color,
        thickness,
        line_type,
    ):
        _ = text, font_face, font_scale, color, thickness, line_type
        text_origins.append(org)
        return img

    monkeypatch.setattr(annotators_module.cv2, "putText", capture_put_text)
    frame = np.zeros((160, 300, 3), dtype=np.uint8)
    track = _track(1, bbox=BBox(x1=10, y1=10, x2=190, y2=50))

    for flag_gap in [0, 6]:
        annotators_module._draw_track_label(
            frame.copy(),
            track,
            "🇩🇪 VW Golf",
            config_factory(
                {
                    "render": {
                        "label_flag_gap_px": flag_gap,
                        "label_scale_reference_box_width_px": 90,
                    }
                }
            ),
        )

    assert text_origins[1][0] - text_origins[0][0] == 12


def test_video_annotator_draws_all_boxes_before_area_sorted_labels(
    default_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draw_order: list[tuple[str, int]] = []

    def capture_box(frame, track, config):
        _ = frame, config
        draw_order.append(("box", track.track_id))

    def capture_label(frame, track, label, config, **kwargs):
        _ = frame, label, config, kwargs
        draw_order.append(("label", track.track_id))

    monkeypatch.setattr(annotators_module, "_draw_track_box", capture_box)
    monkeypatch.setattr(annotators_module, "_draw_track_label", capture_label)

    VideoAnnotator(default_config).annotate(
        np.zeros((160, 180, 3), dtype=np.uint8),
        tracks=[
            _track(1, bbox=BBox(x1=10, y1=10, x2=110, y2=70)),
            _track(2, bbox=BBox(x1=20, y1=80, x2=60, y2=95)),
        ],
        labels_by_track={1: "Large box", 2: "Small box"},
    )

    assert draw_order == [("box", 1), ("box", 2), ("label", 2), ("label", 1)]


def test_video_annotator_preserves_label_order_for_equal_box_areas(
    default_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    label_order: list[int] = []

    def capture_label(frame, track, label, config, **kwargs):
        _ = frame, label, config, kwargs
        label_order.append(track.track_id)

    monkeypatch.setattr(annotators_module, "_draw_track_label", capture_label)

    VideoAnnotator(default_config).annotate(
        np.zeros((160, 180, 3), dtype=np.uint8),
        tracks=[
            _track(1, bbox=BBox(x1=10, y1=10, x2=70, y2=50)),
            _track(2, bbox=BBox(x1=20, y1=80, x2=60, y2=140)),
        ],
        labels_by_track={1: "First equal area", 2: "Second equal area"},
    )

    assert label_order == [1, 2]


def test_video_annotator_does_not_retain_trace_history_between_calls(
    default_config,
) -> None:
    annotator = VideoAnnotator(default_config)
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


def test_format_label_text_prefixes_known_make_with_flag_and_one_space() -> None:
    assert (
        format_label_text(
            MMRResult(make="VW", model="Golf", generation="Mk VIII"),
            "unknown",
            {"VW": "🇩🇪"},
        )
        == "🇩🇪 VW Golf\nMk VIII"
    )
    assert (
        format_label_text(
            MMRResult(make="Unknown Make", model="Model"),
            "unknown",
            {"VW": "🇩🇪"},
        )
        == "Unknown Make Model"
    )


def test_format_label_text_appends_affirmative_tag_emojis_to_first_line() -> None:
    assert (
        format_label_text(
            MMRResult(
                make="VW",
                model="Golf",
                generation="Mk VIII",
                tags=[
                    {"name": "pickup", "value": True},
                    {"name": "taxi", "value": "yes"},
                    {"name": "damaged", "value": True},
                    {"name": "ambulance", "value": 1},
                    {"name": "law enforcement", "value": "true"},
                    {"name": "fleet", "value": False},
                ],
            ),
            "unknown",
            {"VW": "🇩🇪"},
        )
        == "🇩🇪 VW Golf 🚓 🚑 🛻\nMk VIII"
    )


def test_format_label_text_hides_all_old_vehicle_details() -> None:
    assert (
        format_label_text(
            MMRResult(
                make="Mercedes",
                model="OLD",
                generation=" old ",
                variation="OLD",
            ),
            "unknown",
        )
        == "Mercedes"
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


def test_label_text_colors_only_override_accepted_bev_and_mixed_results(
    default_config,
    tmp_path,
) -> None:
    store = DummyRunStore(tmp_path)
    results = {
        1: MMRResult(
            make="Test",
            model="BEV",
            generation="Mk I",
            accepted=True,
        ),
        2: MMRResult(
            make="Test",
            model="Mixed",
            generation="Mk I",
            accepted=True,
        ),
        3: MMRResult(
            make="Test",
            model="Combustion",
            generation="Mk I",
            accepted=True,
        ),
        4: MMRResult(
            make="Test",
            model="Unknown",
            generation="Mk I",
            accepted=True,
        ),
        5: MMRResult(
            make="Test",
            model="Unmatched",
            generation="Mk I",
            accepted=True,
        ),
        6: MMRResult(make="Test", model="Incomplete", accepted=True),
        7: MMRResult(
            make="Test",
            model="Rejected BEV",
            generation="Mk I",
            accepted=False,
        ),
    }
    store.labels_path.parent.mkdir(parents=True, exist_ok=True)
    store.labels_path.write_bytes(
        orjson.dumps(
            {
                str(track_id): result.model_dump(mode="json")
                for track_id, result in results.items()
            }
        )
    )
    powertrain_catalog = {
        VehicleIdentity("Test", "BEV", "Mk I"): PowertrainClass.BEV,
        VehicleIdentity("Test", "Mixed", "Mk I"): PowertrainClass.MIXED,
        VehicleIdentity("Test", "Combustion", "Mk I"): PowertrainClass.COMBUSTION,
        VehicleIdentity("Test", "Unknown", "Mk I"): PowertrainClass.UNKNOWN,
        VehicleIdentity("Test", "Rejected BEV", "Mk I"): PowertrainClass.BEV,
    }

    label_text, label_text_colors, accepted_labels = _label_text_and_colors_by_track(
        default_config,
        _as_run_store(store),
        store.frames_path,
        allow_unclassified_annotations=False,
        origin_country_by_make={},
        powertrain_catalog=powertrain_catalog,
    )

    assert set(label_text) == {1, 2, 3, 4, 5, 6}
    assert label_text_colors == {1: "#47C8FF", 2: "#39FF31"}
    assert set(accepted_labels) == {1, 2, 3, 4, 5, 6}


def test_unclassified_label_text_has_no_powertrain_color_override(
    default_config, tmp_path
) -> None:
    store = DummyRunStore(tmp_path)
    _write_records(
        store.frames_path,
        [
            FrameRecord(
                frame_index=0,
                timestamp_seconds=0.0,
                tracks=[_track(1, vehicle_index=1)],
            )
        ],
    )

    label_text, label_text_colors, accepted_labels = _label_text_and_colors_by_track(
        default_config,
        _as_run_store(store),
        store.frames_path,
        allow_unclassified_annotations=True,
        origin_country_by_make={},
        powertrain_catalog={},
    )

    assert label_text == {1: "UNKNOWN"}
    assert label_text_colors == {}
    assert accepted_labels == {}


def test_show_unclassified_tracks_extends_label_text_beyond_accepted_labels(
    config_factory, tmp_path
) -> None:
    store = DummyRunStore(tmp_path)
    _write_records(
        store.frames_path,
        [
            FrameRecord(
                frame_index=0,
                timestamp_seconds=0.0,
                tracks=[_track(1, vehicle_index=1), _track(2, vehicle_index=2)],
            )
        ],
    )
    store.labels_path.parent.mkdir(parents=True, exist_ok=True)
    store.labels_path.write_bytes(
        orjson.dumps(
            {
                "1": MMRResult(
                    make="Toyota",
                    accepted=True,
                    vehicle_index=1,
                ).model_dump(mode="json")
            }
        )
    )
    config = config_factory({"render": {"show_unclassified_tracks": True}})

    label_text, label_text_colors, accepted_labels = _label_text_and_colors_by_track(
        config,
        _as_run_store(store),
        store.frames_path,
        allow_unclassified_annotations=False,
        origin_country_by_make={},
        powertrain_catalog={},
    )

    assert set(label_text) == {1, 2}
    assert label_text[1] == "Toyota"
    assert label_text[2] == "UNKNOWN"
    assert label_text_colors == {}
    assert set(accepted_labels) == {1}


def test_flags_off_keep_accepted_only_label_text_with_existing_labels(
    default_config, tmp_path
) -> None:
    store = DummyRunStore(tmp_path)
    _write_records(
        store.frames_path,
        [
            FrameRecord(
                frame_index=0,
                timestamp_seconds=0.0,
                tracks=[_track(1, vehicle_index=1), _track(2, vehicle_index=2)],
            )
        ],
    )
    store.labels_path.parent.mkdir(parents=True, exist_ok=True)
    store.labels_path.write_bytes(
        orjson.dumps(
            {
                "1": MMRResult(
                    make="Toyota",
                    accepted=True,
                    vehicle_index=1,
                ).model_dump(mode="json")
            }
        )
    )

    label_text, label_text_colors, accepted_labels = _label_text_and_colors_by_track(
        default_config,
        _as_run_store(store),
        store.frames_path,
        allow_unclassified_annotations=False,
        origin_country_by_make={},
        powertrain_catalog={},
    )

    assert set(label_text) == {1}
    assert set(accepted_labels) == {1}


def test_output_fps_caps_configured_render_fps_at_source_fps(config_factory) -> None:
    config = config_factory({"video": {"fps": 30.0}, "render": {"output_fps": 60.0}})

    assert _output_fps(config) == 30.0


def test_visible_track_ids_applies_crop_eligibility_when_required(
    config_factory,
) -> None:
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
        _render_config(config_factory, require_crop_eligible_track=True),
        records,
    ) == {1}


def test_visible_track_ids_skips_crop_eligibility_when_unclassified_tracks_show(
    config_factory,
) -> None:
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
        _render_config(
            config_factory,
            require_crop_eligible_track=True,
            show_unclassified_tracks=True,
        ),
        records,
    ) == {1, 2}


def test_visible_track_ids_applies_min_box_width_when_summaries_exist(
    config_factory,
) -> None:
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
        _summary(1, max_box_width_px=160),
        _summary(2, max_box_width_px=159),
    ]

    assert visible_track_ids_for_render(
        config_factory(
            {
                "analysis": {"min_box_width_px": 160},
                "render": {"min_visible_track_observations": 1},
            }
        ),
        records,
        track_summaries=summaries,
    ) == {1}


def test_visible_track_ids_applies_min_box_height_when_summaries_exist(
    config_factory,
) -> None:
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
    # Both tracks pass the width gate; track 2 never exceeds the height gate.
    summaries = [
        _summary(1, max_box_width_px=200, max_box_height_px=60),
        _summary(2, max_box_width_px=200, max_box_height_px=40),
    ]

    assert visible_track_ids_for_render(
        config_factory(
            {
                "analysis": {"min_box_width_px": 160, "min_box_height_px": 56},
                "render": {"min_visible_track_observations": 1},
            }
        ),
        records,
        track_summaries=summaries,
    ) == {1}


def test_visible_track_ids_treats_missing_height_stats_as_eligible(
    config_factory,
) -> None:
    records = [
        FrameRecord(
            frame_index=0,
            timestamp_seconds=0.0,
            tracks=[_track(1, vehicle_index=1)],
        )
    ]
    # Manifests written before the two-dimensional size gate carry no height
    # stats; they must stay renderable under the width-only behavior.
    summaries = [_summary(1, max_box_width_px=200)]

    assert visible_track_ids_for_render(
        config_factory(
            {
                "analysis": {"min_box_width_px": 160, "min_box_height_px": 56},
                "render": {"min_visible_track_observations": 1},
            }
        ),
        records,
        track_summaries=summaries,
    ) == {1}


def test_visible_track_ids_keeps_old_behavior_when_summaries_missing(
    config_factory,
) -> None:
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

    assert visible_track_ids_for_render(_render_config(config_factory), records) == {
        1,
        2,
    }


def test_resolve_render_frames_path_returns_raw_path_when_smoothing_disabled(
    config_factory,
    tmp_path,
) -> None:
    store = DummyRunStore(tmp_path)

    assert (
        _resolve_render_frames_path(
            _render_config(config_factory, smoothing_enabled=False),
            build_full_frame_profile(width=32, height=32),
            _as_run_store(store),
            smooth_render_tracks=None,
        )
        == store.frames_path
    )


def test_resolve_render_frames_path_calls_smoother_when_smoothing_enabled(
    config_factory,
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
            _render_config(config_factory, smoothing_enabled=True),
            build_full_frame_profile(width=32, height=32),
            _as_run_store(store),
            smooth_render_tracks=fake_smoother,
        )
        == store.render_frames_path
    )
    assert calls == 1


def test_resolve_render_frames_path_requires_smoother_when_smoothing_enabled(
    config_factory,
    tmp_path,
) -> None:
    store = DummyRunStore(tmp_path)

    with pytest.raises(ValueError, match="no smoothing stage was provided"):
        _resolve_render_frames_path(
            _render_config(config_factory, smoothing_enabled=True),
            build_full_frame_profile(width=32, height=32),
            _as_run_store(store),
            smooth_render_tracks=None,
        )


def test_render_filters_by_visibility_count(
    config_factory, tmp_path, monkeypatch
) -> None:
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
                "1": MMRResult(accepted=True, vehicle_index=1).model_dump(mode="json"),
                "2": MMRResult(accepted=True, vehicle_index=2).model_dump(mode="json"),
            }
        )
    )
    writer = DummyWriter()
    DummyAnnotator.seen_track_ids = []
    DummyAnnotator.seen_labels_by_track = []
    DummyAnnotator.seen_counter_values = []
    _patch_render_io(monkeypatch, writer)

    render_video(
        config=_render_config(config_factory, min_visible_track_observations=2),
        profile=build_full_frame_profile(width=32, height=32),
        video_path=store.manifest.video_path,
        run_store=_as_run_store(store),
    )

    assert [1] in DummyAnnotator.seen_track_ids
    assert [2] not in DummyAnnotator.seen_track_ids
    assert writer.released is True


def test_render_respects_require_crop_eligible_track(
    config_factory, tmp_path, monkeypatch
) -> None:
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
                "1": MMRResult(accepted=True, vehicle_index=1).model_dump(mode="json"),
                "2": MMRResult(accepted=True, vehicle_index=2).model_dump(mode="json"),
            }
        )
    )
    writer = DummyWriter()
    DummyAnnotator.seen_track_ids = []
    _patch_render_io(monkeypatch, writer)

    render_video(
        config=_render_config(config_factory, require_crop_eligible_track=True),
        profile=build_full_frame_profile(width=32, height=32),
        video_path=store.manifest.video_path,
        run_store=_as_run_store(store),
    )

    assert [1] in DummyAnnotator.seen_track_ids
    assert [2] not in DummyAnnotator.seen_track_ids


def test_render_hides_labeled_track_below_min_box_width(
    config_factory, tmp_path, monkeypatch
) -> None:
    store = DummyRunStore(tmp_path)
    store.track_summaries = [
        _summary(1, max_box_width_px=120, vehicle_index=1),
        _summary(2, max_box_width_px=180, vehicle_index=2),
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
                "1": MMRResult(accepted=True, vehicle_index=1).model_dump(mode="json"),
                "2": MMRResult(accepted=True, vehicle_index=2).model_dump(mode="json"),
            }
        )
    )
    writer = DummyWriter()
    DummyAnnotator.seen_track_ids = []
    _patch_render_io(monkeypatch, writer)

    render_video(
        config=config_factory(
            {
                "analysis": {"min_box_width_px": 160},
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


def test_render_min_box_width_applies_when_showing_unclassified_tracks(
    config_factory, tmp_path, monkeypatch
) -> None:
    store = DummyRunStore(tmp_path)
    store.track_summaries = [
        _summary(1, max_box_width_px=120, vehicle_index=1),
        _summary(2, max_box_width_px=180, vehicle_index=2),
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
        config=config_factory(
            {
                "analysis": {"min_box_width_px": 160},
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
    config_factory, tmp_path, monkeypatch
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
    DummyAnnotator.seen_label_text_colors_by_track = []
    _patch_render_io(monkeypatch, writer)

    render_video(
        config=_render_config(config_factory),
        profile=build_full_frame_profile(width=32, height=32),
        video_path=store.manifest.video_path,
        run_store=_as_run_store(store),
        allow_unclassified_annotations=True,
    )

    assert [1] in DummyAnnotator.seen_track_ids
    assert [2] not in DummyAnnotator.seen_track_ids
    assert DummyAnnotator.seen_labels_by_track[-1] == {1: "UNKNOWN"}
    assert DummyAnnotator.seen_label_text_colors_by_track[-1] == {}


def test_render_skips_rejected_classification_labels(
    config_factory, tmp_path, monkeypatch
) -> None:
    store = DummyRunStore(tmp_path)
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
                "1": MMRResult(
                    make="Toyota",
                    accepted=True,
                    vehicle_index=1,
                ).model_dump(mode="json"),
                "2": MMRResult(
                    make="Honda",
                    accepted=False,
                    vehicle_index=2,
                ).model_dump(mode="json"),
            }
        )
    )
    writer = DummyWriter()
    DummyAnnotator.seen_track_ids = []
    DummyAnnotator.seen_labels_by_track = []
    _patch_render_io(monkeypatch, writer)

    render_video(
        config=_render_config(config_factory),
        profile=build_full_frame_profile(width=32, height=32),
        video_path=store.manifest.video_path,
        run_store=_as_run_store(store),
    )

    assert [1] in DummyAnnotator.seen_track_ids
    assert [2] not in DummyAnnotator.seen_track_ids
    assert DummyAnnotator.seen_labels_by_track[-1] == {1: "🇯🇵 Toyota"}


def test_render_passes_exact_powertrain_text_colors_to_annotator(
    config_factory, tmp_path, monkeypatch
) -> None:
    store = DummyRunStore(tmp_path)
    _write_records(
        store.frames_path,
        [
            FrameRecord(
                frame_index=0,
                timestamp_seconds=0.0,
                tracks=[
                    _track(1, vehicle_index=1),
                    _track(2, vehicle_index=2),
                    _track(3, vehicle_index=3),
                ],
            )
        ],
    )
    store.labels_path.parent.mkdir(parents=True, exist_ok=True)
    store.labels_path.write_bytes(
        orjson.dumps(
            {
                "1": MMRResult(
                    make="Tesla",
                    model="Model 3",
                    generation="Mk I (2018)",
                    accepted=True,
                    vehicle_index=1,
                ).model_dump(mode="json"),
                "2": MMRResult(
                    make="Hyundai",
                    model="Kona",
                    generation="Mk I (2017) ~ Mk I EV (2019)",
                    accepted=True,
                    vehicle_index=2,
                ).model_dump(mode="json"),
                "3": MMRResult(
                    make="Audi",
                    model="A4",
                    generation="B5 (1998)",
                    accepted=True,
                    vehicle_index=3,
                ).model_dump(mode="json"),
            }
        )
    )
    writer = DummyWriter()
    DummyAnnotator.seen_label_text_colors_by_track = []
    _patch_render_io(monkeypatch, writer)

    render_video(
        config=_render_config(config_factory),
        profile=build_full_frame_profile(width=32, height=32),
        video_path=store.manifest.video_path,
        run_store=_as_run_store(store),
    )

    assert DummyAnnotator.seen_label_text_colors_by_track[-1] == {
        1: "#47C8FF",
        2: "#39FF31",
    }


def test_render_passes_live_count_to_annotator(
    config_factory, tmp_path, monkeypatch
) -> None:
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
                "1": MMRResult(
                    make="Toyota", accepted=True, vehicle_index=1
                ).model_dump(mode="json"),
                "2": MMRResult(make="VW", accepted=True, vehicle_index=2).model_dump(
                    mode="json"
                ),
            }
        )
    )
    writer = DummyWriter()
    DummyAnnotator.seen_track_ids = []
    DummyAnnotator.seen_labels_by_track = []
    DummyAnnotator.seen_counter_values = []
    _patch_render_io(monkeypatch, writer)

    render_video(
        config=_render_config(config_factory),
        profile=build_full_frame_profile(width=32, height=32),
        video_path=store.manifest.video_path,
        run_store=_as_run_store(store),
    )

    assert DummyAnnotator.seen_counter_values == [1, 2, 2]


def test_render_passes_live_make_statistics_to_annotator(
    config_factory, tmp_path, monkeypatch
) -> None:
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
            FrameRecord(
                frame_index=2,
                timestamp_seconds=2 / 30.0,
                tracks=[
                    _track(1, 2, vehicle_index=1, counted=True),
                    _track(2, 2, vehicle_index=2, counted=True),
                    _track(3, 2, vehicle_index=3, counted=True),
                ],
            ),
        ],
    )
    store.labels_path.parent.mkdir(parents=True, exist_ok=True)
    store.labels_path.write_bytes(
        orjson.dumps(
            {
                "1": MMRResult(
                    make="Toyota",
                    model="Corolla",
                    accepted=True,
                    vehicle_index=1,
                ).model_dump(mode="json"),
                "2": MMRResult(
                    make="VW",
                    model="Golf",
                    accepted=True,
                    vehicle_index=2,
                ).model_dump(mode="json"),
                "3": MMRResult(
                    make="Toyota",
                    model="Yaris",
                    accepted=True,
                    vehicle_index=3,
                ).model_dump(mode="json"),
            }
        )
    )
    writer = DummyWriter()
    _reset_dummy_annotator()
    _patch_render_io(monkeypatch, writer)

    render_video(
        config=_render_config(config_factory),
        profile=build_full_frame_profile(width=32, height=32),
        video_path=store.manifest.video_path,
        run_store=_as_run_store(store),
    )

    assert DummyAnnotator.seen_make_statistics_rows == [
        [MakeStatisticRow(make="Toyota", origin_flag="🇯🇵", count=1, progress=1.0)],
        [
            MakeStatisticRow(make="Toyota", origin_flag="🇯🇵", count=1, progress=1.0),
            MakeStatisticRow(make="VW", origin_flag="🇩🇪", count=1, progress=1.0),
        ],
        [
            MakeStatisticRow(make="Toyota", origin_flag="🇯🇵", count=2, progress=1.0),
            MakeStatisticRow(make="VW", origin_flag="🇩🇪", count=1, progress=0.5),
        ],
    ]


def test_make_statistics_rows_are_sorted_and_limited() -> None:
    rows = _make_statistics_rows(
        {
            1: "Brand 0",
            2: "Brand 1",
            3: "Brand 2",
            4: "Brand 3",
            5: "Brand 4",
            6: "Brand 5",
            7: "Brand 6",
            8: "Brand 7",
            9: "Brand 8",
            10: "Brand 9",
            11: "Brand 0",
        },
        {"Brand 0": "🇯🇵"},
    )

    assert rows == [
        MakeStatisticRow(make="Brand 0", origin_flag="🇯🇵", count=2, progress=1.0),
        MakeStatisticRow(make="Brand 1", origin_flag=None, count=1, progress=0.5),
        MakeStatisticRow(make="Brand 2", origin_flag=None, count=1, progress=0.5),
        MakeStatisticRow(make="Brand 3", origin_flag=None, count=1, progress=0.5),
        MakeStatisticRow(make="Brand 4", origin_flag=None, count=1, progress=0.5),
        MakeStatisticRow(make="Brand 5", origin_flag=None, count=1, progress=0.5),
        MakeStatisticRow(make="Brand 6", origin_flag=None, count=1, progress=0.5),
        MakeStatisticRow(make="Brand 7", origin_flag=None, count=1, progress=0.5),
        MakeStatisticRow(make="Brand 8", origin_flag=None, count=1, progress=0.5),
        MakeStatisticRow(make="Brand 9", origin_flag=None, count=1, progress=0.5),
    ]


def test_make_statistics_rows_order_ties_by_make() -> None:
    rows = _make_statistics_rows(
        {
            1: "VW",
            2: "Audi",
        },
        {},
    )

    assert [row.make for row in rows] == ["Audi", "VW"]


def test_render_counter_counts_only_rendered_accepted_tracks(
    config_factory, tmp_path, monkeypatch
) -> None:
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
                    _track(3, 1, vehicle_index=3, counted=True),
                ],
            ),
        ],
    )
    store.labels_path.parent.mkdir(parents=True, exist_ok=True)
    store.labels_path.write_bytes(
        orjson.dumps(
            {
                "1": MMRResult(
                    make="Toyota",
                    accepted=True,
                    vehicle_index=1,
                ).model_dump(mode="json"),
                "2": MMRResult(
                    make="Honda",
                    accepted=False,
                    vehicle_index=2,
                ).model_dump(mode="json"),
                "3": MMRResult(
                    make="VW",
                    accepted=True,
                    vehicle_index=3,
                ).model_dump(mode="json"),
            }
        )
    )
    writer = DummyWriter()
    DummyAnnotator.seen_track_ids = []
    DummyAnnotator.seen_counter_values = []
    _patch_render_io(monkeypatch, writer)

    render_video(
        config=_render_config(config_factory),
        profile=build_full_frame_profile(width=32, height=32),
        video_path=store.manifest.video_path,
        run_store=_as_run_store(store),
    )

    assert [1, 3] in DummyAnnotator.seen_track_ids
    assert [1, 2, 3] not in DummyAnnotator.seen_track_ids
    assert DummyAnnotator.seen_counter_values == [1, 2, 2]


def test_render_make_statistics_use_counter_render_eligibility(
    config_factory, tmp_path, monkeypatch
) -> None:
    store = DummyRunStore(tmp_path)
    _write_records(
        store.frames_path,
        [
            FrameRecord(
                frame_index=0,
                timestamp_seconds=0.0,
                tracks=[
                    _track(1, 0, vehicle_index=1, counted=True),
                    _track(3, 0, vehicle_index=3, counted=True),
                    _track(4, 0, vehicle_index=4, counted=True),
                ],
            ),
            FrameRecord(
                frame_index=1,
                timestamp_seconds=1 / 30.0,
                tracks=[
                    _track(1, 1, vehicle_index=1, counted=True),
                    _track(2, 1, vehicle_index=2, counted=True),
                    _track(3, 1, vehicle_index=3, counted=True),
                ],
            ),
        ],
    )
    store.labels_path.parent.mkdir(parents=True, exist_ok=True)
    store.labels_path.write_bytes(
        orjson.dumps(
            {
                "1": MMRResult(
                    make="Toyota",
                    accepted=True,
                    vehicle_index=1,
                ).model_dump(mode="json"),
                "2": MMRResult(
                    make="Honda",
                    accepted=False,
                    vehicle_index=2,
                ).model_dump(mode="json"),
                "3": MMRResult(
                    accepted=True,
                    vehicle_index=3,
                ).model_dump(mode="json"),
                "4": MMRResult(
                    make="VW",
                    accepted=True,
                    vehicle_index=4,
                ).model_dump(mode="json"),
            }
        )
    )
    writer = DummyWriter()
    _reset_dummy_annotator()
    _patch_render_io(monkeypatch, writer)

    render_video(
        config=_render_config(config_factory, min_visible_track_observations=2),
        profile=build_full_frame_profile(width=32, height=32),
        video_path=store.manifest.video_path,
        run_store=_as_run_store(store),
    )

    assert DummyAnnotator.seen_counter_values == [2, 2, 2]
    assert DummyAnnotator.seen_make_statistics_rows == [
        [MakeStatisticRow(make="Toyota", origin_flag="🇯🇵", count=1, progress=1.0)],
        [MakeStatisticRow(make="Toyota", origin_flag="🇯🇵", count=1, progress=1.0)],
        [MakeStatisticRow(make="Toyota", origin_flag="🇯🇵", count=1, progress=1.0)],
    ]


def test_render_uses_injected_smoother_when_smoothing_is_enabled(
    config_factory, tmp_path, monkeypatch
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
        config=_render_config(config_factory, smoothing_enabled=True),
        profile=build_full_frame_profile(width=32, height=32),
        video_path=store.manifest.video_path,
        run_store=_as_run_store(store),
        allow_unclassified_annotations=True,
        smooth_render_tracks=fake_smoother,
    )

    assert calls == 1


def test_render_requires_smoother_when_smoothing_is_enabled(
    config_factory, tmp_path, monkeypatch
) -> None:
    store = DummyRunStore(tmp_path)
    writer = DummyWriter()
    _patch_render_io(monkeypatch, writer)

    with pytest.raises(ValueError, match="no smoothing stage was provided"):
        render_video(
            config=_render_config(config_factory, smoothing_enabled=True),
            profile=build_full_frame_profile(width=32, height=32),
            video_path=store.manifest.video_path,
            run_store=_as_run_store(store),
        )


def test_render_counter_counts_canonical_vehicle_index_once_across_split_track_ids(
    config_factory, tmp_path, monkeypatch
) -> None:
    store = DummyRunStore(tmp_path)
    _write_records(
        store.frames_path,
        [
            FrameRecord(
                frame_index=0,
                timestamp_seconds=0.0,
                tracks=[
                    _track(1, 0, vehicle_index=1, counted=True),
                    _track(2, 0, vehicle_index=1, counted=True),
                ],
            ),
            FrameRecord(
                frame_index=1,
                timestamp_seconds=1 / 30.0,
                tracks=[
                    _track(1, 1, vehicle_index=1, counted=True),
                    _track(2, 1, vehicle_index=1, counted=True),
                ],
            ),
        ],
    )
    store.labels_path.parent.mkdir(parents=True, exist_ok=True)
    store.labels_path.write_bytes(
        orjson.dumps(
            {
                "1": MMRResult(
                    make="Toyota",
                    accepted=True,
                    vehicle_index=1,
                ).model_dump(mode="json"),
                "2": MMRResult(
                    make="Toyota",
                    accepted=True,
                    vehicle_index=1,
                ).model_dump(mode="json"),
            }
        )
    )
    writer = DummyWriter()
    DummyAnnotator.seen_track_ids = []
    DummyAnnotator.seen_counter_values = []
    _patch_render_io(monkeypatch, writer)

    render_video(
        config=_render_config(config_factory),
        profile=build_full_frame_profile(width=32, height=32),
        video_path=store.manifest.video_path,
        run_store=_as_run_store(store),
    )

    assert DummyAnnotator.seen_counter_values == [1, 1, 1]
