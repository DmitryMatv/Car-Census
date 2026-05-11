from pathlib import Path

import cv2
import numpy as np

from config import AppConfig
from pipeline.analyze import (
    MutableTrackState,
    _render_bbox_for_track,
    _save_candidate,
)
from pipeline.vehicles import staged_track_crop_dir
from models import BBox


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
