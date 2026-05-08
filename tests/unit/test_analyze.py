from pathlib import Path

import numpy as np

from car_census.config import AppConfig
from car_census.pipeline.analyze import (
    MutableTrackState,
    _save_candidate,
)
from car_census.pipeline.vehicles import staged_track_crop_dir
from car_census.types import BBox


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
