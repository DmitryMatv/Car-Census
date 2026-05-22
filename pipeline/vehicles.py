from __future__ import annotations

import shutil
from pathlib import Path
from typing import Protocol, Sequence

from models import CountEvent, CropCandidate, TrackSummary
from storage.run_store import RunStore


class TrackStateProtocol(Protocol):
    track_id: int
    first_frame_index: int
    last_frame_index: int
    vehicle_index: int | None
    frames_seen: int
    max_box_height_px: float
    counted: bool
    count_event: CountEvent | None
    candidates: list[CropCandidate]


def staged_track_crop_dir(crops_dir: Path, track_id: int) -> Path:
    return crops_dir / ".track_candidates" / f"track_{track_id:06d}"


def vehicle_crop_path(crops_dir: Path, vehicle_index: int, source_path: Path) -> Path:
    suffix = source_path.suffix or ".jpg"
    return crops_dir / f"vehicle_{vehicle_index:06d}{suffix}"


def discard_track_artifacts(state: TrackStateProtocol | None, crops_dir: Path) -> None:
    if state is None:
        return
    for candidate in state.candidates:
        if candidate.image_path.exists():
            candidate.image_path.unlink()
    for path in [
        staged_track_crop_dir(crops_dir, state.track_id),
    ]:
        if path is not None and path.exists() and not any(path.iterdir()):
            path.rmdir()


def track_summary_from_state(state: TrackStateProtocol) -> TrackSummary:
    return TrackSummary(
        track_id=state.track_id,
        vehicle_index=state.vehicle_index,
        first_frame_index=state.first_frame_index,
        last_frame_index=state.last_frame_index,
        frames_seen=state.frames_seen,
        max_box_height_px=state.max_box_height_px,
        counted=state.counted,
        count_event=state.count_event,
        candidates=state.candidates,
    )


def finalize_vehicle_identities(
    run_store: RunStore, track_states: Sequence[TrackStateProtocol]
) -> dict[int, int]:
    eligible_states = [state for state in track_states if state.candidates]
    eligible_states.sort(key=lambda item: (item.first_frame_index, item.track_id))
    vehicle_index_by_track: dict[int, int] = {}

    for vehicle_index, state in enumerate(eligible_states, start=1):
        state.vehicle_index = vehicle_index
        vehicle_index_by_track[state.track_id] = vehicle_index
        finalized_candidates: list[CropCandidate] = []
        for candidate in state.candidates:
            destination = vehicle_crop_path(
                run_store.crops_dir, vehicle_index, candidate.image_path
            )
            if candidate.image_path.exists():
                candidate.image_path.replace(destination)
            finalized_candidates.append(
                candidate.model_copy(
                    update={
                        "vehicle_index": vehicle_index,
                        "image_path": destination,
                    }
                )
            )
        state.candidates = finalized_candidates

    shutil.rmtree(run_store.crops_dir / ".track_candidates", ignore_errors=True)
    return vehicle_index_by_track
