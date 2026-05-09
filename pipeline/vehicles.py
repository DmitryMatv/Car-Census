from __future__ import annotations

import shutil
from pathlib import Path
from typing import Protocol, Sequence

import orjson

from storage.run_store import RunStore
from models import CountEvent, CropCandidate, FrameRecord, TrackSummary


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


def vehicle_crop_dir(crops_dir: Path, vehicle_index: int) -> Path:
    return crops_dir / f"vehicle_{vehicle_index:06d}"


def discard_track_artifacts(state: TrackStateProtocol | None, crops_dir: Path) -> None:
    if state is None:
        return
    for candidate in state.candidates:
        if candidate.image_path.exists():
            candidate.image_path.unlink()
    for path in [
        staged_track_crop_dir(crops_dir, state.track_id),
        (
            vehicle_crop_dir(crops_dir, state.vehicle_index)
            if state.vehicle_index is not None
            else None
        ),
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


def rewrite_frame_vehicle_indices(
    frames_path: Path, vehicle_index_by_track: dict[int, int]
) -> None:
    temp_path = frames_path.with_suffix(f"{frames_path.suffix}.tmp")
    with frames_path.open("rb") as source, temp_path.open("wb") as target:
        for line in source:
            if not line.strip():
                continue
            record = FrameRecord.model_validate(orjson.loads(line))
            tracks = [
                track.model_copy(
                    update={"vehicle_index": vehicle_index_by_track.get(track.track_id)}
                )
                for track in record.tracks
            ]
            target.write(
                orjson.dumps(
                    record.model_copy(update={"tracks": tracks}).model_dump(mode="json")
                )
            )
            target.write(b"\n")
    temp_path.replace(frames_path)


def finalize_vehicle_identities(
    run_store: RunStore, track_states: Sequence[TrackStateProtocol]
) -> dict[int, int]:
    eligible_states = [state for state in track_states if state.candidates]
    eligible_states.sort(key=lambda item: (item.first_frame_index, item.track_id))
    vehicle_index_by_track: dict[int, int] = {}

    for vehicle_index, state in enumerate(eligible_states, start=1):
        state.vehicle_index = vehicle_index
        vehicle_index_by_track[state.track_id] = vehicle_index
        vehicle_dir = vehicle_crop_dir(run_store.crops_dir, vehicle_index)
        vehicle_dir.mkdir(parents=True, exist_ok=True)
        finalized_candidates: list[CropCandidate] = []
        for candidate in state.candidates:
            destination = vehicle_dir / candidate.image_path.name
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
