from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import orjson

from config import AppConfig
from mmr.trafficeye import TrafficEyeClient
from models import MMRResult, TrackSummary
from storage.run_store import RunStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ClassificationTask:
    image_path: Path
    summaries: list[TrackSummary]
    vehicle_index: int | None
    fallback_index: int | None


def _load_track_summaries(path: Path) -> list[TrackSummary]:
    summaries: list[TrackSummary] = []
    if not path.exists():
        return summaries
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                summaries.append(TrackSummary.model_validate(orjson.loads(line)))
    return summaries


def _summary_vehicle_key(summary: TrackSummary) -> int:
    return summary.vehicle_index or summary.track_id


def _apply_identity(
    result: MMRResult,
    vehicle_index: int | None,
    fallback_index: int | None,
) -> MMRResult:
    label_index = vehicle_index or fallback_index
    return result.model_copy(
        update={
            "vehicle_index": vehicle_index,
            "api_classification_index": label_index,
        }
    )


def _chunks(
    items: list[_ClassificationTask], size: int
) -> list[list[_ClassificationTask]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _recognize_tasks(
    client: TrafficEyeClient, tasks: list[_ClassificationTask], batch_size: int
) -> list[MMRResult]:
    if not tasks:
        return []
    if batch_size <= 1:
        return [client.recognize_vehicle_crop(task.image_path) for task in tasks]
    if hasattr(client, "recognize_vehicle_crops"):
        return client.recognize_vehicle_crops([task.image_path for task in tasks])
    return [client.recognize_vehicle_crop(task.image_path) for task in tasks]


def _collect_classification_tasks(
    config: AppConfig, run_store: RunStore
) -> tuple[list[_ClassificationTask], dict[int, MMRResult]]:
    labels_by_track: dict[int, MMRResult] = {}
    classification_tasks: list[_ClassificationTask] = []
    summaries_by_vehicle: dict[int, list[TrackSummary]] = {}
    for summary in _load_track_summaries(run_store.tracks_path):
        if summary.vehicle_index is None and not summary.candidates:
            continue
        summaries_by_vehicle.setdefault(_summary_vehicle_key(summary), []).append(
            summary
        )

    legacy_vehicle_index_count = 0
    grouped_summaries = sorted(
        summaries_by_vehicle.values(),
        key=lambda items: (
            min(item.first_frame_index for item in items),
            min(item.track_id for item in items),
        ),
    )
    for summaries in grouped_summaries:
        vehicle_index = next(
            (summary.vehicle_index for summary in summaries if summary.vehicle_index),
            None,
        )
        fallback_index = None
        frames_seen = sum(summary.frames_seen for summary in summaries)
        if (
            config.analysis.min_track_frames > 0
            and frames_seen < config.analysis.min_track_frames
        ):
            logger.info(
                "Skipping MMR for short track group %s: %s frames < %s",
                [summary.track_id for summary in summaries],
                frames_seen,
                config.analysis.min_track_frames,
            )
            continue
        if vehicle_index is None:
            legacy_vehicle_index_count += 1
            fallback_index = legacy_vehicle_index_count

        candidates = [
            candidate for summary in summaries for candidate in summary.candidates
        ]
        best_candidate = (
            max(candidates, key=lambda item: item.total_score) if candidates else None
        )
        if best_candidate is None:
            result = _apply_identity(
                MMRResult(
                    make="unknown",
                    model="unknown",
                    accepted=False,
                    raw={
                        "skipped_reason": "no_crop_candidates",
                        "track_ids": [summary.track_id for summary in summaries],
                    },
                ),
                vehicle_index,
                fallback_index,
            )
            for summary in summaries:
                labels_by_track[summary.track_id] = result
            continue

        classification_tasks.append(
            _ClassificationTask(
                image_path=_resolve_candidate_image_path(
                    best_candidate.image_path, run_store
                ),
                summaries=summaries,
                vehicle_index=vehicle_index,
                fallback_index=fallback_index,
            )
        )
    return classification_tasks, labels_by_track


def _resolve_candidate_image_path(path: Path, run_store: RunStore) -> Path:
    if path.exists():
        return path
    relocated_path = run_store.crops_dir / path.name
    if relocated_path.exists():
        logger.warning(
            "Using relocated crop %s for stale track path %s",
            relocated_path,
            path,
        )
        return relocated_path
    return path


def classify_tracks(config: AppConfig, run_store: RunStore) -> dict[int, MMRResult]:
    client = TrafficEyeClient(config=config, cache_dir=run_store.mmr_cache_dir)
    classification_tasks, labels_by_track = _collect_classification_tasks(
        config=config, run_store=run_store
    )

    for batch in _chunks(classification_tasks, config.mmr.batch_size):
        results = _recognize_tasks(client, batch, config.mmr.batch_size)
        if len(results) != len(batch):
            raise RuntimeError(
                f"TrafficEye returned {len(results)} MMR results for {len(batch)} crops"
            )
        for task, result in zip(batch, results, strict=True):
            result = _apply_identity(result, task.vehicle_index, task.fallback_index)
            for summary in task.summaries:
                labels_by_track[summary.track_id] = result

    serializable = {
        str(track_id): result.model_dump(mode="json")
        for track_id, result in labels_by_track.items()
    }
    run_store.write_json(run_store.labels_path, serializable)
    logger.info("Classification complete for %s tracks", len(labels_by_track))
    return labels_by_track


def write_skipped_classification_batch_grids(
    config: AppConfig, run_store: RunStore
) -> list[Path]:
    client = TrafficEyeClient(
        config=config,
        cache_dir=run_store.mmr_cache_dir,
        require_api_key=False,
    )
    classification_tasks, _labels_by_track = _collect_classification_tasks(
        config=config, run_store=run_store
    )
    batch_grid_paths: list[Path] = []
    for batch in _chunks(classification_tasks, config.mmr.batch_size):
        batch_grid_path = client.write_vehicle_crop_grid(
            [task.image_path for task in batch]
        )
        if batch_grid_path is not None:
            batch_grid_paths.append(batch_grid_path)
    logger.info(
        "Wrote %s MMR batch grids without classification", len(batch_grid_paths)
    )
    return batch_grid_paths
