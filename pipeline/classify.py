from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from config import AppConfig, CameraProfile
from mmr.trafficeye import TrafficEyeClient
from models import CropCandidate, MMRResult, TrackSummary
from pipeline.analysis_crops import rank_crop_candidate
from pipeline.sequential_duplicates import deduplicate_classified_tracks
from storage.run_store import RunStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ClassificationTask:
    image_path: Path
    summaries: list[TrackSummary]
    vehicle_index: int


def _apply_identity(result: MMRResult, vehicle_index: int) -> MMRResult:
    return result.model_copy(
        update={
            "vehicle_index": vehicle_index,
            "api_classification_index": vehicle_index,
        }
    )


def _apply_default_provenance(result: MMRResult) -> MMRResult:
    if result.evidence_source is not None and result.resolution_method is not None:
        return result
    return result.model_copy(
        update={
            "evidence_source": result.evidence_source or "api_confirmed",
            "resolution_method": result.resolution_method or "external_api",
        }
    )


def _classification_cache_dir(config: AppConfig, run_store: RunStore) -> Path:
    if isinstance(run_store, RunStore):
        manifest = run_store.manifest.read()
        if manifest.retrieval_cache_dir is not None:
            return manifest.retrieval_cache_dir
        return run_store.root.parent / config.project.retrieval_cache_dir
    return run_store.mmr_cache_dir


def _build_client(config: AppConfig, run_store: RunStore) -> TrafficEyeClient:
    client = TrafficEyeClient(
        config=config,
        cache_dir=_classification_cache_dir(config, run_store),
    )
    client.batch_grids_dir = run_store.mmr_batch_grids_dir
    return client


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
    return client.recognize_vehicle_crops([task.image_path for task in tasks])


def _best_candidate_for_summaries(
    summaries: list[TrackSummary], config: AppConfig
) -> CropCandidate | None:
    ranked_candidates = [
        (summary, candidate)
        for summary in summaries
        for candidate in summary.candidates
    ]
    if not ranked_candidates:
        return None
    _summary, candidate = max(
        ranked_candidates,
        key=lambda item: rank_crop_candidate(
            item[1],
            item[0].min_box_width_px,
            item[0].max_box_width_px,
            config,
        ),
    )
    return candidate


def _collect_classification_tasks(
    config: AppConfig, run_store: RunStore
) -> tuple[list[_ClassificationTask], dict[int, MMRResult]]:
    labels_by_track: dict[int, MMRResult] = {}
    classification_tasks: list[_ClassificationTask] = []
    summaries_by_vehicle: dict[int, list[TrackSummary]] = {}
    for summary in run_store.tracks_effective.iter():
        if summary.vehicle_index is None:
            continue
        summaries_by_vehicle.setdefault(summary.vehicle_index, []).append(summary)

    grouped_summaries = sorted(
        summaries_by_vehicle.values(),
        key=lambda items: (
            min(item.first_frame_index for item in items),
            min(item.track_id for item in items),
        ),
    )
    for summaries in grouped_summaries:
        vehicle_index = summaries[0].vehicle_index
        assert vehicle_index is not None
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

        best_candidate = _best_candidate_for_summaries(summaries, config)
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


def _invalidate_stale_labels(run_store: RunStore) -> None:
    if run_store.labels_path.exists():
        run_store.labels_path.unlink()
        logger.warning(
            "Removed stale labels from a previous classification attempt"
        )


def classify_tracks(
    config: AppConfig,
    run_store: RunStore,
    profile: CameraProfile | None = None,
) -> dict[int, MMRResult]:
    _invalidate_stale_labels(run_store)
    client = _build_client(config=config, run_store=run_store)
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
            result = _apply_default_provenance(
                _apply_identity(result, task.vehicle_index)
            )
            for summary in task.summaries:
                labels_by_track[summary.track_id] = result

    run_store.labels.write(labels_by_track)
    if isinstance(run_store, RunStore):
        deduplicate_classified_tracks(
            config=config, run_store=run_store, profile=profile
        )
        labels_by_track = run_store.labels.read()
    logger.info("Classification complete for %s tracks", len(labels_by_track))
    return labels_by_track


def write_skipped_classification_batch_grids(
    config: AppConfig, run_store: RunStore
) -> list[Path]:
    client = TrafficEyeClient(
        config=config,
        cache_dir=_classification_cache_dir(config, run_store),
        require_api_key=False,
    )
    client.batch_grids_dir = run_store.mmr_batch_grids_dir
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
