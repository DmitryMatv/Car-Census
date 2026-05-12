from __future__ import annotations

import logging
from pathlib import Path

import orjson

from config import AppConfig
from mmr.trafficeye import TrafficEyeClient
from storage.run_store import RunStore
from models import MMRResult, TrackSummary

logger = logging.getLogger(__name__)


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


def classify_tracks(config: AppConfig, run_store: RunStore) -> dict[int, MMRResult]:
    client = TrafficEyeClient(config=config, cache_dir=run_store.mmr_cache_dir)
    labels_by_track: dict[int, MMRResult] = {}
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

        result = client.recognize_vehicle_crop(best_candidate.image_path)
        result = _apply_identity(result, vehicle_index, fallback_index)
        for summary in summaries:
            labels_by_track[summary.track_id] = result

    serializable = {
        str(track_id): result.model_dump(mode="json")
        for track_id, result in labels_by_track.items()
    }
    run_store.write_json(run_store.labels_path, serializable)
    logger.info("Classification complete for %s tracks", len(labels_by_track))
    return labels_by_track
