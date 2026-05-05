from __future__ import annotations

import logging
from pathlib import Path

import orjson

from car_census.config import AppConfig
from car_census.mmr.trafficeye import TrafficEyeClient
from car_census.storage.run_store import RunStore
from car_census.types import MMRResult, TrackSummary

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


def classify_tracks(config: AppConfig, run_store: RunStore) -> dict[int, MMRResult]:
    client = TrafficEyeClient(config=config, cache_dir=run_store.mmr_cache_dir)
    labels_by_track: dict[int, MMRResult] = {}
    for summary in _load_track_summaries(run_store.tracks_path):
        best_result: MMRResult | None = None
        attempts = 0
        for candidate in summary.candidates:
            attempts += 1
            result = client.recognize_vehicle_crop(candidate.image_path)
            if best_result is None or (result.model_confidence or 0.0) > (best_result.model_confidence or 0.0):
                best_result = result
            if result.accepted:
                best_result = result
                break
            if attempts >= config.mmr.max_attempts_per_track:
                break
        labels_by_track[summary.track_id] = best_result or MMRResult(make="unknown", model="unknown", accepted=False)

    serializable = {
        str(track_id): result.model_dump(mode="json")
        for track_id, result in labels_by_track.items()
    }
    run_store.write_json(run_store.labels_path, serializable)
    logger.info("Classification complete for %s tracks", len(labels_by_track))
    return labels_by_track
