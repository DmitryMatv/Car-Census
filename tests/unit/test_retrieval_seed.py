from __future__ import annotations

import cv2
import numpy as np

from mmr.retrieval_cache import MMRRetrievalStore
from mmr.retrieval_seed import seed_retrieval_cache
from mmr.trafficeye import build_single_request_payload
from mmr.trafficeye_cache import hash_request
from models import MMRResult, RunManifest, TrackSummary
from storage.run_store import RunStore


def test_seed_retrieval_cache_imports_selected_accepted_run_labels(
    config_factory, tmp_path
) -> None:
    run_dir = tmp_path / "selected-run"
    store = RunStore(run_dir)
    store.ensure_directories()
    crop_path = store.crops_dir / "vehicle.jpg"
    image = np.full((40, 60, 3), 120, dtype=np.uint8)
    assert cv2.imwrite(str(crop_path), image)
    store.manifest.write(
        RunManifest(
            run_id=run_dir.name,
            video_path=tmp_path / "video.mp4",
            camera_id="__full_frame__",
            root_dir=run_dir,
            source_fps=30.0,
            analysis_fps=10.0,
            width=60,
            height=40,
        )
    )
    store.tracks.write_all(
        [
            TrackSummary(
                track_id=1,
                vehicle_index=1,
                first_frame_index=1,
                last_frame_index=10,
                frames_seen=10,
                max_box_width_px=60,
                candidates=[],
            )
        ]
    )
    store.labels.write(
        {
            1: MMRResult(
                make="Toyota",
                model="Corolla",
                make_confidence=0.95,
                accepted=True,
                source_image=crop_path,
            ),
            2: MMRResult(make="Audi", model="A4", accepted=False),
        }
    )

    config = config_factory(None)
    cache_dir = tmp_path / "shared-cache"
    summaries = seed_retrieval_cache(
        run_dirs=[run_dir], config=config, cache_dir=cache_dir
    )

    assert summaries[0].imported == 1
    assert summaries[0].skipped_unaccepted == 1
    assert summaries[0].skipped_missing_image == 0

    image_bytes = crop_path.read_bytes()
    payload = build_single_request_payload(
        width=60,
        height=40,
        mmr_preference=config.mmr.mmr_preference,
    )
    lookup = MMRRetrievalStore(
        cache_dir / "retrieval",
        retrieval_mode=config.mmr.retrieval_mode,
        embedding_model=config.mmr.retrieval_embedding_model,
        embedding_dimensions=config.mmr.retrieval_embedding_dimensions,
        embedding_distance_threshold=config.mmr.retrieval_embedding_distance_threshold,
        phash_max_hamming_distance=config.mmr.retrieval_phash_max_hamming_distance,
        min_neighbors=config.mmr.retrieval_min_neighbors,
        min_make_confidence=config.mmr.accept_model_confidence,
    ).lookup(
        image_bytes=image_bytes,
        request_hash=hash_request(image_bytes, payload),
        request_payload=payload,
    )

    assert lookup.reason == "exact_match"
    assert lookup.match is not None
    assert lookup.match.result.make == "Toyota"
