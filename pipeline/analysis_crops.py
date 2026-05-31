from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2

from config import AppConfig
from models import BBox, CropCandidate
from pipeline.vehicles import staged_track_crop_dir
from storage.run_store import RunStore
from utils.image_quality import laplacian_sharpness


class CandidateTrackState(Protocol):
    track_id: int
    min_box_width_px: float | None
    max_box_width_px: float
    candidates: list[CropCandidate]
    last_candidate_time: float | None


class CropStore(Protocol):
    @property
    def crops_dir(self) -> Path: ...


@dataclass(frozen=True, order=True, slots=True)
class CropCandidateRank:
    scale_score: float
    sharpness: float
    edge_margin_score: float
    area_score: float
    recency_score: int


def score_candidate(
    crop: cv2.typing.MatLike, bbox: BBox, frame_shape: tuple[int, int, int]
) -> tuple[float, float, float]:
    sharpness = laplacian_sharpness(crop)
    area_score = bbox.area
    height, width = frame_shape[:2]
    margin_left = bbox.x1
    margin_top = bbox.y1
    margin_right = width - bbox.x2
    margin_bottom = height - bbox.y2
    edge_margin_score = min(margin_left, margin_top, margin_right, margin_bottom)
    return sharpness, edge_margin_score, area_score


def crop_candidate_rank(
    *,
    bbox_width: float,
    sharpness: float,
    edge_margin_score: float,
    area_score: float,
    frame_index: int,
    min_box_width_px: float | None,
    max_box_width_px: float,
    target_ratio: float,
) -> CropCandidateRank:
    min_width = min_box_width_px if min_box_width_px is not None else max_box_width_px
    target_width = min_width + ((max_box_width_px - min_width) * target_ratio)
    scale_error = abs(bbox_width - target_width) / max(target_width, 1.0)
    return CropCandidateRank(
        scale_score=-scale_error,
        sharpness=sharpness,
        edge_margin_score=edge_margin_score,
        area_score=area_score,
        recency_score=-frame_index,
    )


def rank_crop_candidate(
    candidate: CropCandidate,
    min_box_width_px: float | None,
    max_box_width_px: float,
    config: AppConfig,
) -> CropCandidateRank:
    vehicle_bbox = candidate.vehicle_bbox or candidate.bbox
    return crop_candidate_rank(
        bbox_width=vehicle_bbox.width,
        sharpness=candidate.sharpness,
        edge_margin_score=candidate.edge_margin_score,
        area_score=candidate.area_score,
        frame_index=candidate.frame_index,
        min_box_width_px=min_box_width_px,
        max_box_width_px=max_box_width_px,
        target_ratio=config.analysis.crop_target_box_range_ratio,
    )


def expand_crop_bbox(bbox: BBox, config: AppConfig) -> BBox:
    padding_ratio = config.analysis.crop_padding_ratio
    padding_px = config.analysis.crop_padding_px
    pad_x = (bbox.width * padding_ratio) + padding_px
    pad_y = (bbox.height * padding_ratio) + padding_px
    return BBox(
        x1=bbox.x1 - pad_x,
        y1=bbox.y1 - pad_y,
        x2=bbox.x2 + pad_x,
        y2=bbox.y2 + pad_y,
    )


def save_candidate(
    store: CropStore,
    track_state: CandidateTrackState,
    frame: cv2.typing.MatLike,
    bbox: BBox,
    frame_index: int,
    timestamp_seconds: float,
    config: AppConfig,
) -> None:
    from roi.geometry import clip_bbox_to_frame

    clipped = clip_bbox_to_frame(expand_crop_bbox(bbox, config), frame.shape)
    if clipped is None:
        return
    if (
        track_state.last_candidate_time is not None
        and timestamp_seconds - track_state.last_candidate_time
        < config.analysis.crop_min_spacing_seconds
    ):
        return
    crop = frame[int(clipped.y1) : int(clipped.y2), int(clipped.x1) : int(clipped.x2)]
    if crop.size == 0:
        return
    sharpness, edge_margin_score, area_score = score_candidate(
        crop, clipped, frame.shape
    )
    track_dir = staged_track_crop_dir(store.crops_dir, track_state.track_id)
    track_dir.mkdir(parents=True, exist_ok=True)
    image_path = track_dir / f"frame_{frame_index:08d}.jpg"
    candidate = CropCandidate(
        track_id=track_state.track_id,
        vehicle_index=None,
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
        bbox=clipped,
        vehicle_bbox=bbox,
        image_path=image_path,
        sharpness=sharpness,
        edge_margin_score=edge_margin_score,
        area_score=area_score,
    )
    current = track_state.candidates[0] if track_state.candidates else None
    if current is not None and rank_crop_candidate(
        current,
        track_state.min_box_width_px,
        track_state.max_box_width_px,
        config,
    ) >= rank_crop_candidate(
        candidate,
        track_state.min_box_width_px,
        track_state.max_box_width_px,
        config,
    ):
        return

    cv2.imwrite(
        str(image_path),
        crop,
        [cv2.IMWRITE_JPEG_QUALITY, config.analysis.crop_jpeg_quality],
    )
    if current is not None and current.image_path.exists():
        current.image_path.unlink()
    track_state.candidates = [candidate]
    track_state.last_candidate_time = timestamp_seconds


def render_bbox_for_track(
    bbox: BBox, frame_shape: tuple[int, int, int], config: AppConfig
) -> BBox:
    from roi.geometry import clip_bbox_to_frame

    return clip_bbox_to_frame(expand_crop_bbox(bbox, config), frame_shape) or bbox


class CropCandidateSelector:
    def __init__(self, config: AppConfig, store: RunStore) -> None:
        self.config = config
        self.store = store

    def maybe_save_candidate(
        self,
        *,
        track_state: CandidateTrackState,
        frame: cv2.typing.MatLike,
        bbox: BBox,
        frame_index: int,
        timestamp_seconds: float,
    ) -> None:
        if bbox.width < self.config.analysis.min_box_width_px:
            return
        save_candidate(
            store=self.store,
            track_state=track_state,
            frame=frame,
            bbox=bbox,
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            config=self.config,
        )

    def render_bbox_for_track(
        self,
        bbox: BBox,
        frame_shape: tuple[int, int, int],
    ) -> BBox:
        return render_bbox_for_track(bbox, frame_shape, self.config)
