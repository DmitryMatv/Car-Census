from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def bottom_center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, self.y2)

    def intersection_area(self, other: BBox) -> float:
        width = max(0.0, min(self.x2, other.x2) - max(self.x1, other.x1))
        height = max(0.0, min(self.y2, other.y2) - max(self.y1, other.y1))
        return width * height

    def iou(self, other: BBox) -> float:
        intersection = self.intersection_area(other)
        union = self.area + other.area - intersection
        return intersection / union if union > 0.0 else 0.0

    def smaller_coverage(self, other: BBox) -> float:
        smaller_area = min(self.area, other.area)
        if smaller_area <= 0.0:
            return 0.0
        return self.intersection_area(other) / smaller_area

    def area_ratio(self, other: BBox) -> float:
        larger_area = max(self.area, other.area)
        if larger_area <= 0.0:
            return 0.0
        return min(self.area, other.area) / larger_area

    def center_distance(self, other: BBox) -> float:
        x, y = self.center
        other_x, other_y = other.center
        return math.hypot(x - other_x, y - other_y)

    def contains_point(
        self,
        point: tuple[float, float],
        *,
        inclusive: bool = True,
    ) -> bool:
        x, y = point
        if inclusive:
            return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2
        return self.x1 <= x < self.x2 and self.y1 <= y < self.y2


class TrackedObject(BaseModel):
    track_id: int
    vehicle_index: int | None = None
    frame_index: int
    timestamp_seconds: float
    bbox: BBox
    confidence: float
    class_id: int | None = None
    class_name: str | None = None
    centroid: tuple[float, float]
    bottom_center: tuple[float, float]
    inside_roi: bool
    counted: bool = False
    crossed_line: bool = False


class CropCandidate(BaseModel):
    track_id: int
    vehicle_index: int | None = None
    frame_index: int
    timestamp_seconds: float
    bbox: BBox
    vehicle_bbox: BBox | None = None
    image_path: Path
    sharpness: float
    edge_margin_score: float
    area_score: float
    # Fraction of the vehicle box covered by other live tracks' boxes in the
    # same frame (0.0 on records written before the contamination tier was
    # introduced; those rank as clean).
    sibling_overlap_fraction: float = 0.0


class CountEvent(BaseModel):
    track_id: int
    frame_index: int
    timestamp_seconds: float
    direction: str


class CachedFrameDetection(BaseModel):
    bbox: BBox
    confidence: float
    class_id: int | None = None
    class_name: str | None = None


class CachedFrameDetections(BaseModel):
    """Raw detector output for one sampled analysis frame, stored verbatim
    so tracking/linking can be re-run without re-paying detector inference.
    Detections are in global frame coordinates, exactly as passed to the
    tracker; ``edge_suppressed_bboxes`` echoes the boxes the edge-suppression
    layer flagged for that frame."""

    frame_index: int
    timestamp_seconds: float
    detections: list[CachedFrameDetection] = Field(default_factory=list)
    edge_suppressed_bboxes: list[BBox] = Field(default_factory=list)


class MMRResult(BaseModel):
    make: str | None = None
    model: str | None = None
    make_confidence: float | None = None
    model_confidence: float | None = None
    category: str | None = None
    category_confidence: float | None = None
    generation: str | None = None
    generation_confidence: float | None = None
    variation: str | None = None
    variation_confidence: float | None = None
    color: str | None = None
    color_confidence: float | None = None
    view: str | None = None
    view_confidence: float | None = None
    view8: str | None = None
    view8_confidence: float | None = None
    tags: list[dict[str, Any]] = Field(default_factory=list)
    detection_box: BBox | None = None
    detection_confidence: float | None = None
    raw: dict[str, Any] = Field(default_factory=dict)
    accepted: bool = False
    source_image: Path | None = None
    vehicle_index: int | None = None
    api_classification_index: int | None = None
    evidence_source: Literal["api_confirmed", "human_adjudicated"] | None = None
    resolution_method: (
        Literal[
            "external_api",
            "exact_retrieval",
            "embedding_retrieval",
            "human_adjudication",
        ]
        | None
    ) = None
    retrieval_record_id: str | None = None
    retrieval_distance: float | None = None
    retrieval_neighbor_count: int | None = None


class TrackSummary(BaseModel):
    track_id: int
    vehicle_index: int | None = None
    first_frame_index: int
    last_frame_index: int
    frames_seen: int
    min_box_width_px: float | None = None
    max_box_width_px: float
    min_box_height_px: float | None = None
    max_box_height_px: float | None = None
    speed_mps_median: float | None = None
    speed_mps_max: float | None = None
    counted: bool = False
    count_event: CountEvent | None = None
    candidates: list[CropCandidate] = Field(default_factory=list)
    final_label: MMRResult | None = None

    @model_validator(mode="before")
    @classmethod
    def _migrate_height_metric(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        migrated = dict(data)
        # Legacy artifacts used *_box_height_px to mean width. Copy the value
        # into the width fields and drop the keys so they are not mistaken
        # for the modern per-dimension height stats (which stay unset/None).
        # Modern records carry both width and height keys and are kept as-is.
        for legacy, current in (
            ("min_box_height_px", "min_box_width_px"),
            ("max_box_height_px", "max_box_width_px"),
        ):
            if current not in migrated and legacy in migrated:
                migrated[current] = migrated.pop(legacy)
        return migrated


class FrameRecord(BaseModel):
    frame_index: int
    timestamp_seconds: float
    tracks: list[TrackedObject] = Field(default_factory=list)


class RunManifest(BaseModel):
    run_id: str
    video_path: Path
    camera_id: str | None = None
    root_dir: Path
    source_fps: float
    analysis_fps: float
    width: int
    height: int
    frame_count: int = 0
    retrieval_cache_dir: Path | None = None
