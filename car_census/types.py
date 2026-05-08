from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


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


class Detection(BaseModel):
    bbox: BBox
    confidence: float
    class_id: int | None = None
    class_name: str | None = None


class TrackedObject(BaseModel):
    track_id: int
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
    frame_index: int
    timestamp_seconds: float
    bbox: BBox
    image_path: Path
    sharpness: float
    edge_margin_score: float
    area_score: float
    total_score: float


class CountEvent(BaseModel):
    track_id: int
    frame_index: int
    timestamp_seconds: float
    direction: str


class MMRResult(BaseModel):
    make: str | None = None
    model: str | None = None
    make_confidence: float | None = None
    model_confidence: float | None = None
    raw: dict[str, Any] = Field(default_factory=dict)
    accepted: bool = False
    source_image: Path | None = None


class TrackSummary(BaseModel):
    track_id: int
    first_frame_index: int
    last_frame_index: int
    frames_seen: int
    max_box_height_px: float
    counted: bool = False
    count_event: CountEvent | None = None
    candidates: list[CropCandidate] = Field(default_factory=list)
    final_label: MMRResult | None = None


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
