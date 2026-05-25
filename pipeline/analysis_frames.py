from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2

from config import CameraProfile
from detectors.base import Detector
from models import Detection
from roi.geometry import crop_to_polygon, map_bbox_to_global
from utils.video import iter_sampled_frames

SampledFramesFunc = Callable[
    [Path, float, float], Iterator[tuple[int, float, cv2.typing.MatLike]]
]


@dataclass(slots=True)
class SampledFrame:
    frame_index: int
    timestamp_seconds: float
    frame: cv2.typing.MatLike
    roi_frame: cv2.typing.MatLike
    offset: tuple[int, int]


def iter_sampled_frame_batches(
    *,
    video_path: Path,
    source_fps: float,
    target_fps: float,
    profile: CameraProfile,
    batch_size: int,
    iter_sampled_frames_func: SampledFramesFunc = iter_sampled_frames,
) -> Iterator[list[SampledFrame]]:
    batch: list[SampledFrame] = []
    for frame_index, timestamp_seconds, frame in iter_sampled_frames_func(
        video_path, source_fps, target_fps
    ):
        roi_frame, offset = crop_to_polygon(frame, profile.polygon.points)
        batch.append(
            SampledFrame(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                frame=frame,
                roi_frame=roi_frame,
                offset=offset,
            )
        )
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def iter_detected_sampled_frames(
    *,
    detector: Detector,
    video_path: Path,
    source_fps: float,
    target_fps: float,
    profile: CameraProfile,
    batch_size: int,
    iter_sampled_frames_func: SampledFramesFunc = iter_sampled_frames,
) -> Iterator[tuple[SampledFrame, list[Detection]]]:
    for batch in iter_sampled_frame_batches(
        video_path=video_path,
        source_fps=source_fps,
        target_fps=target_fps,
        profile=profile,
        batch_size=batch_size,
        iter_sampled_frames_func=iter_sampled_frames_func,
    ):
        batch_detections = detector.detect_batch([item.roi_frame for item in batch])
        if len(batch_detections) != len(batch):
            raise RuntimeError(
                f"Detector returned {len(batch_detections)} detection sets for "
                f"{len(batch)} frames"
            )
        yield from zip(batch, batch_detections, strict=True)


def map_detections_to_global(
    detections: Sequence[Detection],
    offset: tuple[int, int],
) -> list[Detection]:
    return [
        detection.model_copy(
            update={"bbox": map_bbox_to_global(detection.bbox, offset)}
        )
        for detection in detections
    ]
