from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(slots=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_seconds(self) -> float:
        if self.fps <= 0:
            return 0.0
        return self.frame_count / self.fps


def open_capture(video_path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    return capture


def read_video_metadata(video_path: Path) -> VideoMetadata:
    capture = open_capture(video_path)
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return VideoMetadata(
            width=width, height=height, fps=fps, frame_count=frame_count
        )
    finally:
        capture.release()


def read_first_frame(video_path: Path) -> np.ndarray:
    capture = open_capture(video_path)
    try:
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read first frame from video: {video_path}")
        return frame
    finally:
        capture.release()


def validate_video_fps(
    metadata: VideoMetadata, expected_fps: float, tolerance: float
) -> None:
    if metadata.fps <= 0:
        raise RuntimeError("Could not determine input video FPS.")
    if abs(metadata.fps - expected_fps) > tolerance:
        raise RuntimeError(
            "Input video FPS does not match configured FPS. "
            f"Expected {expected_fps:.3f} +/- {tolerance:.3f}, got {metadata.fps:.3f}."
        )


def iter_video_frames(
    video_path: Path,
    fps: float,
) -> Iterator[tuple[int, float, np.ndarray]]:
    capture = open_capture(video_path)
    try:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            timestamp = frame_index / fps
            yield frame_index, timestamp, frame
            frame_index += 1
    finally:
        capture.release()


def iter_sampled_frames(
    video_path: Path,
    source_fps: float,
    target_fps: float,
) -> Iterator[tuple[int, float, np.ndarray]]:
    if source_fps <= 0:
        raise RuntimeError(f"Invalid source FPS for video: {video_path}")
    if target_fps <= 0:
        raise RuntimeError(f"Invalid target FPS for video: {video_path}")

    capture = open_capture(video_path)
    try:
        frame_index = 0
        next_sample_time = 0.0
        sample_interval = 1.0 / min(target_fps, source_fps)
        epsilon = 1e-9
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            timestamp = frame_index / source_fps
            if timestamp + epsilon >= next_sample_time:
                yield frame_index, timestamp, frame
                next_sample_time += sample_interval
            frame_index += 1
    finally:
        capture.release()


def build_video_writer(
    output_path: Path,
    fps: float,
    width: int,
    height: int,
    codec: str,
) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_path}")
    return writer
