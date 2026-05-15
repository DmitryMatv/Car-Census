from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Protocol

import cv2
import numpy as np

logger = logging.getLogger(__name__)


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


class FrameWriter(Protocol):
    def write(self, frame: np.ndarray) -> None:
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError


class OpenCVFrameWriter:
    def __init__(self, writer: cv2.VideoWriter) -> None:
        self.writer = writer

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def release(self) -> None:
        self.writer.release()


class FFmpegRawVideoWriter:
    def __init__(
        self,
        *,
        output_path: Path,
        fps: float,
        width: int,
        height: int,
        ffmpeg_path: str,
        encoder: str,
        preset: str,
        cq: int,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            ffmpeg_path,
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps}",
            "-i",
            "-",
            "-an",
            "-c:v",
            encoder,
            "-preset",
            preset,
            "-cq",
            str(cq),
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise RuntimeError(
                f"Could not start FFmpeg encoder '{ffmpeg_path}': {exc}"
            ) from exc
        self.output_path = output_path
        self.command = command

    def write(self, frame: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("FFmpeg writer stdin is closed")
        try:
            self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError("FFmpeg encoder stopped while writing frames") from exc

    def release(self) -> None:
        stderr = b""
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        if self.process.stderr is not None:
            stderr = self.process.stderr.read()
        return_code = self.process.wait()
        if return_code != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                "FFmpeg encoder failed with exit code "
                f"{return_code}: {detail or 'no stderr'}"
            )


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


def has_ffmpeg_encoder(ffmpeg_path: str, encoder: str) -> bool:
    try:
        result = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-encoders"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    if result.returncode != 0:
        return False
    if encoder not in result.stdout:
        return False
    try:
        probe = subprocess.run(
            [
                ffmpeg_path,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=size=128x128:rate=1:duration=1",
                "-frames:v",
                "1",
                "-an",
                "-c:v",
                encoder,
                "-f",
                "null",
                "-",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return probe.returncode == 0


def build_frame_writer(
    *,
    output_path: Path,
    fps: float,
    width: int,
    height: int,
    codec: str,
    encode_backend: Literal["opencv", "auto-nvenc", "ffmpeg-nvenc"],
    ffmpeg_path: str,
    nvenc_codec: str,
    nvenc_preset: str,
    nvenc_cq: int,
) -> FrameWriter:
    if encode_backend == "opencv":
        return OpenCVFrameWriter(
            build_video_writer(
                output_path=output_path,
                fps=fps,
                width=width,
                height=height,
                codec=codec,
            )
        )

    nvenc_available = has_ffmpeg_encoder(ffmpeg_path, nvenc_codec)
    if not nvenc_available:
        message = (
            f"FFmpeg encoder '{nvenc_codec}' is not available via '{ffmpeg_path}'."
        )
        if encode_backend == "ffmpeg-nvenc":
            raise RuntimeError(message)
        logger.warning("%s Falling back to OpenCV video writer.", message)
        return OpenCVFrameWriter(
            build_video_writer(
                output_path=output_path,
                fps=fps,
                width=width,
                height=height,
                codec=codec,
            )
        )

    try:
        return FFmpegRawVideoWriter(
            output_path=output_path,
            fps=fps,
            width=width,
            height=height,
            ffmpeg_path=ffmpeg_path,
            encoder=nvenc_codec,
            preset=nvenc_preset,
            cq=nvenc_cq,
        )
    except RuntimeError:
        if encode_backend == "ffmpeg-nvenc":
            raise
        logger.warning("Could not start FFmpeg NVENC writer; falling back to OpenCV.")
        return OpenCVFrameWriter(
            build_video_writer(
                output_path=output_path,
                fps=fps,
                width=width,
                height=height,
                codec=codec,
            )
        )
