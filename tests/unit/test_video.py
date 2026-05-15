from pathlib import Path

import numpy as np
import pytest

import utils.video as video_module
from utils.video import (
    OpenCVFrameWriter,
    VideoMetadata,
    build_frame_writer,
    build_video_writer,
    has_ffmpeg_encoder,
    iter_sampled_frames,
    iter_video_frames,
    validate_video_fps,
)


def test_iter_video_frames_yields_every_frame_with_configured_timestamps(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "sample.mp4"
    writer = build_video_writer(
        output_path=video_path,
        fps=30.0,
        width=16,
        height=16,
        codec="mp4v",
    )
    try:
        for index in range(5):
            frame = np.full((16, 16, 3), index * 10, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()

    frames = list(iter_video_frames(video_path, fps=30.0))

    assert [frame_index for frame_index, _timestamp, _frame in frames] == [
        0,
        1,
        2,
        3,
        4,
    ]
    assert [timestamp for _frame_index, timestamp, _frame in frames] == pytest.approx(
        [0.0, 1 / 30, 2 / 30, 3 / 30, 4 / 30]
    )


def test_iter_sampled_frames_uses_configured_analysis_fps(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    writer = build_video_writer(
        output_path=video_path,
        fps=30.0,
        width=16,
        height=16,
        codec="mp4v",
    )
    try:
        for index in range(10):
            frame = np.full((16, 16, 3), index * 10, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()

    frames = list(iter_sampled_frames(video_path, source_fps=30.0, target_fps=10.0))

    assert [frame_index for frame_index, _timestamp, _frame in frames] == [0, 3, 6, 9]
    assert [timestamp for _frame_index, timestamp, _frame in frames] == pytest.approx(
        [0.0, 0.1, 0.2, 0.3]
    )


def test_iter_sampled_frames_uses_all_frames_when_analysis_fps_exceeds_source(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "sample.mp4"
    writer = build_video_writer(
        output_path=video_path,
        fps=30.0,
        width=16,
        height=16,
        codec="mp4v",
    )
    try:
        for _index in range(4):
            writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
    finally:
        writer.release()

    frames = list(iter_sampled_frames(video_path, source_fps=30.0, target_fps=60.0))

    assert [frame_index for frame_index, _timestamp, _frame in frames] == [0, 1, 2, 3]


def test_validate_video_fps_accepts_nominal_29_97_with_default_tolerance() -> None:
    metadata = VideoMetadata(width=16, height=16, fps=29.97, frame_count=10)

    validate_video_fps(metadata, expected_fps=30.0, tolerance=0.05)


def test_validate_video_fps_rejects_wrong_fps() -> None:
    metadata = VideoMetadata(width=16, height=16, fps=25.0, frame_count=10)

    with pytest.raises(RuntimeError, match="Input video FPS"):
        validate_video_fps(metadata, expected_fps=30.0, tolerance=0.05)


def test_validate_video_fps_rejects_unknown_fps() -> None:
    metadata = VideoMetadata(width=16, height=16, fps=0.0, frame_count=10)

    with pytest.raises(RuntimeError, match="Could not determine input video FPS"):
        validate_video_fps(metadata, expected_fps=30.0, tolerance=0.05)


def test_build_frame_writer_uses_opencv_backend(tmp_path: Path) -> None:
    writer = build_frame_writer(
        output_path=tmp_path / "opencv.mp4",
        fps=30.0,
        width=16,
        height=16,
        codec="mp4v",
        encode_backend="opencv",
        ffmpeg_path="ffmpeg",
        nvenc_codec="h264_nvenc",
        nvenc_preset="p4",
        nvenc_cq=23,
    )
    try:
        assert isinstance(writer, OpenCVFrameWriter)
    finally:
        writer.release()


def test_has_ffmpeg_encoder_probe_uses_nvenc_supported_dimensions(monkeypatch) -> None:
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        stdout = " V....D h264_nvenc           NVIDIA NVENC H.264 encoder"
        return video_module.subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(video_module.subprocess, "run", fake_run)

    assert has_ffmpeg_encoder("ffmpeg", "h264_nvenc")

    assert "color=size=640x360:rate=1:duration=1" in commands[1]


def test_build_frame_writer_auto_nvenc_falls_back_to_opencv(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(video_module, "has_ffmpeg_encoder", lambda *args: False)

    writer = build_frame_writer(
        output_path=tmp_path / "fallback.mp4",
        fps=30.0,
        width=16,
        height=16,
        codec="mp4v",
        encode_backend="auto-nvenc",
        ffmpeg_path="ffmpeg",
        nvenc_codec="h264_nvenc",
        nvenc_preset="p4",
        nvenc_cq=23,
    )
    try:
        assert isinstance(writer, OpenCVFrameWriter)
    finally:
        writer.release()


def test_build_frame_writer_ffmpeg_nvenc_requires_encoder(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(video_module, "has_ffmpeg_encoder", lambda *args: False)

    with pytest.raises(RuntimeError, match="h264_nvenc"):
        build_frame_writer(
            output_path=tmp_path / "required.mp4",
            fps=30.0,
            width=16,
            height=16,
            codec="mp4v",
            encode_backend="ffmpeg-nvenc",
            ffmpeg_path="ffmpeg",
            nvenc_codec="h264_nvenc",
            nvenc_preset="p4",
            nvenc_cq=23,
        )
