from pathlib import Path

import numpy as np
import pytest

from utils.video import (
    VideoMetadata,
    build_video_writer,
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
