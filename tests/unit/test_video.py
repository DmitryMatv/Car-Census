from pathlib import Path

import cv2
import numpy as np

from utils.video import build_video_writer, iter_sampled_frames


def test_iter_sampled_frames_uses_all_frames_when_target_fps_is_zero(
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

    frames = list(iter_sampled_frames(video_path, target_fps=0.0))
    assert [frame_index for frame_index, _timestamp, _frame in frames] == [
        0,
        1,
        2,
        3,
        4,
    ]


def test_iter_sampled_frames_uses_all_frames_when_target_fps_exceeds_source(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "sample.mp4"
    writer = build_video_writer(
        output_path=video_path,
        fps=15.0,
        width=16,
        height=16,
        codec="mp4v",
    )
    try:
        for _index in range(4):
            writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
    finally:
        writer.release()

    capture = cv2.VideoCapture(str(video_path))
    try:
        assert capture.isOpened() is True
    finally:
        capture.release()

    frames = list(iter_sampled_frames(video_path, target_fps=30.0))
    assert [frame_index for frame_index, _timestamp, _frame in frames] == [0, 1, 2, 3]
