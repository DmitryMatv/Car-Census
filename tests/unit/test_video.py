from io import BytesIO
from pathlib import Path
from subprocess import CompletedProcess

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


class _FakeFFmpegProcess:
    def __init__(
        self,
        *,
        stdin,
        stdout,
        return_code: int,
    ) -> None:
        self.stdin = BytesIO() if stdin == video_module.subprocess.PIPE else None
        self.stdout = BytesIO() if stdout == video_module.subprocess.PIPE else None
        self.stderr = None
        self._return_code = return_code

    def wait(self) -> int:
        return self._return_code


def _install_fake_ffmpeg_process(
    monkeypatch: pytest.MonkeyPatch,
    *,
    return_code: int = 0,
    stderr: bytes = b"",
) -> list[tuple[list[str], dict[str, object]]]:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_popen(command, **kwargs):
        stderr_target = kwargs["stderr"]
        stderr_target.write(stderr)
        stderr_target.flush()
        calls.append((command, kwargs))
        return _FakeFFmpegProcess(
            stdin=kwargs.get("stdin"),
            stdout=kwargs.get("stdout"),
            return_code=return_code,
        )

    monkeypatch.setattr(video_module.subprocess, "Popen", fake_popen)
    return calls


def _raw_ffmpeg_writer(tmp_path: Path):
    return video_module.FFmpegRawVideoWriter(
        output_path=tmp_path / "raw.mp4",
        fps=30.0,
        width=16,
        height=16,
        ffmpeg_path="ffmpeg",
        encoder="h264_nvenc",
        preset="p4",
        cq=23,
    )


def _cpu_ffmpeg_writer(tmp_path: Path):
    return video_module.FFmpegCPUFrameWriter(
        output_path=tmp_path / "cpu.mp4",
        fps=30.0,
        width=16,
        height=16,
        ffmpeg_path="ffmpeg",
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


def test_iter_sampled_frames_samples_30_fps_source_at_15_fps(tmp_path: Path) -> None:
    video_path = tmp_path / "sample-15-fps.mp4"
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

    frames = list(iter_sampled_frames(video_path, source_fps=30.0, target_fps=15.0))

    assert [frame_index for frame_index, _timestamp, _frame in frames] == [
        0,
        2,
        4,
        6,
        8,
    ]
    assert [timestamp for _frame_index, timestamp, _frame in frames] == pytest.approx(
        [0.0, 1 / 15, 2 / 15, 3 / 15, 4 / 15]
    )


def test_iter_sampled_frames_can_include_unsampled_terminal_frame(
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
        for index in range(8):
            frame = np.full((16, 16, 3), index * 10, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()

    frames = list(
        iter_sampled_frames(
            video_path,
            source_fps=30.0,
            target_fps=10.0,
            include_terminal_frame=True,
        )
    )

    assert [frame_index for frame_index, _timestamp, _frame in frames] == [0, 3, 6, 7]
    assert [timestamp for _frame_index, timestamp, _frame in frames] == pytest.approx(
        [0.0, 0.1, 0.2, 7 / 30]
    )


def test_iter_sampled_frames_does_not_duplicate_terminal_frame(
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
        for index in range(10):
            frame = np.full((16, 16, 3), index * 10, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()

    frames = list(
        iter_sampled_frames(
            video_path,
            source_fps=30.0,
            target_fps=10.0,
            include_terminal_frame=True,
        )
    )

    assert [frame_index for frame_index, _timestamp, _frame in frames] == [0, 3, 6, 9]


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


def test_ffmpeg_streaming_processes_disable_progress_and_use_file_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_fake_ffmpeg_process(monkeypatch)
    raw_writer = _raw_ffmpeg_writer(tmp_path)
    cpu_writer = _cpu_ffmpeg_writer(tmp_path)
    decoder = video_module.FFmpegFrameDecoder(tmp_path / "input.mp4")

    try:
        assert len(calls) == 3
        for command, kwargs in calls:
            assert command[1:5] == [
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostats",
            ]
            assert kwargs["stderr"] != video_module.subprocess.PIPE
            assert hasattr(kwargs["stderr"], "seek")
    finally:
        raw_writer.release()
        cpu_writer.release()
        decoder.close()


def test_ffmpeg_temp_stderr_preserves_failure_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_ffmpeg_process(
        monkeypatch,
        return_code=1,
        stderr=b"simulated ffmpeg failure",
    )
    writer = _cpu_ffmpeg_writer(tmp_path)

    with pytest.raises(RuntimeError, match="simulated ffmpeg failure"):
        writer.release()

    assert writer._stderr_file.closed


@pytest.mark.parametrize("process_kind", ["raw", "cpu", "decoder"])
def test_ffmpeg_start_failure_closes_temp_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    process_kind: str,
) -> None:
    created_stderr_files = []
    temporary_file = video_module.tempfile.TemporaryFile

    def tracked_temporary_file(*args, **kwargs):
        handle = temporary_file(*args, **kwargs)
        created_stderr_files.append(handle)
        return handle

    def fail_popen(*args, **kwargs):
        _ = args, kwargs
        raise OSError("cannot start")

    monkeypatch.setattr(video_module.tempfile, "TemporaryFile", tracked_temporary_file)
    monkeypatch.setattr(video_module.subprocess, "Popen", fail_popen)

    with pytest.raises(RuntimeError, match="Could not start FFmpeg"):
        if process_kind == "raw":
            _raw_ffmpeg_writer(tmp_path)
        elif process_kind == "cpu":
            _cpu_ffmpeg_writer(tmp_path)
        else:
            video_module.FFmpegFrameDecoder(tmp_path / "input.mp4")

    assert len(created_stderr_files) == 1
    assert created_stderr_files[0].closed


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

    def fake_run(command, **_kwargs) -> CompletedProcess[str]:
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
