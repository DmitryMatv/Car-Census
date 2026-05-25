from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RunLayout:
    root: Path

    @property
    def analysis_dir(self) -> Path:
        return self.root / "analysis"

    @property
    def crops_dir(self) -> Path:
        return self.root / "crops"

    @property
    def mmr_dir(self) -> Path:
        return self.root / "mmr"

    @property
    def mmr_cache_dir(self) -> Path:
        return self.mmr_dir / "cache"

    @property
    def mmr_batch_grids_dir(self) -> Path:
        return self.mmr_dir / "batch_grids"

    @property
    def manifest_path(self) -> Path:
        return self.root / "run.json"

    @property
    def frames_path(self) -> Path:
        return self.analysis_dir / "frames.jsonl"

    @property
    def render_frames_path(self) -> Path:
        return self.analysis_dir / "render_frames.jsonl"

    @property
    def tracks_path(self) -> Path:
        return self.analysis_dir / "tracks.jsonl"

    @property
    def detection_stats_path(self) -> Path:
        return self.analysis_dir / "detection_stats.json"

    @property
    def count_events_path(self) -> Path:
        return self.analysis_dir / "count_events.jsonl"

    @property
    def labels_path(self) -> Path:
        return self.mmr_dir / "labels.json"

    @property
    def output_video_path(self) -> Path:
        return self.root / "annotated.mp4"

    @property
    def report_csv_path(self) -> Path:
        return self.root / "report.csv"

    def ensure_directories(self) -> None:
        for path in [
            self.root,
            self.analysis_dir,
            self.crops_dir,
            self.mmr_dir,
            self.mmr_cache_dir,
            self.mmr_batch_grids_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)
