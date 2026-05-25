from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from storage.run_layout import RunLayout
from storage.run_repositories import RunRepositories


class RunStore:
    def __init__(self, root: Path | RunLayout) -> None:
        self.layout = root if isinstance(root, RunLayout) else RunLayout(root)
        self.repositories = RunRepositories.create(self.layout)
        self.manifest = self.repositories.manifest
        self.frames = self.repositories.frames
        self.tracks = self.repositories.tracks
        self.counts = self.repositories.counts
        self.labels = self.repositories.labels
        self.detection_stats = self.repositories.detection_stats

    @classmethod
    def create(
        cls,
        output_root: Path,
        camera_id: str,
        video_stem: str,
    ) -> "RunStore":
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_camera_id = "full-frame" if camera_id == "__full_frame__" else camera_id
        run_id = f"{run_camera_id}-{video_stem}-{timestamp}"
        store = cls(output_root / run_id)
        store.ensure_directories()
        return store

    @classmethod
    def from_existing(cls, run_dir: Path) -> "RunStore":
        store = cls(run_dir)
        store.ensure_directories()
        return store

    def ensure_directories(self) -> None:
        self.layout.ensure_directories()

    @property
    def root(self) -> Path:
        return self.layout.root

    @property
    def analysis_dir(self) -> Path:
        return self.layout.analysis_dir

    @property
    def crops_dir(self) -> Path:
        return self.layout.crops_dir

    @property
    def mmr_dir(self) -> Path:
        return self.layout.mmr_dir

    @property
    def mmr_cache_dir(self) -> Path:
        return self.layout.mmr_cache_dir

    @property
    def mmr_batch_grids_dir(self) -> Path:
        return self.layout.mmr_batch_grids_dir

    @property
    def manifest_path(self) -> Path:
        return self.layout.manifest_path

    @property
    def frames_path(self) -> Path:
        return self.layout.frames_path

    @property
    def render_frames_path(self) -> Path:
        return self.layout.render_frames_path

    @property
    def tracks_path(self) -> Path:
        return self.layout.tracks_path

    @property
    def detection_stats_path(self) -> Path:
        return self.layout.detection_stats_path

    @property
    def count_events_path(self) -> Path:
        return self.layout.count_events_path

    @property
    def labels_path(self) -> Path:
        return self.layout.labels_path

    @property
    def output_video_path(self) -> Path:
        return self.layout.output_video_path

    @property
    def report_csv_path(self) -> Path:
        return self.layout.report_csv_path
