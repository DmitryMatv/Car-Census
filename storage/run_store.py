from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from models import CountEvent, RunManifest, TrackSummary
from config import validate_camera_id
from storage.run_artifacts import (
    DetectionStatsFile,
    FrameRecordsFile,
    JsonlModelFile,
    JsonModelFile,
    LabelsFile,
)
from storage.run_layout import RunLayout


def _run_descriptor(camera_id: str, video_stem: str) -> str:
    if camera_id == "__full_frame__" or camera_id == video_stem:
        return video_stem
    return f"{video_stem}--camera-{validate_camera_id(camera_id)}"


def _compact_utc_timestamp(timestamp: datetime | None = None) -> str:
    current = timestamp or datetime.now(UTC)
    return current.astimezone(UTC).strftime("%Y%m%d-%H%M%SZ")


def _allocate_run_root(output_root: Path, base_run_id: str) -> Path:
    collision_number = 1
    while True:
        run_id = (
            base_run_id
            if collision_number == 1
            else f"{base_run_id}--{collision_number:02d}"
        )
        run_root = output_root / run_id
        try:
            run_root.mkdir(exist_ok=False)
        except FileExistsError:
            collision_number += 1
            continue
        return run_root


class RunStore:
    def __init__(self, root: Path | RunLayout) -> None:
        self.layout = root if isinstance(root, RunLayout) else RunLayout(root)
        self.manifest = JsonModelFile(self.layout.manifest_path, RunManifest)
        self.frames = FrameRecordsFile(self.layout)
        self.tracks = JsonlModelFile(self.layout.tracks_path, TrackSummary)
        self.counts = JsonlModelFile(self.layout.count_events_path, CountEvent)
        self.labels = LabelsFile(self.layout.labels_path)
        self.detection_stats = DetectionStatsFile(self.layout.detection_stats_path)

    @classmethod
    def create(
        cls,
        output_root: Path,
        camera_id: str,
        video_stem: str,
    ) -> "RunStore":
        output_root.mkdir(parents=True, exist_ok=True)
        descriptor = _run_descriptor(camera_id, video_stem)
        timestamp = _compact_utc_timestamp()
        run_root = _allocate_run_root(output_root, f"{descriptor}--{timestamp}")
        store = cls(run_root)
        store.ensure_directories()
        return store

    @classmethod
    def from_existing(cls, run_dir: Path) -> "RunStore":
        if run_dir.is_symlink():
            raise ValueError(f"Run directory must not be a symlink: {run_dir}")
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
        if not run_dir.is_dir():
            raise NotADirectoryError(f"Run path is not a directory: {run_dir}")
        store = cls(run_dir.resolve())
        if not store.manifest_path.is_file():
            raise FileNotFoundError(
                f"Run manifest does not exist: {store.manifest_path}"
            )
        store.manifest.read()
        return store

    def validate_analysis_artifacts(self) -> None:
        self.manifest.read()
        required_paths = [
            self.frames_path,
            self.tracks_path,
            self.detection_stats_path,
        ]
        missing = [path for path in required_paths if not path.is_file()]
        if missing:
            missing_paths = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(
                f"Analysis did not produce required artifacts: {missing_paths}"
            )

    def rebase_analysis_paths(self, destination_root: Path) -> None:
        source_root = self.root.resolve()
        destination_root = destination_root.resolve()
        manifest = self.manifest.read().model_copy(
            update={
                "run_id": destination_root.name,
                "root_dir": destination_root,
            }
        )
        self.manifest.write(manifest)

        summaries: list[TrackSummary] = []
        for summary in self.tracks.iter():
            candidates = []
            for candidate in summary.candidates:
                candidate_path = candidate.image_path.resolve()
                try:
                    relative_path = candidate_path.relative_to(source_root)
                except ValueError as exc:
                    raise ValueError(
                        "Analysis crop path is outside the staging run directory: "
                        f"{candidate.image_path}"
                    ) from exc
                candidates.append(
                    candidate.model_copy(
                        update={"image_path": destination_root / relative_path}
                    )
                )
            summaries.append(summary.model_copy(update={"candidates": candidates}))
        self.tracks.write_all(summaries)

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
