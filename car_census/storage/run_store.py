from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import orjson

from car_census.types import RunManifest


class RunStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.analysis_dir = root / "analysis"
        self.crops_dir = root / "crops"
        self.mmr_dir = root / "mmr"
        self.render_dir = root / "render"
        self.reports_dir = root / "reports"
        self.mmr_cache_dir = self.mmr_dir / "cache"

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
        for path in [
            self.root,
            self.analysis_dir,
            self.crops_dir,
            self.mmr_dir,
            self.render_dir,
            self.reports_dir,
            self.mmr_cache_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    @property
    def manifest_path(self) -> Path:
        return self.root / "run.json"

    @property
    def frames_path(self) -> Path:
        return self.analysis_dir / "frames.jsonl"

    @property
    def tracks_path(self) -> Path:
        return self.analysis_dir / "tracks.jsonl"

    @property
    def count_events_path(self) -> Path:
        return self.analysis_dir / "count_events.jsonl"

    @property
    def labels_path(self) -> Path:
        return self.mmr_dir / "labels.json"

    @property
    def output_video_path(self) -> Path:
        return self.render_dir / "annotated.mp4"

    @property
    def counts_json_path(self) -> Path:
        return self.reports_dir / "counts.json"

    @property
    def counts_csv_path(self) -> Path:
        return self.reports_dir / "counts.csv"

    def write_json(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))

    def write_jsonl(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as handle:
            handle.write(orjson.dumps(payload))
            handle.write(b"\n")

    def write_manifest(self, manifest: RunManifest) -> None:
        self.write_json(self.manifest_path, manifest.model_dump(mode="json"))

    def read_manifest(self) -> RunManifest:
        return RunManifest.model_validate(orjson.loads(self.manifest_path.read_bytes()))
