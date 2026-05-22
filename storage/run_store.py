from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

import orjson
from pydantic import BaseModel

from models import CountEvent, FrameRecord, MMRResult, RunManifest, TrackSummary

TModel = TypeVar("TModel", bound=BaseModel)


class RunStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.analysis_dir = root / "analysis"
        self.crops_dir = root / "crops"
        self.mmr_dir = root / "mmr"
        self.mmr_cache_dir = self.mmr_dir / "cache"
        self.mmr_batch_grids_dir = self.mmr_dir / "batch_grids"

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
            self.mmr_cache_dir,
            self.mmr_batch_grids_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

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

    def _write_json(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))

    def _append_jsonl(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as handle:
            handle.write(orjson.dumps(payload))
            handle.write(b"\n")

    def _iter_jsonl(self, path: Path, model_type: type[TModel]) -> Iterator[TModel]:
        if not path.exists():
            return
        with path.open("rb") as handle:
            for line in handle:
                if line.strip():
                    yield model_type.model_validate(orjson.loads(line))

    def _write_jsonl(self, path: Path, payloads: Iterable[object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            for payload in payloads:
                handle.write(orjson.dumps(payload))
                handle.write(b"\n")

    def write_manifest(self, manifest: RunManifest) -> None:
        self._write_json(self.manifest_path, manifest.model_dump(mode="json"))

    def read_manifest(self) -> RunManifest:
        return RunManifest.model_validate(orjson.loads(self.manifest_path.read_bytes()))

    def append_frame_record(self, record: FrameRecord) -> None:
        self._append_jsonl(self.frames_path, record.model_dump(mode="json"))

    def iter_frame_records(self, *, smoothed: bool = False) -> Iterator[FrameRecord]:
        path = self.render_frames_path if smoothed else self.frames_path
        yield from self._iter_jsonl(path, FrameRecord)

    def read_frame_records(self, *, smoothed: bool = False) -> list[FrameRecord]:
        return list(self.iter_frame_records(smoothed=smoothed))

    def write_frame_records(
        self, records: Iterable[FrameRecord], *, smoothed: bool = False
    ) -> None:
        path = self.render_frames_path if smoothed else self.frames_path
        self._write_jsonl(path, (record.model_dump(mode="json") for record in records))

    def rewrite_frame_vehicle_indices(
        self, vehicle_index_by_track: dict[int, int]
    ) -> None:
        temp_path = self.frames_path.with_suffix(f"{self.frames_path.suffix}.tmp")
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        with temp_path.open("wb") as target:
            for record in self.iter_frame_records(smoothed=False):
                tracks = [
                    track.model_copy(
                        update={
                            "vehicle_index": vehicle_index_by_track.get(track.track_id)
                        }
                    )
                    for track in record.tracks
                ]
                payload = record.model_copy(update={"tracks": tracks}).model_dump(
                    mode="json"
                )
                target.write(orjson.dumps(payload))
                target.write(b"\n")
        temp_path.replace(self.frames_path)

    def append_track_summary(self, summary: TrackSummary) -> None:
        self._append_jsonl(self.tracks_path, summary.model_dump(mode="json"))

    def iter_track_summaries(self) -> Iterator[TrackSummary]:
        yield from self._iter_jsonl(self.tracks_path, TrackSummary)

    def read_track_summaries(self) -> list[TrackSummary]:
        return list(self.iter_track_summaries())

    def append_count_event(self, event: CountEvent) -> None:
        self._append_jsonl(self.count_events_path, event.model_dump(mode="json"))

    def iter_count_events(self) -> Iterator[CountEvent]:
        yield from self._iter_jsonl(self.count_events_path, CountEvent)

    def read_count_events(self) -> list[CountEvent]:
        return list(self.iter_count_events())

    def write_labels(self, labels_by_track: dict[int, MMRResult]) -> None:
        self._write_json(
            self.labels_path,
            {
                str(track_id): result.model_dump(mode="json")
                for track_id, result in labels_by_track.items()
            },
        )

    def read_labels(self) -> dict[int, MMRResult]:
        if not self.labels_path.exists():
            return {}
        raw = orjson.loads(self.labels_path.read_bytes())
        return {
            int(track_id): MMRResult.model_validate(payload)
            for track_id, payload in raw.items()
        }

    def write_detection_stats(self, payload: Mapping[str, object]) -> None:
        self._write_json(self.detection_stats_path, dict(payload))

    def read_detection_stats(self) -> dict[str, object]:
        if not self.detection_stats_path.exists():
            return {}
        raw = orjson.loads(self.detection_stats_path.read_bytes())
        return raw if isinstance(raw, dict) else {}
