from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass

import orjson

from models import CountEvent, FrameRecord, MMRResult, RunManifest, TrackSummary
from storage.json_artifacts import (
    append_jsonl,
    iter_jsonl,
    read_json,
    write_json,
    write_jsonl,
)
from storage.run_layout import RunLayout


class ManifestRepository:
    def __init__(self, layout: RunLayout) -> None:
        self._layout = layout

    def write(self, manifest: RunManifest) -> None:
        write_json(self._layout.manifest_path, manifest.model_dump(mode="json"))

    def read(self) -> RunManifest:
        return RunManifest.model_validate(read_json(self._layout.manifest_path))


class FrameRecordRepository:
    def __init__(self, layout: RunLayout) -> None:
        self._layout = layout

    def append(self, record: FrameRecord) -> None:
        append_jsonl(self._layout.frames_path, record.model_dump(mode="json"))

    def iter(self, *, smoothed: bool = False) -> Iterator[FrameRecord]:
        path = self._layout.render_frames_path if smoothed else self._layout.frames_path
        yield from iter_jsonl(path, FrameRecord)

    def read_all(self, *, smoothed: bool = False) -> list[FrameRecord]:
        return list(self.iter(smoothed=smoothed))

    def write_all(
        self, records: Iterable[FrameRecord], *, smoothed: bool = False
    ) -> None:
        path = self._layout.render_frames_path if smoothed else self._layout.frames_path
        write_jsonl(path, (record.model_dump(mode="json") for record in records))

    def rewrite_vehicle_indices(self, vehicle_index_by_track: dict[int, int]) -> None:
        temp_path = self._layout.frames_path.with_suffix(
            f"{self._layout.frames_path.suffix}.tmp"
        )
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        with temp_path.open("wb") as target:
            for record in self.iter(smoothed=False):
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
        temp_path.replace(self._layout.frames_path)


class TrackSummaryRepository:
    def __init__(self, layout: RunLayout) -> None:
        self._layout = layout

    def append(self, summary: TrackSummary) -> None:
        append_jsonl(self._layout.tracks_path, summary.model_dump(mode="json"))

    def iter(self) -> Iterator[TrackSummary]:
        yield from iter_jsonl(self._layout.tracks_path, TrackSummary)

    def read_all(self) -> list[TrackSummary]:
        return list(self.iter())


class CountEventRepository:
    def __init__(self, layout: RunLayout) -> None:
        self._layout = layout

    def append(self, event: CountEvent) -> None:
        append_jsonl(self._layout.count_events_path, event.model_dump(mode="json"))

    def iter(self) -> Iterator[CountEvent]:
        yield from iter_jsonl(self._layout.count_events_path, CountEvent)

    def read_all(self) -> list[CountEvent]:
        return list(self.iter())


class LabelRepository:
    def __init__(self, layout: RunLayout) -> None:
        self._layout = layout

    def write(self, labels_by_track: dict[int, MMRResult]) -> None:
        write_json(
            self._layout.labels_path,
            {
                str(track_id): result.model_dump(mode="json")
                for track_id, result in labels_by_track.items()
            },
        )

    def read(self) -> dict[int, MMRResult]:
        if not self._layout.labels_path.exists():
            return {}
        raw = read_json(self._layout.labels_path)
        if not isinstance(raw, dict):
            return {}
        return {
            int(track_id): MMRResult.model_validate(payload)
            for track_id, payload in raw.items()
        }


class DetectionStatsRepository:
    def __init__(self, layout: RunLayout) -> None:
        self._layout = layout

    def write(self, payload: Mapping[str, object]) -> None:
        write_json(self._layout.detection_stats_path, dict(payload))

    def read(self) -> dict[str, object]:
        if not self._layout.detection_stats_path.exists():
            return {}
        raw = read_json(self._layout.detection_stats_path)
        return raw if isinstance(raw, dict) else {}


@dataclass(frozen=True, slots=True)
class RunRepositories:
    manifest: ManifestRepository
    frames: FrameRecordRepository
    tracks: TrackSummaryRepository
    counts: CountEventRepository
    labels: LabelRepository
    detection_stats: DetectionStatsRepository

    @classmethod
    def create(cls, layout: RunLayout) -> "RunRepositories":
        return cls(
            manifest=ManifestRepository(layout),
            frames=FrameRecordRepository(layout),
            tracks=TrackSummaryRepository(layout),
            counts=CountEventRepository(layout),
            labels=LabelRepository(layout),
            detection_stats=DetectionStatsRepository(layout),
        )
