from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import BinaryIO, Generic, TypeVar

import orjson
from pydantic import BaseModel

from models import FrameRecord, MMRResult
from storage.json_artifacts import iter_jsonl, read_json, write_json, write_jsonl
from storage.run_layout import RunLayout

TModel = TypeVar("TModel", bound=BaseModel)


class JsonModelFile(Generic[TModel]):
    def __init__(self, path: Path, model_type: type[TModel]) -> None:
        self._path = path
        self._model_type = model_type

    def write(self, model: TModel) -> None:
        write_json(self._path, model.model_dump(mode="json"))

    def read(self) -> TModel:
        return self._model_type.model_validate(read_json(self._path))


class JsonlModelWriter(Generic[TModel]):
    def __init__(self, path: Path) -> None:
        self._path = path
        self._handle: BinaryIO | None = None

    def __enter__(self) -> "JsonlModelWriter[TModel]":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("wb")
        return self

    def __exit__(self, *args: object) -> None:
        if self._handle is not None:
            self._handle.close()

    def write(self, model: TModel) -> None:
        if self._handle is None:
            raise RuntimeError("JSONL writer is not open")
        self._handle.write(orjson.dumps(model.model_dump(mode="json")))
        self._handle.write(b"\n")


class JsonlModelFile(Generic[TModel]):
    def __init__(self, path: Path, model_type: type[TModel]) -> None:
        self._path = path
        self._model_type = model_type

    def append(self, model: TModel) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("ab") as handle:
            handle.write(orjson.dumps(model.model_dump(mode="json")))
            handle.write(b"\n")

    def iter(self) -> Iterator[TModel]:
        yield from iter_jsonl(self._path, self._model_type)

    def read_all(self) -> list[TModel]:
        return list(self.iter())

    def write_all(self, records: Iterable[TModel]) -> None:
        write_jsonl(self._path, (record.model_dump(mode="json") for record in records))

    def open_writer(self) -> JsonlModelWriter[TModel]:
        return JsonlModelWriter(self._path)


class FrameRecordsFile:
    def __init__(self, layout: RunLayout) -> None:
        self._layout = layout

    def _path(self, *, smoothed: bool) -> Path:
        return self._layout.render_frames_path if smoothed else self._layout.frames_path

    def append(self, record: FrameRecord) -> None:
        JsonlModelFile(self._layout.frames_path, FrameRecord).append(record)

    def iter(self, *, smoothed: bool = False) -> Iterator[FrameRecord]:
        yield from iter_jsonl(self._path(smoothed=smoothed), FrameRecord)

    def read_all(self, *, smoothed: bool = False) -> list[FrameRecord]:
        return list(self.iter(smoothed=smoothed))

    def write_all(
        self, records: Iterable[FrameRecord], *, smoothed: bool = False
    ) -> None:
        write_jsonl(
            self._path(smoothed=smoothed),
            (record.model_dump(mode="json") for record in records),
        )

    def open_writer(self, *, smoothed: bool = False) -> JsonlModelWriter[FrameRecord]:
        return JsonlModelWriter(self._path(smoothed=smoothed))

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
                            "vehicle_index": vehicle_index_by_track.get(
                                track.track_id, track.vehicle_index
                            )
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


class LabelsFile:
    def __init__(self, path: Path) -> None:
        self._path = path

    def write(self, labels_by_track: dict[int, MMRResult]) -> None:
        write_json(
            self._path,
            {
                str(track_id): result.model_dump(mode="json")
                for track_id, result in labels_by_track.items()
            },
        )

    def read(self) -> dict[int, MMRResult]:
        if not self._path.exists():
            return {}
        raw = read_json(self._path)
        if not isinstance(raw, dict):
            return {}
        return {
            int(track_id): MMRResult.model_validate(payload)
            for track_id, payload in raw.items()
        }


class DetectionStatsFile:
    def __init__(self, path: Path) -> None:
        self._path = path

    def write(self, payload: Mapping[str, object]) -> None:
        write_json(self._path, dict(payload))

    def read(self) -> dict[str, object]:
        if not self._path.exists():
            return {}
        raw = read_json(self._path)
        return raw if isinstance(raw, dict) else {}
