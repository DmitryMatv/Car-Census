from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypeVar

import orjson
from pydantic import BaseModel

TModel = TypeVar("TModel", bound=BaseModel)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))


def read_json(path: Path) -> object:
    return orjson.loads(path.read_bytes())


def iter_jsonl(path: Path, model_type: type[TModel]) -> Iterator[TModel]:
    if not path.exists():
        return
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                yield model_type.model_validate(orjson.loads(line))


def write_jsonl(path: Path, payloads: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for payload in payloads:
            handle.write(orjson.dumps(payload))
            handle.write(b"\n")
