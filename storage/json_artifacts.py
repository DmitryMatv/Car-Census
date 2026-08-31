from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, TypeVar

import orjson
from pydantic import BaseModel

TModel = TypeVar("TModel", bound=BaseModel)


@contextmanager
def atomic_write(path: Path) -> Iterator[BinaryIO]:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temp_path = Path(temp_name)
    handle = os.fdopen(file_descriptor, "wb")
    try:
        yield handle
        handle.flush()
        os.fsync(handle.fileno())
    except BaseException:
        handle.close()
        temp_path.unlink(missing_ok=True)
        raise
    handle.close()
    os.replace(temp_path, path)


def write_json(path: Path, payload: object) -> None:
    with atomic_write(path) as handle:
        handle.write(orjson.dumps(payload, option=orjson.OPT_INDENT_2))


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
    with atomic_write(path) as handle:
        for payload in payloads:
            handle.write(orjson.dumps(payload))
            handle.write(b"\n")
