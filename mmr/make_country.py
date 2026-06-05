from __future__ import annotations

import csv
import io
from importlib.resources import files
from pathlib import Path
from typing import IO

MakeCountryCatalog = dict[str, str]
_DATA_PACKAGE = "mmr.data"
_MAKE_COUNTRY_RESOURCE = "MakeCountry.csv"


def _read_make_country_catalog(handle: IO[str], source: object) -> MakeCountryCatalog:
    reader = csv.DictReader(handle)
    try:
        return {row["make"]: row["origin_country"] for row in reader}
    except KeyError as e:
        raise ValueError(f"Missing required column in {source}: {e}") from e


def load_make_country_catalog(path: Path) -> MakeCountryCatalog:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return _read_make_country_catalog(handle, path)


def load_default_make_country_catalog() -> MakeCountryCatalog:
    resource = files(_DATA_PACKAGE).joinpath(_MAKE_COUNTRY_RESOURCE)
    if not resource.is_file():
        raise FileNotFoundError(
            f"Packaged make-country catalog not found: {_MAKE_COUNTRY_RESOURCE}"
        )
    with resource.open("rb") as raw:
        handle = io.TextIOWrapper(raw, encoding="utf-8", newline="")
        return _read_make_country_catalog(handle, _MAKE_COUNTRY_RESOURCE)
