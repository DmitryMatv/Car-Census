from __future__ import annotations

import csv
import io
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from importlib.resources import files
from pathlib import Path
from typing import IO

from models import MMRResult


class PowertrainClass(StrEnum):
    BEV = "BEV"
    COMBUSTION = "COMBUSTION"
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class VehicleIdentity:
    make: str
    model: str
    generation: str
    variation: str = ""


PowertrainCatalog = dict[VehicleIdentity, PowertrainClass]
_DATA_PACKAGE = "mmr.data"
_POWERTRAIN_RESOURCE = "MakeModelGenVar.csv"


def _read_powertrain_catalog(handle: IO[str], source: object) -> PowertrainCatalog:
    reader = csv.DictReader(handle)
    try:
        return {
            VehicleIdentity(
                make=row["make"],
                model=row["model"],
                generation=row["generation"],
                variation=row["variation"],
            ): PowertrainClass(row["powertrain_class"])
            for row in reader
        }
    except KeyError as e:
        raise ValueError(f"Missing required column in {source}: {e}") from e
    except ValueError as e:
        raise ValueError(f"Invalid powertrain_class value in {source}: {e}") from e


def load_powertrain_catalog(path: Path) -> PowertrainCatalog:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return _read_powertrain_catalog(handle, path)


def load_default_powertrain_catalog() -> PowertrainCatalog:
    resource = files(_DATA_PACKAGE).joinpath(_POWERTRAIN_RESOURCE)
    if not resource.is_file():
        raise FileNotFoundError(
            f"Packaged powertrain catalog not found: {_POWERTRAIN_RESOURCE}"
        )
    with resource.open("rb") as raw:
        handle = io.TextIOWrapper(raw, encoding="utf-8", newline="")
        return _read_powertrain_catalog(handle, _POWERTRAIN_RESOURCE)


def lookup_powertrain_class(
    catalog: Mapping[VehicleIdentity, PowertrainClass], result: MMRResult
) -> PowertrainClass | None:
    if result.make is None or result.model is None or result.generation is None:
        return None
    identity = VehicleIdentity(
        make=result.make,
        model=result.model,
        generation=result.generation,
        variation=result.variation or "",
    )
    return catalog.get(identity)
