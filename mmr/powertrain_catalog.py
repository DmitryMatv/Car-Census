from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

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


def load_powertrain_catalog(path: Path) -> PowertrainCatalog:
    with path.open("r", encoding="utf-8", newline="") as handle:
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
            raise ValueError(f"Missing required column in {path}: {e}") from e
        except ValueError as e:
            raise ValueError(f"Invalid powertrain_class value in {path}: {e}") from e


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
