import csv
from pathlib import Path

from mmr.powertrain_catalog import (
    PowertrainClass,
    VehicleIdentity,
    load_default_powertrain_catalog,
    load_powertrain_catalog,
    lookup_powertrain_class,
)
from models import MMRResult


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["make", "model", "generation", "variation", "powertrain_class"]
        )
        writer.writerows(rows)


def _catalog_row(
    powertrain_class: str = "BEV",
    *,
    make: str = "Test",
    model: str = "Car",
    generation: str = "Mk I",
    variation: str = "",
) -> list[str]:
    return [make, model, generation, variation, powertrain_class]


def test_load_and_lookup_require_exact_complete_identity(tmp_path: Path) -> None:
    path = tmp_path / "catalog.csv"
    _write_csv(
        path,
        [
            _catalog_row("BEV"),
            _catalog_row(
                "UNKNOWN",
                model="Fuel Cell",
                generation="Mk I",
            ),
        ],
    )

    catalog = load_powertrain_catalog(path)

    assert (
        lookup_powertrain_class(
            catalog,
            MMRResult(make="Test", model="Car", generation="Mk I"),
        )
        == PowertrainClass.BEV
    )
    assert (
        lookup_powertrain_class(
            catalog,
            MMRResult(make="Test", model="Fuel Cell", generation="Mk I"),
        )
        == PowertrainClass.UNKNOWN
    )
    assert (
        lookup_powertrain_class(
            catalog,
            MMRResult(make="test", model="Car", generation="Mk I"),
        )
        is None
    )
    assert (
        lookup_powertrain_class(
            catalog,
            MMRResult(make="Test", model="Car"),
        )
        is None
    )


def test_load_default_powertrain_catalog_loads_packaged_resource() -> None:
    catalog = load_default_powertrain_catalog()

    assert catalog
    assert all(isinstance(identity, VehicleIdentity) for identity in catalog)
    assert all(isinstance(value, PowertrainClass) for value in catalog.values())
