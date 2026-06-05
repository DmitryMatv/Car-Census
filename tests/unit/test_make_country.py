from __future__ import annotations

import csv
from pathlib import Path

from mmr.make_country import (
    load_default_make_country_catalog,
    load_make_country_catalog,
)


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["make", "origin_country"])
        writer.writerows(rows)


def test_load_make_country_catalog(tmp_path: Path) -> None:
    path = tmp_path / "make-country.csv"
    _write_csv(path, [["VW", "🇩🇪"], ["Volvo", "🇸🇪"]])

    assert load_make_country_catalog(path) == {
        "VW": "🇩🇪",
        "Volvo": "🇸🇪",
    }


def test_load_default_make_country_catalog_loads_packaged_resource() -> None:
    catalog = load_default_make_country_catalog()

    assert catalog
    assert all(isinstance(make, str) and make for make in catalog)
    assert all(isinstance(flag, str) and flag for flag in catalog.values())
