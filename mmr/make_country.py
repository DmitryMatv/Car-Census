from __future__ import annotations

import csv
from pathlib import Path

MakeCountryCatalog = dict[str, str]


def load_make_country_catalog(path: Path) -> MakeCountryCatalog:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        try:
            return {row["make"]: row["origin_country"] for row in reader}
        except KeyError as e:
            raise ValueError(f"Missing required column in {path}: {e}") from e
