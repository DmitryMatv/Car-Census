from car_census.stats.counts import aggregate_counts
from car_census.types import MMRResult


def test_aggregate_counts_groups_by_make_and_model() -> None:
    labels = {
        1: MMRResult(make="Audi", model="A4"),
        2: MMRResult(make="Audi", model="A4"),
        3: MMRResult(make="Toyota", model="Corolla"),
    }
    payload = aggregate_counts(labels, {1, 2, 3})
    assert payload["total_counted"] == 3
    assert payload["by_make"]["Audi"] == 2
    assert payload["by_make_model"]["Audi A4"] == 2
