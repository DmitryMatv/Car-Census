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


def test_aggregate_counts_dedupes_shared_vehicle_index() -> None:
    labels = {
        10: MMRResult(make="Audi", model="A4", vehicle_index=7),
        11: MMRResult(make="Audi", model="A4", vehicle_index=7),
        20: MMRResult(make="Toyota", model="Corolla", vehicle_index=8),
    }

    payload = aggregate_counts(labels, {10, 11, 20})

    assert payload["total_counted"] == 2
    assert payload["by_make"]["Audi"] == 1
    assert payload["by_make_model"]["Audi A4"] == 1


def test_aggregate_counts_ignores_counted_tracks_without_labels() -> None:
    labels = {
        2: MMRResult(make="Toyota", model="Corolla", vehicle_index=1),
    }

    payload = aggregate_counts(labels, {1, 2})

    assert payload["total_counted"] == 1
    assert payload["by_make_model"]["Toyota Corolla"] == 1
