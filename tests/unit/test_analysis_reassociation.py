from config import AppConfig
from models import BBox
from pipeline.analysis_diagnostics import AnalysisDiagnostics
from pipeline.analysis_reassociation import StaleReassociationRejector
from pipeline.analysis_tracking import TrackObservation
from roi.transform import ViewTransformer


def _observation(
    track_id: int,
    bottom_center: tuple[float, float] = (20.0, 40.0),
) -> TrackObservation:
    x, y = bottom_center
    bbox = BBox(x1=x - 10, y1=y - 20, x2=x + 10, y2=y)
    return TrackObservation(
        index=0,
        track_id=track_id,
        bbox=bbox,
        confidence=0.9,
        class_id=2,
        class_name="car",
        bottom_center=bbox.bottom_center,
    )


def _rejector(
    config_factory,
    max_reassociation_gap_seconds: float | None = 0.5,
    tracker_overrides: dict[str, object] | None = None,
    view_transformer: ViewTransformer | None = None,
) -> tuple[StaleReassociationRejector, AnalysisDiagnostics]:
    diagnostics = AnalysisDiagnostics()
    overrides: dict[str, object] = {
        "max_reassociation_gap_seconds": max_reassociation_gap_seconds
    }
    if tracker_overrides:
        overrides.update(tracker_overrides)
    config: AppConfig = config_factory({"tracker": overrides})
    return (
        StaleReassociationRejector(
            tracker_config=config.tracker,
            diagnostics=diagnostics,
            view_transformer=view_transformer,
        ),
        diagnostics,
    )


def _identity_meter_transformer() -> ViewTransformer:
    corners = [[0, 0], [2000, 0], [2000, 1200], [0, 1200]]
    return ViewTransformer(corners, corners)


def test_stale_reassociation_rejector_accepts_continuous_and_exact_limit(
    config_factory,
) -> None:
    rejector, diagnostics = _rejector(config_factory)
    observation = _observation(7)

    first = rejector.reject([observation], 0.0)
    continuous = rejector.reject([observation], 0.1)
    exact_limit = rejector.reject([observation], 0.6)

    assert first.observations == [observation]
    assert continuous.observations == [observation]
    assert exact_limit.observations == [observation]
    assert exact_limit.dropped_track_ids == set()
    assert diagnostics.stale_reassociation_observations_suppressed == 0
    assert diagnostics.stale_reassociation_track_ids_dropped == 0


def test_stale_reassociation_rejector_retires_id_after_long_gap(config_factory) -> None:
    rejector, diagnostics = _rejector(config_factory)
    observation = _observation(7)

    rejector.reject([observation], 0.0)
    stale = rejector.reject([observation], 0.6)
    retired_return = rejector.reject([observation], 0.7)

    assert stale.observations == []
    assert stale.dropped_track_ids == {7}
    assert retired_return.observations == []
    assert retired_return.dropped_track_ids == {7}
    assert diagnostics.stale_reassociation_observations_suppressed == 2
    assert diagnostics.stale_reassociation_track_ids_dropped == 1


def test_stale_reassociation_rejector_can_be_disabled(config_factory) -> None:
    rejector, diagnostics = _rejector(config_factory, None)
    observation = _observation(7)

    rejector.reject([observation], 0.0)
    later = rejector.reject([observation], 10.0)

    assert later.observations == [observation]
    assert later.dropped_track_ids == set()
    assert diagnostics.stale_reassociation_observations_suppressed == 0
    assert diagnostics.stale_reassociation_track_ids_dropped == 0


def test_world_gate_accepts_physically_plausible_return(config_factory) -> None:
    transformer = _identity_meter_transformer()
    rejector, diagnostics = _rejector(config_factory, view_transformer=transformer)

    rejector.reject([_observation(7)], 0.0)
    result = rejector.reject([_observation(7, (40.0, 40.0))], 1.0)

    assert [obs.track_id for obs in result.observations] == [7]
    assert result.dropped_track_ids == set()
    assert diagnostics.world_reassociation_observations_accepted == 1
    assert diagnostics.world_reassociation_tracks_retained == 1
    assert diagnostics.stale_reassociation_track_ids_dropped == 0


def test_world_gate_rejects_implausible_teleport(config_factory) -> None:
    transformer = _identity_meter_transformer()
    rejector, diagnostics = _rejector(config_factory, view_transformer=transformer)

    rejector.reject([_observation(7)], 0.0)
    stale = rejector.reject([_observation(7, (400.0, 400.0))], 1.0)
    retired_return = rejector.reject([_observation(7)], 1.1)

    assert stale.observations == []
    assert stale.dropped_track_ids == {7}
    assert retired_return.dropped_track_ids == {7}
    assert diagnostics.world_reassociation_observations_accepted == 0
    assert diagnostics.stale_reassociation_track_ids_dropped == 1


def test_world_gate_respects_absolute_distance_cap(config_factory) -> None:
    transformer = _identity_meter_transformer()
    rejector, diagnostics = _rejector(
        config_factory,
        tracker_overrides={"world_reassociation_max_distance_m": 10.0},
        view_transformer=transformer,
    )

    rejector.reject([_observation(7)], 0.0)
    result = rejector.reject([_observation(7, (40.0, 40.0))], 1.0)

    assert result.observations == []
    assert result.dropped_track_ids == {7}
    assert diagnostics.world_reassociation_tracks_retained == 0


def test_world_gate_retires_gap_beyond_world_ceiling(config_factory) -> None:
    transformer = _identity_meter_transformer()
    rejector, diagnostics = _rejector(
        config_factory,
        view_transformer=transformer,
    )

    rejector.reject([_observation(7)], 0.0)
    result = rejector.reject([_observation(7, (25.0, 40.0))], 2.5)

    assert result.observations == []
    assert result.dropped_track_ids == {7}
    assert diagnostics.stale_reassociation_track_ids_dropped == 1


def test_world_gate_disabled_falls_back_to_legacy_retirement(
    config_factory,
) -> None:
    transformer = _identity_meter_transformer()
    rejector, diagnostics = _rejector(
        config_factory,
        tracker_overrides={"world_reassociation_enabled": False},
        view_transformer=transformer,
    )

    rejector.reject([_observation(7)], 0.0)
    result = rejector.reject([_observation(7, (25.0, 40.0))], 1.0)

    assert result.observations == []
    assert result.dropped_track_ids == {7}
    assert diagnostics.world_reassociation_observations_accepted == 0


def test_world_gate_without_transformer_keeps_legacy_behavior(
    config_factory,
) -> None:
    rejector, diagnostics = _rejector(config_factory)

    rejector.reject([_observation(7)], 0.0)
    result = rejector.reject([_observation(7, (21.0, 40.0))], 1.0)

    assert result.observations == []
    assert result.dropped_track_ids == {7}
    assert diagnostics.world_reassociation_observations_accepted == 0
