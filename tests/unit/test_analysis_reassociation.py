from config import AppConfig
from models import BBox
from pipeline.analysis_diagnostics import AnalysisDiagnostics
from pipeline.analysis_reassociation import StaleReassociationRejector
from pipeline.analysis_tracking import TrackObservation


def _observation(track_id: int) -> TrackObservation:
    bbox = BBox(x1=10, y1=20, x2=30, y2=40)
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
    max_reassociation_gap_seconds: float | None = 0.5,
) -> tuple[StaleReassociationRejector, AnalysisDiagnostics]:
    diagnostics = AnalysisDiagnostics()
    config = AppConfig.model_validate(
        {"tracker": {"max_reassociation_gap_seconds": max_reassociation_gap_seconds}}
    )
    return (
        StaleReassociationRejector(
            tracker_config=config.tracker,
            diagnostics=diagnostics,
        ),
        diagnostics,
    )


def test_stale_reassociation_rejector_accepts_continuous_and_exact_limit() -> None:
    rejector, diagnostics = _rejector()
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


def test_stale_reassociation_rejector_retires_id_after_long_gap() -> None:
    rejector, diagnostics = _rejector()
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


def test_stale_reassociation_rejector_can_be_disabled() -> None:
    rejector, diagnostics = _rejector(None)
    observation = _observation(7)

    rejector.reject([observation], 0.0)
    later = rejector.reject([observation], 10.0)

    assert later.observations == [observation]
    assert later.dropped_track_ids == set()
    assert diagnostics.stale_reassociation_observations_suppressed == 0
    assert diagnostics.stale_reassociation_track_ids_dropped == 0
