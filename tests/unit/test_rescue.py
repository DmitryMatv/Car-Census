import numpy as np
import pytest
from config import RescueConfig
from reid import TrackAppearanceMemory
from roi.transform import ViewTransformer
from tracking_adapters.rescue import RescueEngine, RescueMatch, RescueRejection

# 10 pixels map to 1 meter.
_TRANSFORMER = ViewTransformer(
    source_points=[[0, 0], [100, 0], [100, 100], [0, 100]],
    target_points=[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
)


def _config(**overrides: float) -> RescueConfig:
    values: dict[str, float] = {
        "max_gap_seconds": 2.0,
        "max_speed_mps": 40.0,
        "max_distance_m": 4.0,
        "lateral_tolerance_m": 2.0,
        "velocity_fit_points": 5,
        "min_direction_speed_mps": 1.0,
    }
    values.update(overrides)
    return RescueConfig(enabled=True, **values)  # type: ignore[arg-type]


def _engine(**config_overrides: float) -> RescueEngine:
    return RescueEngine(_config(**config_overrides), _TRANSFORMER)


def _observe_motion(engine: RescueEngine, track_id: int, speed_mps: float) -> None:
    # Car moving along +x: one analysis frame = 0.1 s. Bottom-center pixels:
    # world x meters * 10, y = 50 px (5 m lane center).
    for step in range(3):
        engine.observe(
            track_id,
            timestamp=step * 0.1,
            bottom_center=(50.0 + step * speed_mps, 50.0),
        )


def _candidate_bbox(
    world_x_m: float, world_y_m: float
) -> tuple[float, float, float, float]:
    """A 40x30 px box whose BOTTOM-CENTER sits at the given world point."""
    bottom_px_x = world_x_m * 10.0
    bottom_px_y = world_y_m * 10.0
    return (bottom_px_x - 20, bottom_px_y - 30, bottom_px_x + 20, bottom_px_y)


def test_engine_inactive_without_transformer() -> None:
    engine = RescueEngine(_config(), None)

    assert engine.active is False

    match, rejections = engine.match(_candidate_bbox(10.0, 5.0), 2.0, busy_ids=set())

    assert match is None
    assert rejections == []


def test_engine_inactive_when_disabled() -> None:
    engine = RescueEngine(_config(), _TRANSFORMER)
    engine.config = RescueConfig(
        enabled=False,
        max_gap_seconds=2.0,
        max_speed_mps=40.0,
        max_distance_m=4.0,
        lateral_tolerance_m=2.0,
        velocity_fit_points=5,
        min_direction_speed_mps=1.0,
    )

    assert engine.active is False


def test_accepts_plausible_handoff() -> None:
    engine = _engine()
    _observe_motion(engine, track_id=7, speed_mps=5.0)

    match, rejections = engine.match(_candidate_bbox(6.6, 5.0), 0.3, busy_ids=set())

    assert rejections == []
    assert match is not None
    assert match.old_track_id == 7
    assert match.gap_seconds == pytest.approx(0.1)
    assert match.distance_m <= 1.0
    assert match.implied_speed_mps == pytest.approx(6.0, abs=0.5)


def test_rejects_when_gap_exceeded() -> None:
    engine = _engine()
    _observe_motion(engine, track_id=7, speed_mps=5.0)

    match, rejections = engine.match(_candidate_bbox(6.6, 5.0), 5.0, busy_ids=set())

    assert match is None
    assert [r.reason for r in rejections] == ["gap_exceeded"]
    assert rejections[0].old_track_id == 7


def test_rejects_when_implied_speed_exceeded() -> None:
    engine = _engine()
    _observe_motion(engine, track_id=7, speed_mps=5.0)

    match, rejections = engine.match(_candidate_bbox(30.0, 5.0), 0.3, busy_ids=set())

    assert match is None
    assert [r.reason for r in rejections] == ["speed_exceeded"]


def test_rejects_when_lateral_exceeded() -> None:
    engine = _engine()
    _observe_motion(engine, track_id=7, speed_mps=5.0)

    match, rejections = engine.match(_candidate_bbox(6.6, 7.5), 0.3, busy_ids=set())

    assert match is None
    assert [r.reason for r in rejections] == ["lateral_exceeded"]
    assert rejections[0].lateral_m is not None
    assert rejections[0].lateral_m > 2.0


def test_skips_lateral_gate_when_direction_unreliable() -> None:
    engine = _engine(min_direction_speed_mps=10.0)
    _observe_motion(engine, track_id=7, speed_mps=0.05)

    match, _rejections = engine.match(_candidate_bbox(5.5, 6.0), 0.3, busy_ids=set())

    assert match is not None
    assert match.lateral_m is None


def test_busy_ids_are_never_takeover_sources() -> None:
    engine = _engine()
    _observe_motion(engine, track_id=7, speed_mps=5.0)

    match, rejections = engine.match(_candidate_bbox(6.6, 5.0), 0.3, busy_ids={7})

    assert match is None
    assert rejections == []


def test_insufficient_history_is_rejected() -> None:
    engine = _engine()
    engine.observe(7, timestamp=0.0, bottom_center=(50.0, 50.0))

    match, rejections = engine.match(_candidate_bbox(6.6, 5.0), 0.3, busy_ids=set())

    assert match is None
    assert [r.reason for r in rejections] == ["insufficient_history"]


def test_appearance_mismatch_rejects() -> None:
    engine = _engine()
    _observe_motion(engine, track_id=7, speed_mps=5.0)
    memory = TrackAppearanceMemory(history_length=4)
    memory.observe(7, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    candidate = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    match, rejections = engine.match(
        _candidate_bbox(6.6, 5.0),
        0.3,
        busy_ids=set(),
        memory=memory,
        candidate_vector=candidate,
        min_appearance_similarity=0.60,
    )

    assert match is None
    assert [r.reason for r in rejections] == ["appearance_mismatch"]
    assert rejections[0].appearance_similarity == pytest.approx(0.0)


def test_appearance_match_accepts() -> None:
    engine = _engine()
    _observe_motion(engine, track_id=7, speed_mps=5.0)
    memory = TrackAppearanceMemory(history_length=4)
    vector = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    memory.observe(7, vector)

    match, rejections = engine.match(
        _candidate_bbox(6.6, 5.0),
        0.3,
        busy_ids=set(),
        memory=memory,
        candidate_vector=vector,
        min_appearance_similarity=0.60,
    )

    assert rejections == []
    assert isinstance(match, RescueMatch)
    assert match.appearance_similarity == pytest.approx(1.0)


def test_appearance_missing_is_geometry_only() -> None:
    engine = _engine()
    _observe_motion(engine, track_id=7, speed_mps=5.0)

    match, rejections = engine.match(
        _candidate_bbox(6.6, 5.0),
        0.3,
        busy_ids=set(),
        memory=TrackAppearanceMemory(),
        candidate_vector=np.ones(3, dtype=np.float32),
        min_appearance_similarity=0.60,
    )

    assert rejections == []
    assert match is not None
    assert match.appearance_similarity is None


def test_best_candidate_wins_when_multiple_tracks_claim() -> None:
    engine = _engine()
    _observe_motion(engine, track_id=7, speed_mps=5.0)
    _observe_motion(engine, track_id=9, speed_mps=5.0)
    # Track 9 is 1 m behind track 7; the candidate sits at track 7's position.
    for point in engine.trajectories[9].points:
        point.x -= 10.0

    match, _rejections = engine.match(_candidate_bbox(6.6, 5.0), 0.3, busy_ids=set())

    assert match is not None
    assert match.old_track_id == 7


def test_non_finite_world_projection_is_ignored() -> None:
    engine = _engine()
    engine.observe(7, timestamp=0.0, bottom_center=(50.0, 50.0))
    engine.observe(7, timestamp=0.1, bottom_center=(60.0, 50.0))
    # A point projecting off the calibrated area yields inf and is dropped.
    engine.observe(7, timestamp=0.2, bottom_center=(float("nan"), 50.0))

    assert len(engine.trajectories[7].points) == 2


def test_rejection_carries_metrics_for_audit() -> None:
    engine = _engine()
    _observe_motion(engine, track_id=7, speed_mps=5.0)

    _match, rejections = engine.match(_candidate_bbox(30.0, 5.0), 0.3, busy_ids=set())

    assert len(rejections) == 1
    rejection: RescueRejection = rejections[0]
    assert rejection.gap_seconds == pytest.approx(0.1)
    assert rejection.distance_m == pytest.approx(23.5, abs=0.5)
    assert rejection.implied_speed_mps == pytest.approx(240.0, abs=1.0)
