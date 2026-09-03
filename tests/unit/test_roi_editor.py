from pathlib import Path

import yaml

from config import CameraProfile, CountLineConfig, HomographyConfig, PolygonZoneConfig
from roi.editor import (
    MIN_HOMOGRAPHY_POINTS,
    EditorState,
    build_profile,
    load_existing_profile,
)


def _state(
    polygon: list[list[int]] | None = None,
    sources: list[list[float]] | None = None,
    targets: list[list[float]] | None = None,
) -> EditorState:
    return EditorState(
        polygon=polygon if polygon is not None else [[0, 0], [100, 0], [100, 100]],
        homography_sources=sources,
        homography_targets=targets,
    )


def test_find_point_hits_polygon_and_homography() -> None:
    state = _state(
        sources=[[50.0, 50.0], [200.0, 200.0]],
        targets=[[0.0, 0.0], [3.75, 0.0]],
    )
    assert state.find_point(52, 51) == ("homography", 0)
    assert state.find_point(0, 0) == ("polygon", 0)
    assert state.find_point(500, 500) is None


def test_set_point_moves_polygon_and_homography_independently() -> None:
    state = _state(
        sources=[[50.0, 50.0]],
        targets=[[1.0, 2.0]],
    )
    state.set_point(("polygon", 0), 10, 20)
    state.set_point(("homography", 0), 60, 70)
    assert state.polygon[0] == [10, 20]
    assert state.homography_sources == [[60.0, 70.0]]
    assert state.homography_targets == [[1.0, 2.0]]


def test_delete_homography_blocked_at_minimum_pair_count() -> None:
    sources = [[float(i), float(i)] for i in range(MIN_HOMOGRAPHY_POINTS)]
    targets = [[0.0, float(i)] for i in range(MIN_HOMOGRAPHY_POINTS)]
    state = _state(sources=sources, targets=targets)
    assert state.delete_point(("homography", 0)) is False
    assert state.homography_pair_count() == MIN_HOMOGRAPHY_POINTS


def test_delete_homography_removes_source_and_target_in_sync() -> None:
    sources = [[float(i), float(i)] for i in range(MIN_HOMOGRAPHY_POINTS + 1)]
    targets = [[0.0, float(i)] for i in range(MIN_HOMOGRAPHY_POINTS + 1)]
    state = _state(sources=sources, targets=targets)
    assert state.delete_point(("homography", 1)) is True
    assert state.homography_sources == [
        [0.0, 0.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 4.0],
    ]
    assert state.homography_targets == [
        [0.0, 0.0],
        [0.0, 2.0],
        [0.0, 3.0],
        [0.0, 4.0],
    ]


def test_delete_polygon_point() -> None:
    state = _state()
    assert state.delete_point(("polygon", 1)) is True
    assert state.polygon == [[0, 0], [100, 100]]


def test_undo_follows_active_mode() -> None:
    state = _state(sources=[], targets=[])
    state.add_homography_mode = True
    state.add_homography_pair(10, 20, 1.5, 2.5)
    assert state.undo() is True
    assert state.homography_pair_count() == 0
    state.add_homography_mode = False
    assert state.undo() is True
    assert len(state.polygon) == 2


def test_save_error_reports_polygon_and_homography_minimums() -> None:
    assert _state(polygon=[[0, 0], [1, 1]]).save_error() is not None
    state = _state(sources=[[0.0, 0.0]], targets=[[0.0, 1.0]])
    assert "homography" in (state.save_error() or "")
    assert _state().save_error() is None


def test_build_profile_omits_empty_homography() -> None:
    profile = build_profile("cam", [[0, 0], [1, 1], [2, 2]], [], [], None)
    assert profile.homography is None
    profile = build_profile(
        "cam",
        [[0, 0], [1, 1], [2, 2]],
        [[float(i), 0.0] for i in range(MIN_HOMOGRAPHY_POINTS)],
        [[0.0, float(i)] for i in range(MIN_HOMOGRAPHY_POINTS)],
        None,
    )
    assert profile.homography is not None


def test_save_and_load_round_trip_preserves_all_blocks(tmp_path: Path) -> None:
    profile = CameraProfile(
        camera_id="cam",
        polygon=PolygonZoneConfig(points=[[0, 0], [100, 0], [100, 100]]),
        count_line=CountLineConfig(start=[0, 0], end=[10, 10]),
        homography=HomographyConfig(
            source_points=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            target_points=[[0.0, 0.0], [3.75, 0.0], [0.0, 3.0], [3.75, 3.0]],
        ),
    )
    path = tmp_path / "cam.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(profile.model_dump(mode="json"), handle, sort_keys=False)
    loaded = load_existing_profile(path)
    assert loaded == profile
