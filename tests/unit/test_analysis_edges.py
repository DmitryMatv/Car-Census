from models import BBox
from pipeline.analysis_edges import track_matches_edge_detection


def _box(x1: float, y1: float, x2: float, y2: float) -> BBox:
    return BBox(x1=x1, y1=y1, x2=x2, y2=y2)


def test_substantial_duplicate_of_edge_detection_is_suppressed() -> None:
    track = _box(100.0, 100.0, 200.0, 150.0)
    duplicate = _box(105.0, 103.0, 198.0, 148.0)

    assert track_matches_edge_detection(track, [duplicate]) is True


def test_partial_overlap_with_edge_detection_is_kept() -> None:
    # Far-field vehicle brushing a static-structure misfire: overlap exists
    # but neither box is a duplicate of the other.
    track = _box(1157.0, 313.0, 1207.0, 346.0)
    misfire = _box(1169.0, 302.0, 1211.0, 318.0)
    assert track.iou(misfire) < 0.5

    assert track_matches_edge_detection(track, [misfire]) is False


def test_misfire_inside_vehicle_is_kept() -> None:
    # A small misfire fully inside a large vehicle box must not suppress it.
    vehicle = _box(500.0, 600.0, 1000.0, 900.0)
    misfire = _box(700.0, 700.0, 740.0, 730.0)

    assert track_matches_edge_detection(vehicle, [misfire]) is False


def test_disjoint_edge_detection_is_kept() -> None:
    track = _box(100.0, 100.0, 150.0, 140.0)
    detection = _box(900.0, 900.0, 960.0, 950.0)

    assert track_matches_edge_detection(track, [detection]) is False
