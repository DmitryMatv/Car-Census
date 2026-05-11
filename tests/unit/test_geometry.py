from roi.geometry import (
    bbox_touches_frame_edge,
    bbox_touches_polygon_edge,
    bbox_touches_rect_edge,
    line_crossing_direction,
    point_in_polygon,
)
from models import BBox


def test_point_in_polygon_detects_inside_point() -> None:
    polygon = [[0, 0], [100, 0], [100, 100], [0, 100]]
    assert point_in_polygon((50, 50), polygon) is True
    assert point_in_polygon((150, 50), polygon) is False


def test_line_crossing_direction_detects_a_to_b() -> None:
    direction = line_crossing_direction(
        previous_point=(10, 10),
        current_point=(10, -10),
        line_start=(0, 0),
        line_end=(100, 0),
    )
    assert direction == "B_TO_A"


def test_bbox_touches_frame_edge_detects_exact_edge() -> None:
    assert bbox_touches_frame_edge(BBox(x1=10, y1=5, x2=99, y2=50), (100, 100, 3))
    assert not bbox_touches_frame_edge(BBox(x1=10, y1=5, x2=98, y2=50), (100, 100, 3))


def test_bbox_touches_frame_edge_respects_margin() -> None:
    assert bbox_touches_frame_edge(
        BBox(x1=10, y1=10, x2=80, y2=80), (100, 100, 3), margin_px=10
    )
    assert not bbox_touches_frame_edge(
        BBox(x1=11, y1=11, x2=88, y2=88), (100, 100, 3), margin_px=10
    )


def test_bbox_touches_rect_edge_detects_offset_crop_edge() -> None:
    assert bbox_touches_rect_edge(
        BBox(x1=50, y1=25, x2=100, y2=75),
        left=50,
        top=20,
        right=150,
        bottom=120,
    )
    assert not bbox_touches_rect_edge(
        BBox(x1=51, y1=25, x2=100, y2=75),
        left=50,
        top=20,
        right=150,
        bottom=120,
    )


def test_bbox_touches_frame_edge_uses_rendered_pixel_coordinates() -> None:
    assert bbox_touches_frame_edge(BBox(x1=0.6, y1=5, x2=40, y2=30), (100, 100, 3))
    assert bbox_touches_frame_edge(BBox(x1=5, y1=5, x2=98.2, y2=30), (100, 100, 3))


def test_bbox_touches_polygon_edge_detects_slanted_boundary_intersection() -> None:
    polygon = [[10, 10], [90, 30], [90, 90], [10, 90]]

    assert bbox_touches_polygon_edge(BBox(x1=48, y1=18, x2=56, y2=28), polygon)
    assert not bbox_touches_rect_edge(
        BBox(x1=48, y1=18, x2=56, y2=28),
        left=10,
        top=10,
        right=91,
        bottom=91,
    )


def test_bbox_touches_polygon_edge_ignores_interior_box() -> None:
    polygon = [[10, 10], [90, 30], [90, 90], [10, 90]]

    assert not bbox_touches_polygon_edge(BBox(x1=50, y1=40, x2=60, y2=50), polygon)


def test_bbox_touches_polygon_edge_respects_margin() -> None:
    polygon = [[10, 10], [90, 30], [90, 90], [10, 90]]
    bbox = BBox(x1=50, y1=23, x2=58, y2=31)

    assert not bbox_touches_polygon_edge(bbox, polygon)
    assert bbox_touches_polygon_edge(bbox, polygon, margin_px=3)
