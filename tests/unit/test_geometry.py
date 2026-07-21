from models import BBox
from roi.geometry import (
    bbox_touches_frame_edge,
    bbox_touches_polygon_edge,
    bbox_touches_rect_edge,
    point_in_polygon,
)


def test_bbox_bottom_center_uses_lower_box_edge() -> None:
    bbox = BBox(x1=10, y1=20, x2=30, y2=80)

    assert bbox.bottom_center == (20.0, 80)
    assert bbox.center == (20.0, 50.0)


def test_bbox_intersection_method_detects_overlap_and_separation() -> None:
    bbox = BBox(x1=0, y1=0, x2=10, y2=10)

    assert bbox.intersection_area(BBox(x1=5, y1=5, x2=15, y2=12)) == 25.0
    assert bbox.intersection_area(BBox(x1=11, y1=5, x2=15, y2=12)) == 0.0


def test_box_iou_method_handles_overlap_and_zero_union() -> None:
    bbox = BBox(x1=0, y1=0, x2=10, y2=10)

    assert bbox.iou(BBox(x1=5, y1=0, x2=15, y2=10)) == 50.0 / 150.0
    assert BBox(x1=0, y1=0, x2=0, y2=0).iou(BBox(x1=1, y1=1, x2=1, y2=1)) == 0.0


def test_bbox_coverage_and_area_ratio() -> None:
    larger = BBox(x1=0, y1=0, x2=10, y2=10)
    smaller = BBox(x1=2, y1=2, x2=6, y2=7)

    assert larger.smaller_coverage(smaller) == 1.0
    assert larger.area_ratio(smaller) == 0.2
    assert larger.smaller_coverage(BBox(x1=0, y1=0, x2=0, y2=0)) == 0.0
    assert larger.area_ratio(BBox(x1=0, y1=0, x2=0, y2=0)) == 0.0


def test_bbox_center_distance() -> None:
    assert (
        BBox(x1=0, y1=0, x2=2, y2=2).center_distance(BBox(x1=6, y1=8, x2=8, y2=10))
        == 10.0
    )


def test_box_contains_point_method_inclusive() -> None:
    bbox = BBox(x1=10, y1=20, x2=30, y2=40)

    assert bbox.contains_point((10, 20))
    assert bbox.contains_point((30, 40))
    assert not bbox.contains_point((30.1, 40))


def test_box_contains_point_method_exclusive_right_bottom_edges() -> None:
    bbox = BBox(x1=10, y1=20, x2=30, y2=40)

    assert bbox.contains_point((10, 20), inclusive=False)
    assert not bbox.contains_point((30, 40), inclusive=False)
    assert not bbox.contains_point((30, 30), inclusive=False)
    assert not bbox.contains_point((20, 40), inclusive=False)


def test_point_in_polygon_detects_inside_point() -> None:
    polygon = [[0, 0], [100, 0], [100, 100], [0, 100]]
    assert point_in_polygon((50, 50), polygon) is True
    assert point_in_polygon((150, 50), polygon) is False


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


def test_bbox_touches_frame_edge_detects_invalid_and_outside_boxes() -> None:
    assert bbox_touches_frame_edge(BBox(x1=-20, y1=10, x2=-1, y2=30), (100, 100, 3))
    assert bbox_touches_frame_edge(BBox(x1=101, y1=10, x2=120, y2=30), (100, 100, 3))
    assert bbox_touches_frame_edge(BBox(x1=40, y1=10, x2=40, y2=30), (100, 100, 3))


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


def test_bbox_touches_polygon_edge_detects_box_already_outside_polygon() -> None:
    polygon = [[10, 10], [90, 30], [90, 90], [10, 90]]

    assert bbox_touches_polygon_edge(BBox(x1=45, y1=2, x2=55, y2=8), polygon)
