from car_census.roi.geometry import line_crossing_direction, point_in_polygon


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
