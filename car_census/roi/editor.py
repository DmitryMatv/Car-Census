from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from car_census.config import CameraProfile, CountLineConfig, PolygonZoneConfig
from car_census.utils.video import read_first_frame


WINDOW_NAME = "car-census ROI editor"


def edit_camera_profile(video_path: Path, camera_id: str, output_path: Path) -> CameraProfile:
    frame = read_first_frame(video_path)
    polygon: list[list[int]] = []
    line: list[list[int]] = []
    mode = {"value": "polygon"}

    def on_mouse(event: int, x: int, y: int, *_args: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if mode["value"] == "polygon":
            polygon.append([x, y])
        elif len(line) < 2:
            line.append([x, y])

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    while True:
        canvas = frame.copy()
        if polygon:
            for point in polygon:
                cv2.circle(canvas, tuple(point), 5, (0, 255, 0), -1)
            if len(polygon) >= 2:
                cv2.polylines(canvas, [np.array(polygon, dtype=np.int32)], False, (0, 255, 0), 2)
        if len(line) == 1:
            cv2.circle(canvas, tuple(line[0]), 5, (0, 165, 255), -1)
        elif len(line) == 2:
            cv2.line(canvas, tuple(line[0]), tuple(line[1]), (0, 165, 255), 2)
        status = (
            "Polygon mode: left click points, press N when done"
            if mode["value"] == "polygon"
            else "Line mode: click two points, press S to save"
        )
        cv2.putText(canvas, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            cv2.destroyWindow(WINDOW_NAME)
            raise RuntimeError("ROI editing cancelled by user")
        if key == ord("u"):
            if mode["value"] == "polygon" and polygon:
                polygon.pop()
            elif mode["value"] == "line" and line:
                line.pop()
        if key == ord("n") and len(polygon) >= 3:
            mode["value"] = "line"
        if key == ord("s") and len(polygon) >= 3 and len(line) == 2:
            break

    cv2.destroyWindow(WINDOW_NAME)
    profile = CameraProfile(
        camera_id=camera_id,
        polygon=PolygonZoneConfig(points=polygon),
        count_line=CountLineConfig(start=line[0], end=line[1], direction="BOTH"),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(profile.model_dump(mode="json"), handle, sort_keys=False)
    return profile
