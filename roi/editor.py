from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from config import CameraProfile, PolygonZoneConfig
from utils.video import read_first_frame

WINDOW_NAME = "Car Census ROI Editor"


def edit_camera_profile(
    video_path: Path, camera_id: str, output_path: Path
) -> CameraProfile:
    frame = read_first_frame(video_path)
    polygon: list[list[int]] = []

    def on_mouse(event: int, x: int, y: int, *_args: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        polygon.append([x, y])

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    while True:
        canvas = frame.copy()
        if polygon:
            for point in polygon:
                cv2.circle(canvas, tuple(point), 5, (0, 255, 0), -1)
            if len(polygon) >= 2:
                cv2.polylines(
                    canvas, [np.array(polygon, dtype=np.int32)], False, (0, 255, 0), 2
                )
        status = "Left click polygon points, press S to save"
        cv2.putText(
            canvas, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )
        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            cv2.destroyWindow(WINDOW_NAME)
            raise RuntimeError("ROI editing cancelled by user")
        if key == ord("u") and polygon:
            polygon.pop()
        if key == ord("s") and len(polygon) >= 3:
            break

    cv2.destroyWindow(WINDOW_NAME)
    profile = CameraProfile(
        camera_id=camera_id,
        polygon=PolygonZoneConfig(points=polygon),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(profile.model_dump(mode="json"), handle, sort_keys=False)
    return profile
