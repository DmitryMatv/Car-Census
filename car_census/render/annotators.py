from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
import supervision as sv

from car_census.config import AppConfig, CameraProfile, FULL_FRAME_CAMERA_ID
from car_census.render.styles import COUNT_LINE_COLOR
from car_census.types import TrackedObject


class VideoAnnotator:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        # Tracks do not carry a reliable class_id during rendering, so color by track id.
        self.corner = sv.BoxCornerAnnotator(
            thickness=config.render.corner_thickness,
            color_lookup=sv.ColorLookup.TRACK,
        )
        self.label = sv.LabelAnnotator(
            text_scale=config.render.label_font_scale,
            text_thickness=config.render.label_thickness,
            color_lookup=sv.ColorLookup.TRACK,
        )
        self.trace = sv.TraceAnnotator(
            trace_length=config.render.trace_length,
            position=sv.Position.BOTTOM_CENTER,
            thickness=config.render.line_thickness,
            color_lookup=sv.ColorLookup.TRACK,
        )

    def annotate(
        self,
        frame: np.ndarray,
        profile: CameraProfile,
        tracks: Sequence[TrackedObject],
        labels_by_track: dict[int, str],
    ) -> np.ndarray:
        annotated = frame.copy()
        cv2.polylines(
            annotated,
            [np.array(profile.polygon.points, dtype=np.int32)],
            True,
            (80, 80, 80),
            self.config.render.line_thickness,
        )
        if profile.camera_id != FULL_FRAME_CAMERA_ID:
            cv2.line(
                annotated,
                tuple(profile.count_line.start),
                tuple(profile.count_line.end),
                COUNT_LINE_COLOR,
                self.config.render.line_thickness * 2,
            )
        if not tracks:
            return annotated

        detections = sv.Detections(
            xyxy=np.array(
                [[t.bbox.x1, t.bbox.y1, t.bbox.x2, t.bbox.y2] for t in tracks],
                dtype=np.float32,
            ),
            tracker_id=np.array([t.track_id for t in tracks], dtype=np.int32),
            data={},
        )
        label_text = [
            labels_by_track.get(track.track_id, self.config.render.unknown_label)
            for track in tracks
        ]
        annotated = self.trace.annotate(scene=annotated, detections=detections)
        annotated = self.corner.annotate(scene=annotated, detections=detections)
        annotated = self.label.annotate(scene=annotated, detections=detections, labels=label_text)
        return annotated
