from __future__ import annotations

from typing import Sequence

import numpy as np
import supervision as sv

from car_census.config import AppConfig, CameraProfile
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
            position=sv.Position.CENTER,
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
        _ = profile
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
        annotated = self.label.annotate(
            scene=annotated, detections=detections, labels=label_text
        )
        return annotated
