from __future__ import annotations

from typing import Sequence

import numpy as np
import supervision as sv

from config import AppConfig
from models import TrackedObject


def _tracks_to_detections(tracks: Sequence[TrackedObject]) -> sv.Detections:
    return sv.Detections(
        xyxy=np.array(
            [
                [track.bbox.x1, track.bbox.y1, track.bbox.x2, track.bbox.y2]
                for track in tracks
            ],
            dtype=np.float32,
        ),
        confidence=np.array([track.confidence for track in tracks], dtype=np.float32),
        class_id=np.array(
            [track.class_id if track.class_id is not None else 0 for track in tracks],
            dtype=np.int32,
        ),
        tracker_id=np.array([track.track_id for track in tracks], dtype=np.int32),
    )


class VideoAnnotator:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.corner_annotator = sv.BoxCornerAnnotator(
            color=sv.Color.from_hex(config.render.box_color),
            thickness=config.render.corner_thickness,
            corner_length=config.render.corner_length,
        )
        self.label_annotator = sv.LabelAnnotator(
            color=sv.Color.from_hex(config.render.label_bg_color),
            text_color=sv.Color.from_hex(config.render.label_text_color),
            text_scale=config.render.label_font_scale,
            text_thickness=config.render.label_thickness,
            text_padding=config.render.label_padding_px,
            text_position=sv.Position.TOP_LEFT,
            text_offset=(0, -config.render.label_gap_px),
        )

    def annotate(
        self,
        frame: np.ndarray,
        tracks: Sequence[TrackedObject],
        labels_by_track: dict[int, str],
    ) -> np.ndarray:
        annotated = frame.copy()
        if not tracks:
            return annotated

        detections = _tracks_to_detections(tracks)
        labels = [
            labels_by_track.get(track.track_id, self.config.render.unknown_label)
            for track in tracks
        ]
        annotated = self.corner_annotator.annotate(annotated, detections)
        annotated = self.label_annotator.annotate(
            annotated,  # type: ignore[arg-type]
            detections,
            labels=labels,
        )
        return annotated
