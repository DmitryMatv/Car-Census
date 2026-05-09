from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
import supervision as sv

from config import AppConfig, CameraProfile
from models import TrackedObject


def _color_to_bgr(color_hex: str) -> tuple[int, int, int]:
    return sv.Color.from_hex(color_hex).as_bgr()


def label_box_bounds(
    frame_shape: tuple[int, ...],
    track: TrackedObject,
    label: str,
    config: AppConfig,
) -> tuple[int, int, int, int, int]:
    text_size, baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        config.render.label_font_scale,
        config.render.label_thickness,
    )
    text_width, text_height = text_size
    padding = config.render.label_padding_px
    label_width = text_width + (padding * 2)
    label_height = text_height + baseline + (padding * 2)
    frame_height, frame_width = frame_shape[:2]
    bbox_center_x = (track.bbox.x1 + track.bbox.x2) / 2.0

    left = int(round(bbox_center_x - (label_width / 2.0)))
    left = max(0, min(left, max(0, frame_width - label_width)))
    top = int(round(track.bbox.y2 + config.render.label_gap_px))
    top = max(0, min(top, max(0, frame_height - label_height)))

    return left, top, left + label_width, top + label_height, baseline


class VideoAnnotator:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.box_color = sv.Color.from_hex(config.render.box_color)
        self.label_bg_color = _color_to_bgr(config.render.box_color)
        self.label_text_color = _color_to_bgr(config.render.label_text_color)
        self.corner = sv.BoxCornerAnnotator(
            color=self.box_color,
            thickness=config.render.corner_thickness,
            corner_length=config.render.corner_length,
            color_lookup=sv.ColorLookup.INDEX,
        )
        self.trace = sv.TraceAnnotator(
            color=self.box_color,
            trace_length=config.render.trace_length,
            position=sv.Position.CENTER,
            thickness=config.render.line_thickness,
            color_lookup=sv.ColorLookup.INDEX,
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
        for track, label in zip(tracks, label_text, strict=True):
            self._draw_label(annotated, track, label)
        return annotated

    def _draw_label(self, frame: np.ndarray, track: TrackedObject, label: str) -> None:
        left, top, right, bottom, baseline = label_box_bounds(
            frame.shape, track, label, self.config
        )
        padding = self.config.render.label_padding_px
        cv2.rectangle(frame, (left, top), (right, bottom), self.label_bg_color, -1)
        cv2.putText(
            frame,
            label,
            (left + padding, bottom - padding - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.render.label_font_scale,
            self.label_text_color,
            self.config.render.label_thickness,
            cv2.LINE_AA,
        )
