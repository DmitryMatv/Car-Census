from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
import supervision as sv
from supervision.annotators.core import spread_out_boxes

from config import AppConfig, CameraProfile
from models import TrackedObject


@dataclass(frozen=True)
class LabelLayout:
    left: int
    top: int
    right: int
    bottom: int
    baseline: int
    text_origin: tuple[int, int]


def _color_to_bgr(color_hex: str) -> tuple[int, int, int]:
    color = color_hex.removeprefix("#")
    if len(color) != 6:
        raise ValueError(f"Expected 6-digit hex color, got: {color_hex}")
    red = int(color[0:2], 16)
    green = int(color[2:4], 16)
    blue = int(color[4:6], 16)
    return blue, green, red


def _ensure_odd_kernel(radius_px: int) -> int:
    if radius_px <= 0:
        return 0
    kernel = (radius_px * 2) + 1
    return kernel if kernel % 2 == 1 else kernel + 1


def _blend_layer(base: np.ndarray, layer: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0.0:
        return base
    return cv2.addWeighted(base, 1.0, layer, alpha, 0.0)


def _clip_rect(
    left: int,
    top: int,
    right: int,
    bottom: int,
    frame_shape: tuple[int, ...],
) -> tuple[int, int, int, int] | None:
    frame_height, frame_width = frame_shape[:2]
    clipped_left = max(0, min(left, frame_width))
    clipped_top = max(0, min(top, frame_height))
    clipped_right = max(0, min(right, frame_width))
    clipped_bottom = max(0, min(bottom, frame_height))
    if clipped_left >= clipped_right or clipped_top >= clipped_bottom:
        return None
    return clipped_left, clipped_top, clipped_right, clipped_bottom


def _overlay_rect(
    frame: np.ndarray,
    left: int,
    top: int,
    right: int,
    bottom: int,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    clipped = _clip_rect(left, top, right, bottom, frame.shape)
    if clipped is None or alpha <= 0.0:
        return
    clipped_left, clipped_top, clipped_right, clipped_bottom = clipped
    roi = frame[clipped_top:clipped_bottom, clipped_left:clipped_right]
    color_roi = np.full_like(roi, color)
    roi[:] = cv2.addWeighted(roi, 1.0 - alpha, color_roi, alpha, 0.0)


def _overlay_text_mask_roi(
    frame: np.ndarray,
    mask: np.ndarray,
    left: int,
    top: int,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    if alpha <= 0.0 or mask.size == 0:
        return
    right = left + mask.shape[1]
    bottom = top + mask.shape[0]
    clipped = _clip_rect(left, top, right, bottom, frame.shape)
    if clipped is None:
        return

    clipped_left, clipped_top, clipped_right, clipped_bottom = clipped
    mask_left = clipped_left - left
    mask_top = clipped_top - top
    mask_right = mask_left + (clipped_right - clipped_left)
    mask_bottom = mask_top + (clipped_bottom - clipped_top)
    clipped_mask = mask[mask_top:mask_bottom, mask_left:mask_right]
    text_pixels = clipped_mask > 0
    if not np.any(text_pixels):
        return

    roi = frame[clipped_top:clipped_bottom, clipped_left:clipped_right]
    color_pixels = np.full_like(roi[text_pixels], color)
    roi[text_pixels] = cv2.addWeighted(
        roi[text_pixels], 1.0 - alpha, color_pixels, alpha, 0.0
    )


def _overlay_image_roi(
    frame: np.ndarray,
    image: np.ndarray,
    left: int,
    top: int,
    alpha: float,
) -> None:
    if alpha <= 0.0 or image.size == 0:
        return
    right = left + image.shape[1]
    bottom = top + image.shape[0]
    clipped = _clip_rect(left, top, right, bottom, frame.shape)
    if clipped is None:
        return

    clipped_left, clipped_top, clipped_right, clipped_bottom = clipped
    image_left = clipped_left - left
    image_top = clipped_top - top
    image_right = image_left + (clipped_right - clipped_left)
    image_bottom = image_top + (clipped_bottom - clipped_top)
    clipped_image = image[image_top:image_bottom, image_left:image_right]
    glow_pixels = np.any(clipped_image > 0, axis=2)
    if not np.any(glow_pixels):
        return

    roi = frame[clipped_top:clipped_bottom, clipped_left:clipped_right]
    roi[glow_pixels] = cv2.addWeighted(
        roi[glow_pixels],
        1.0,
        clipped_image[glow_pixels],
        alpha,
        0.0,
    )


def _draw_glow(
    base: np.ndarray, layer: np.ndarray, radius_px: int, alpha: float
) -> np.ndarray:
    kernel = _ensure_odd_kernel(radius_px)
    if kernel == 0:
        return _blend_layer(base, layer, alpha)
    blurred = cv2.GaussianBlur(layer, (kernel, kernel), 0)
    return _blend_layer(base, blurred, alpha)


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


def _draw_trace(
    frame: np.ndarray,
    points: Sequence[tuple[int, int]],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    if len(points) < 2 or thickness <= 0:
        return
    cv2.polylines(
        frame,
        [np.array(points, dtype=np.int32)],
        isClosed=False,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )


def _label_metrics(label: str, config: AppConfig) -> tuple[int, int, int, int, int]:
    text_size, baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        config.render.label_font_scale,
        config.render.label_thickness,
    )
    text_width, text_height = text_size
    padding = config.render.label_padding_px
    label_width = text_width + (padding * 2)
    label_height = text_height + (padding * 2)
    return text_width, text_height, baseline, label_width, label_height


def _layout_from_bounds(
    left: int,
    top: int,
    label: str,
    config: AppConfig,
) -> LabelLayout:
    _text_width, text_height, baseline, label_width, label_height = _label_metrics(
        label, config
    )
    padding = config.render.label_padding_px
    text_origin = (
        left + padding,
        top + int(round((label_height + text_height) / 2.0)),
    )
    return LabelLayout(
        left=left,
        top=top,
        right=left + label_width,
        bottom=top + label_height,
        baseline=baseline,
        text_origin=text_origin,
    )


def _anchored_label_layout(
    frame_shape: tuple[int, ...],
    track: TrackedObject,
    label: str,
    config: AppConfig,
) -> LabelLayout:
    _text_width, _text_height, _baseline, label_width, label_height = _label_metrics(
        label, config
    )
    frame_height, frame_width = frame_shape[:2]
    left = int(round(track.bbox.x1 - config.render.label_padding_px + 1))
    left = max(0, min(left, max(0, frame_width - label_width)))
    top = int(round(track.bbox.y1 - label_height - config.render.label_gap_px))
    top = max(0, min(top, max(0, frame_height - label_height)))
    return _layout_from_bounds(left, top, label, config)


def _clamp_label_box(
    box: np.ndarray,
    anchor_box: np.ndarray,
    frame_shape: tuple[int, ...],
    max_offset_px: int,
) -> np.ndarray:
    frame_height, frame_width = frame_shape[:2]
    width = box[2] - box[0]
    height = box[3] - box[1]
    left = float(box[0])
    top = float(box[1])

    if max_offset_px >= 0:
        left = max(
            anchor_box[0] - max_offset_px,
            min(left, anchor_box[0] + max_offset_px),
        )
        top = max(
            anchor_box[1] - max_offset_px,
            min(top, anchor_box[1] + max_offset_px),
        )

    left = max(0.0, min(left, max(0.0, frame_width - width)))
    top = max(0.0, min(top, max(0.0, frame_height - height)))
    return np.array([left, top, left + width, top + height], dtype=np.float32)


def resolve_label_box_bounds(
    frame_shape: tuple[int, ...],
    tracks: Sequence[TrackedObject],
    labels: Sequence[str],
    config: AppConfig,
) -> list[LabelLayout]:
    if len(tracks) != len(labels):
        raise ValueError("tracks and labels must have the same length")
    anchored_layouts = [
        _anchored_label_layout(frame_shape, track, label, config)
        for track, label in zip(tracks, labels, strict=True)
    ]
    if not config.render.label_smart_position or len(anchored_layouts) < 2:
        return anchored_layouts

    anchor_boxes = np.array(
        [
            [layout.left, layout.top, layout.right, layout.bottom]
            for layout in anchored_layouts
        ],
        dtype=np.float32,
    )
    spread_boxes = spread_out_boxes(anchor_boxes.copy()).astype(np.float32)
    clamped_boxes = np.array(
        [
            _clamp_label_box(
                box=box,
                anchor_box=anchor_box,
                frame_shape=frame_shape,
                max_offset_px=config.render.label_max_offset_px,
            )
            for box, anchor_box in zip(spread_boxes, anchor_boxes, strict=True)
        ],
        dtype=np.float32,
    )

    layouts: list[LabelLayout] = []
    for box, label in zip(clamped_boxes, labels, strict=True):
        layouts.append(
            _layout_from_bounds(
                left=int(round(float(box[0]))),
                top=int(round(float(box[1]))),
                label=label,
                config=config,
            )
        )
    return layouts


def label_box_bounds(
    frame_shape: tuple[int, ...],
    track: TrackedObject,
    label: str,
    config: AppConfig,
) -> tuple[int, int, int, int, int]:
    layout = _anchored_label_layout(frame_shape, track, label, config)
    return layout.left, layout.top, layout.right, layout.bottom, layout.baseline


class VideoAnnotator:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.corner_annotator = sv.BoxCornerAnnotator(
            color=sv.Color.from_hex(config.render.box_color),
            thickness=config.render.corner_thickness,
            corner_length=config.render.corner_length,
        )
        self.glow_corner_annotator = sv.BoxCornerAnnotator(
            color=sv.Color.from_hex(config.render.glow_color),
            thickness=config.render.corner_thickness + 2,
            corner_length=config.render.corner_length,
        )
        self.glow_color = _color_to_bgr(config.render.glow_color)
        self.label_bg_color = _color_to_bgr(config.render.label_bg_color)
        self.label_text_color = _color_to_bgr(config.render.label_text_color)
        self.label_shadow_color = _color_to_bgr(config.render.label_shadow_color)

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

        glow_layer = np.zeros_like(annotated)
        solid_layer = np.zeros_like(annotated)
        detections = _tracks_to_detections(tracks)
        label_text = [
            labels_by_track.get(track.track_id, self.config.render.unknown_label)
            for track in tracks
        ]

        if self.config.render.glow_enabled:
            self.glow_corner_annotator.annotate(glow_layer, detections)
        self.corner_annotator.annotate(solid_layer, detections)

        if self.config.render.glow_enabled:
            annotated = _draw_glow(
                annotated,
                glow_layer,
                self.config.render.glow_radius_px,
                self.config.render.glow_alpha,
            )
        annotated = _blend_layer(annotated, solid_layer, 1.0)

        layouts = resolve_label_box_bounds(
            frame_shape=annotated.shape,
            tracks=tracks,
            labels=label_text,
            config=self.config,
        )
        for label, layout in zip(label_text, layouts, strict=True):
            self._draw_label(annotated, label, layout)
        return annotated

    def _draw_label(self, frame: np.ndarray, label: str, layout: LabelLayout) -> None:
        _overlay_rect(
            frame,
            layout.left,
            layout.top,
            layout.right,
            layout.bottom,
            self.label_bg_color,
            self.config.render.label_bg_alpha,
        )

        if self.config.render.label_shadow_enabled:
            shadow_offset = self.config.render.label_shadow_offset_px
            shadow_thickness = (
                self.config.render.label_thickness
                + self.config.render.label_shadow_thickness_extra
            )
            shadow_margin = max(2, shadow_offset + shadow_thickness + 2)
            shadow_left = layout.left
            shadow_top = layout.top
            shadow_right = layout.right + shadow_margin
            shadow_bottom = layout.bottom + shadow_margin
            shadow_width = max(0, shadow_right - shadow_left)
            shadow_height = max(0, shadow_bottom - shadow_top)
            shadow_mask = np.zeros((shadow_height, shadow_width), dtype=np.uint8)
            shadow_origin = (
                layout.text_origin[0] + shadow_offset - shadow_left,
                layout.text_origin[1] + shadow_offset - shadow_top,
            )
            cv2.putText(
                shadow_mask,
                label,
                shadow_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.render.label_font_scale,
                255,
                shadow_thickness,
                cv2.LINE_AA,
            )
            _overlay_text_mask_roi(
                frame,
                shadow_mask,
                shadow_left,
                shadow_top,
                self.label_shadow_color,
                self.config.render.label_shadow_alpha,
            )

        if self.config.render.label_glow_alpha > 0.0:
            glow_margin = _ensure_odd_kernel(self.config.render.label_glow_radius_px)
            if glow_margin == 0:
                glow_margin = 1
            glow_left = layout.left - glow_margin
            glow_top = layout.top - glow_margin
            glow_right = layout.right + glow_margin
            glow_bottom = layout.bottom + glow_margin
            glow_width = max(0, glow_right - glow_left)
            glow_height = max(0, glow_bottom - glow_top)
            text_glow_layer = np.zeros((glow_height, glow_width, 3), dtype=np.uint8)
            glow_origin = (
                layout.text_origin[0] - glow_left,
                layout.text_origin[1] - glow_top,
            )
            cv2.putText(
                text_glow_layer,
                label,
                glow_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.render.label_font_scale,
                self.glow_color,
                self.config.render.label_thickness + 2,
                cv2.LINE_AA,
            )
            if glow_margin > 1:
                text_glow_layer = cv2.GaussianBlur(
                    text_glow_layer,
                    (glow_margin, glow_margin),
                    0,
                )
            _overlay_image_roi(
                frame,
                text_glow_layer,
                glow_left,
                glow_top,
                self.config.render.label_glow_alpha,
            )

        cv2.putText(
            frame,
            label,
            layout.text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.render.label_font_scale,
            self.label_text_color,
            self.config.render.label_thickness,
            cv2.LINE_AA,
        )
