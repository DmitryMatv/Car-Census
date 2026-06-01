from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from config import AppConfig
from models import TrackedObject


class VideoAnnotator:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def annotate(
        self,
        frame: np.ndarray,
        tracks: Sequence[TrackedObject],
        labels_by_track: dict[int, str],
        counter_value: int | None = None,
    ) -> np.ndarray:
        annotated = frame.copy()
        for track in tracks:
            _draw_track_box(annotated, track, self.config)
        for track in tracks:
            label = labels_by_track.get(
                track.track_id, self.config.render.unknown_label
            )
            display_label = _label_for_track_size(track, label, self.config)
            if display_label is not None:
                _draw_track_label(annotated, track, display_label, self.config)
        if self.config.render.counter_enabled and counter_value is not None:
            _draw_counter(annotated, counter_value, self.config)
        return annotated


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    normalized = hex_color.lstrip("#")
    if len(normalized) != 6:
        raise ValueError(f"Expected #RRGGBB color, got {hex_color!r}")
    red = int(normalized[0:2], 16)
    green = int(normalized[2:4], 16)
    blue = int(normalized[4:6], 16)
    return blue, green, red


def _label_line_sizes(
    lines: Sequence[str],
    *,
    font: int,
    scale: float,
    normal_thickness: int,
    title_thickness: int,
) -> list[tuple[tuple[int, int], int]]:
    sizes: list[tuple[tuple[int, int], int]] = []
    for index, line in enumerate(lines):
        size, baseline = cv2.getTextSize(
            line,
            font,
            scale,
            title_thickness if index == 0 else normal_thickness,
        )
        sizes.append(((int(size[0]), int(size[1])), int(baseline)))
    return sizes


def _first_label_line(label: str) -> str | None:
    for line in label.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def _tiny_id_label(track: TrackedObject, config: AppConfig) -> str:
    if (
        config.render.label_tiny_id_source == "vehicle_index"
        and track.vehicle_index is not None
    ):
        return f"#{track.vehicle_index}"
    return f"#{track.track_id}"


def _label_for_track_size(
    track: TrackedObject,
    label: str,
    config: AppConfig,
) -> str | None:
    if not config.render.label_simplification_enabled:
        return label

    width = track.bbox.width
    if width >= config.render.label_full_min_box_width_px:
        return label
    if width >= config.render.label_make_model_min_box_width_px:
        return _first_label_line(label)
    if config.render.label_tiny_mode == "id":
        return _tiny_id_label(track, config)
    return None


def _draw_track_box(
    frame: np.ndarray,
    track: TrackedObject,
    config: AppConfig,
) -> None:
    if not config.render.box_enabled:
        return

    frame_height, frame_width = frame.shape[:2]
    x1 = min(max(0, int(round(track.bbox.x1))), frame_width - 1)
    y1 = min(max(0, int(round(track.bbox.y1))), frame_height - 1)
    x2 = min(max(0, int(round(track.bbox.x2))), frame_width - 1)
    y2 = min(max(0, int(round(track.bbox.y2))), frame_height - 1)
    if x2 <= x1 or y2 <= y1:
        return

    region = frame[y1 : y2 + 1, x1 : x2 + 1]
    overlay = region.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (x2 - x1, y2 - y1),
        _hex_to_bgr(config.render.box_color),
        config.render.box_thickness,
    )
    cv2.addWeighted(
        overlay,
        config.render.box_alpha,
        region,
        1.0 - config.render.box_alpha,
        0,
        dst=region,
    )


def _draw_track_label(
    frame: np.ndarray,
    track: TrackedObject,
    label: str,
    config: AppConfig,
) -> None:
    lines = [line.strip() for line in label.splitlines() if line.strip()]
    if not lines:
        return

    font = cv2.FONT_HERSHEY_DUPLEX
    scale = config.render.label_font_scale
    normal_thickness = max(1, config.render.label_thickness)
    padding = config.render.label_padding_px
    gap = config.render.label_gap_px
    line_gap = max(2, int(round(4 * scale)))
    sizes = _label_line_sizes(
        lines,
        font=font,
        scale=scale,
        normal_thickness=normal_thickness,
        title_thickness=normal_thickness,
    )

    text_width = max(size[0][0] for size in sizes)
    text_height = sum(size[0][1] + size[1] for size in sizes)
    text_height += line_gap * (len(lines) - 1)
    box_width = text_width + padding * 2
    box_height = text_height + padding * 2

    frame_height, frame_width = frame.shape[:2]
    x1 = int(round(track.bbox.x1))
    y1 = int(round(track.bbox.y2)) + gap
    x1 = min(max(0, x1), max(0, frame_width - box_width))
    y1 = max(0, y1)
    if y1 >= frame_height:
        return
    x2 = x1 + box_width
    y2 = min(frame_height, y1 + box_height)

    label_region = frame[y1:y2, x1:x2]
    overlay = label_region.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (box_width, y2 - y1),
        _hex_to_bgr(config.render.label_bg_color),
        -1,
    )
    cv2.addWeighted(overlay, 0.5, label_region, 0.5, 0, dst=label_region)

    text_color = _hex_to_bgr(config.render.label_text_color)
    baseline_y = y1 + padding
    for line, (size, baseline) in zip(lines, sizes, strict=True):
        baseline_y += size[1]
        cv2.putText(
            frame,
            line,
            (x1 + padding, baseline_y),
            font,
            scale,
            text_color,
            normal_thickness,
            cv2.LINE_AA,
        )
        baseline_y += baseline + line_gap


def _counter_origin(
    *,
    frame_width: int,
    frame_height: int,
    box_width: int,
    box_height: int,
    padding: int,
    position: str,
) -> tuple[int, int]:
    if position.endswith("right"):
        x1 = frame_width - box_width - padding
    else:
        x1 = padding
    if position.startswith("bottom"):
        y1 = frame_height - box_height - padding
    else:
        y1 = padding
    return (
        min(max(0, x1), max(0, frame_width - box_width)),
        min(max(0, y1), max(0, frame_height - box_height)),
    )


def _draw_counter(frame: np.ndarray, value: int, config: AppConfig) -> None:
    text = f"Cars: {value}"
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = config.render.label_font_scale * 2
    thickness = max(1, config.render.label_thickness)
    padding = config.render.label_padding_px * 2
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        scale,
        thickness,
    )
    box_width = int(text_width) + padding * 2
    box_height = int(text_height) + int(baseline) + padding * 2
    frame_height, frame_width = frame.shape[:2]
    x1, y1 = _counter_origin(
        frame_width=frame_width,
        frame_height=frame_height,
        box_width=box_width,
        box_height=box_height,
        padding=padding,
        position=config.render.counter_position,
    )
    x2 = x1 + box_width
    y2 = y1 + box_height

    counter_region = frame[y1:y2, x1:x2]
    overlay = counter_region.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (box_width, box_height),
        _hex_to_bgr(config.render.label_bg_color),
        -1,
    )
    cv2.addWeighted(overlay, 0.5, counter_region, 0.5, 0, dst=counter_region)
    cv2.putText(
        frame,
        text,
        (x1 + padding, y1 + padding + int(text_height)),
        font,
        scale,
        _hex_to_bgr(config.render.label_text_color),
        thickness,
        cv2.LINE_AA,
    )
