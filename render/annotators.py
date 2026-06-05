from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import AppConfig
from models import TrackedObject

_EMOJI_FONT_NAME = "NotoColorEmoji.ttf"
_EMOJI_FONT_NATIVE_SIZE = 109


@dataclass(frozen=True)
class _LabelLineLayout:
    text: str
    flag: str | None
    text_size: tuple[int, int]
    baseline: int
    flag_width: int
    flag_gap_width: int

    @property
    def width(self) -> int:
        return self.flag_width + self.flag_gap_width + self.text_size[0]


class VideoAnnotator:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def annotate(
        self,
        frame: np.ndarray,
        tracks: Sequence[TrackedObject],
        labels_by_track: dict[int, str],
        counter_value: int | None = None,
        label_text_colors_by_track: Mapping[int, str] | None = None,
    ) -> np.ndarray:
        annotated = frame.copy()
        for track in tracks:
            _draw_track_box(annotated, track, self.config)
        label_draw_items: list[tuple[float, int, TrackedObject, str]] = []
        for index, track in enumerate(tracks):
            label = labels_by_track.get(
                track.track_id, self.config.render.unknown_label
            )
            label_draw_items.append((track.bbox.area, index, track, label))
        for _, _, track, display_label in sorted(label_draw_items):
            _draw_track_label(
                annotated,
                track,
                display_label,
                self.config,
                text_color=(
                    label_text_colors_by_track.get(track.track_id)
                    if label_text_colors_by_track is not None
                    else None
                ),
            )
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


def _is_flag_emoji(value: str) -> bool:
    return len(value) == 2 and all(
        "\U0001f1e6" <= character <= "\U0001f1ff" for character in value
    )


def _split_leading_flag(line: str) -> tuple[str | None, str]:
    if len(line) >= 4 and line[2] == " " and _is_flag_emoji(line[:2]):
        return line[:2], line[3:]
    return None, line


@lru_cache(maxsize=None)
def _render_native_flag_emoji(flag: str) -> Image.Image:
    if not _is_flag_emoji(flag):
        raise ValueError(f"Expected flag emoji, got {flag!r}")
    try:
        font = ImageFont.truetype(_EMOJI_FONT_NAME, _EMOJI_FONT_NATIVE_SIZE)
    except OSError as error:
        raise RuntimeError(
            "Noto Color Emoji is required to render country flags. "
            "Install NotoColorEmoji.ttf so Pillow can find it."
        ) from error

    canvas = Image.new("RGBA", (160, 140), (0, 0, 0, 0))
    ImageDraw.Draw(canvas).text(
        (0, 0),
        flag,
        font=font,
        embedded_color=True,
    )
    bounding_box = canvas.getbbox()
    if bounding_box is None:
        raise RuntimeError(f"Noto Color Emoji did not render flag {flag!r}")
    return canvas.crop(bounding_box)


@lru_cache(maxsize=512)
def _render_flag_emoji(flag: str, target_height: int) -> np.ndarray:
    if target_height <= 0:
        raise ValueError(f"Expected positive flag height, got {target_height}")
    cropped = _render_native_flag_emoji(flag)
    target_width = max(1, round(cropped.width * target_height / cropped.height))
    resized = cropped.resize(
        (target_width, target_height),
        resample=Image.Resampling.LANCZOS,
    )
    return np.asarray(resized, dtype=np.uint8)


def _label_line_layouts(
    lines: Sequence[str],
    *,
    font: int,
    scale: float,
    normal_thickness: int,
    title_thickness: int,
    flag_gap_width: int,
) -> list[_LabelLineLayout]:
    layouts: list[_LabelLineLayout] = []
    for index, line in enumerate(lines):
        flag, text = _split_leading_flag(line)
        thickness = title_thickness if index == 0 else normal_thickness
        size, baseline = cv2.getTextSize(
            text,
            font,
            scale,
            thickness,
        )
        flag_width = 0
        if flag is not None:
            flag_width = int(_render_flag_emoji(flag, max(1, int(size[1]))).shape[1])
        layouts.append(
            _LabelLineLayout(
                text=text,
                flag=flag,
                text_size=(int(size[0]), int(size[1])),
                baseline=int(baseline),
                flag_width=flag_width,
                flag_gap_width=flag_gap_width if flag is not None else 0,
            )
        )
    return layouts


def _label_scale_factor(track: TrackedObject, config: AppConfig) -> float:
    return track.bbox.width / config.render.label_scale_reference_box_width_px


def _draw_track_box(
    frame: np.ndarray,
    track: TrackedObject,
    config: AppConfig,
) -> None:
    if not config.render.box_enabled:
        return

    frame_height, frame_width = frame.shape[:2]
    x1 = min(max(0, round(track.bbox.x1)), frame_width - 1)
    y1 = min(max(0, round(track.bbox.y1)), frame_height - 1)
    x2 = min(max(0, round(track.bbox.x2)), frame_width - 1)
    y2 = min(max(0, round(track.bbox.y2)), frame_height - 1)
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


def _alpha_composite_rgba(
    frame: np.ndarray,
    rgba: np.ndarray,
    x: int,
    y: int,
) -> None:
    frame_height, frame_width = frame.shape[:2]
    image_height, image_width = rgba.shape[:2]
    frame_x1 = max(0, x)
    frame_y1 = max(0, y)
    frame_x2 = min(frame_width, x + image_width)
    frame_y2 = min(frame_height, y + image_height)
    if frame_x1 >= frame_x2 or frame_y1 >= frame_y2:
        return

    image_x1 = frame_x1 - x
    image_y1 = frame_y1 - y
    image_x2 = image_x1 + (frame_x2 - frame_x1)
    image_y2 = image_y1 + (frame_y2 - frame_y1)
    source = rgba[image_y1:image_y2, image_x1:image_x2]
    destination = frame[frame_y1:frame_y2, frame_x1:frame_x2]
    alpha = source[:, :, 3:4].astype(np.float32) / 255.0
    source_bgr = source[:, :, :3][:, :, ::-1].astype(np.float32)
    destination[:] = (
        source_bgr * alpha + destination.astype(np.float32) * (1.0 - alpha)
    ).astype(np.uint8)


def _draw_track_label(
    frame: np.ndarray,
    track: TrackedObject,
    label: str,
    config: AppConfig,
    *,
    scale_factor: float | None = None,
    text_color: str | None = None,
) -> None:
    lines = [line.strip() for line in label.splitlines() if line.strip()]
    if not lines:
        return
    if track.bbox.width <= 0:
        return

    scale_factor = (
        scale_factor if scale_factor is not None else _label_scale_factor(track, config)
    )
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = config.render.label_font_scale * scale_factor
    normal_thickness = max(1, config.render.label_thickness)
    padding = max(0, round(config.render.label_padding_px * scale_factor))
    gap = max(0, round(config.render.label_gap_px * scale_factor))
    flag_gap = max(0, round(config.render.label_flag_gap_px * scale_factor))
    line_gap = max(0, round(4 * scale))
    layouts = _label_line_layouts(
        lines,
        font=font,
        scale=scale,
        normal_thickness=normal_thickness,
        title_thickness=normal_thickness,
        flag_gap_width=flag_gap,
    )

    text_width = max(layout.width for layout in layouts)
    interline_descent = sum(layout.baseline for layout in layouts[:-1])
    final_descent = max(0, layouts[-1].baseline - padding)
    text_height = sum(layout.text_size[1] for layout in layouts)
    text_height += interline_descent + final_descent
    text_height += line_gap * (len(lines) - 1)
    box_width = text_width + padding * 2
    box_height = text_height + padding * 2

    frame_height, frame_width = frame.shape[:2]
    x1 = round(track.bbox.x1)
    y1 = round(track.bbox.y2) + gap
    y1 = max(0, y1)
    x2 = x1 + box_width
    y2 = y1 + box_height
    if x2 <= 0 or x1 >= frame_width or y2 <= 0 or y1 >= frame_height:
        return

    clipped_x1 = max(0, x1)
    clipped_y1 = max(0, y1)
    clipped_x2 = min(frame_width, x2)
    clipped_y2 = min(frame_height, y2)

    label_region = frame[clipped_y1:clipped_y2, clipped_x1:clipped_x2]
    overlay = label_region.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (clipped_x2 - clipped_x1, clipped_y2 - clipped_y1),
        _hex_to_bgr(config.render.label_bg_color),
        -1,
    )
    cv2.addWeighted(
        overlay,
        config.render.label_bg_alpha,
        label_region,
        1.0 - config.render.label_bg_alpha,
        0,
        dst=label_region,
    )

    resolved_text_color = _hex_to_bgr(
        text_color if text_color is not None else config.render.label_text_color
    )
    baseline_y = y1 + padding
    for layout in layouts:
        baseline_y += layout.text_size[1]
        text_x = x1 + padding
        if layout.flag is not None:
            flag_image = _render_flag_emoji(
                layout.flag,
                max(1, layout.text_size[1]),
            )
            _alpha_composite_rgba(
                frame,
                flag_image,
                text_x,
                baseline_y - int(flag_image.shape[0]),
            )
            text_x += layout.flag_width + layout.flag_gap_width
        cv2.putText(
            frame,
            layout.text,
            (text_x, baseline_y),
            font,
            scale,
            resolved_text_color,
            normal_thickness,
            cv2.LINE_AA,
        )
        baseline_y += layout.baseline + line_gap


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
    text = f"{value}"
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
    text_y = padding + int(text_height) + max(0, int(round(baseline / 2)) - 1)
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
    cv2.addWeighted(
        overlay,
        config.render.label_bg_alpha,
        counter_region,
        1.0 - config.render.label_bg_alpha,
        0,
        dst=counter_region,
    )
    cv2.putText(
        frame,
        text,
        (x1 + padding, y1 + text_y),
        font,
        scale,
        _hex_to_bgr(config.render.label_text_color),
        thickness,
        cv2.LINE_AA,
    )
