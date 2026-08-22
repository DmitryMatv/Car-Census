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
from render.label_emojis import TAG_EMOJI_ORDER

_EMOJI_FONT_NAME = "NotoColorEmoji.ttf"
_EMOJI_FONT_NATIVE_SIZE = 109
_MAKE_STATS_BAR_COLOR = "#D6D6D6"
_MAKE_STATS_PULSE_BAR_COLOR = "#E8E8E8"
_MAKE_STATS_TRACK_COLOR = "#505050"
_MAKE_STATS_SEPARATOR_COLOR = "#8A8A8A"
_MAKE_STATS_PULSE_FRAMES = 8
_MAKE_STATS_ROW_EASING = 0.35
_MAKE_STATS_PROGRESS_EASING = 0.30


@dataclass(frozen=True)
class MakeStatisticRow:
    make: str
    origin_flag: str | None
    count: int
    progress: float


@dataclass(frozen=True)
class _LabelLineLayout:
    text: str
    flags: tuple[str, ...]
    text_size: tuple[int, int]
    baseline: int
    flag_widths: tuple[int, ...]
    flag_gap_width: int
    trailing_emojis: tuple[str, ...]
    trailing_emoji_widths: tuple[int, ...]

    @property
    def width(self) -> int:
        flags_width = sum(self.flag_widths)
        trailing_width = sum(self.trailing_emoji_widths)
        trailing_gap_width = self.flag_gap_width * len(self.trailing_emojis)
        leading_gap_width = self.flag_gap_width if self.flags else 0
        return (
            flags_width
            + leading_gap_width
            + self.text_size[0]
            + trailing_width
            + trailing_gap_width
        )


class VideoAnnotator:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._make_row_y_by_make: dict[str, float] = {}
        self._make_progress_by_make: dict[str, float] = {}
        self._make_count_by_make: dict[str, int] = {}
        self._make_pulse_frames_by_make: dict[str, int] = {}

    def annotate(
        self,
        frame: np.ndarray,
        tracks: Sequence[TrackedObject],
        labels_by_track: dict[int, str],
        counter_value: int | None = None,
        label_text_colors_by_track: Mapping[int, str] | None = None,
        make_statistics_rows: Sequence[MakeStatisticRow] = (),
    ) -> np.ndarray:
        draw_items: list[tuple[float, TrackedObject, str, str | None]] = []
        for track in tracks:
            label = labels_by_track.get(
                track.track_id, self.config.render.unknown_label
            )
            text_color = (
                label_text_colors_by_track.get(track.track_id)
                if label_text_colors_by_track is not None
                else None
            )
            draw_items.append((track.bbox.area, track, label, text_color))
        draw_items.sort(key=lambda item: item[0])

        annotated = frame.copy()
        for track in tracks:
            _draw_track_box(annotated, track, self.config)
        for _, track, label, tc in draw_items:
            _draw_track_label(
                annotated,
                track,
                label,
                self.config,
                text_color=tc,
            )

        if self.config.render.counter_enabled:
            if make_statistics_rows:
                self._draw_make_statistics(
                    annotated,
                    make_statistics_rows,
                    counter_value=counter_value,
                )
            elif counter_value is not None:
                _draw_counter(annotated, counter_value, self.config)
        return annotated

    def _draw_make_statistics(
        self,
        frame: np.ndarray,
        rows: Sequence[MakeStatisticRow],
        *,
        counter_value: int | None,
    ) -> None:
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = max(0.45, self.config.render.label_font_scale)
        total_scale = scale * 1.12
        thickness = max(1, self.config.render.label_thickness)
        padding = max(2, self.config.render.label_padding_px * 2)
        row_gap = max(2, padding // 3)
        total_gap = max(4, padding // 2)
        column_gap = max(4, padding)
        flag_gap = max(0, self.config.render.label_flag_gap_px)
        frame_height, frame_width = frame.shape[:2]
        if frame_height <= padding * 2 or frame_width <= padding * 2:
            return

        (_, text_height), baseline = cv2.getTextSize("Ag", font, scale, thickness)
        row_height = int(text_height) + max(2, int(baseline))
        (_, total_text_height), total_baseline = cv2.getTextSize(
            str(counter_value or 0),
            font,
            total_scale,
            thickness,
        )
        total_row_height = int(total_text_height) + max(2, int(total_baseline))
        total_block_height = total_gap + 1 + total_gap + total_row_height
        available_height = frame_height - padding * 2
        max_rows = max(
            0,
            (available_height - padding * 2 - total_block_height + row_gap)
            // (row_height + row_gap),
        )
        visible_rows = list(rows[:max_rows])
        if not visible_rows:
            return

        count_width = max(
            cv2.getTextSize(str(row.count), font, scale, thickness)[0][0]
            for row in visible_rows
        )
        flag_height = max(1, int(text_height))
        row_flags = [_split_flag_emojis(row.origin_flag or "") for row in visible_rows]
        row_flag_widths = [
            sum(int(_render_flag_emoji(flag, flag_height).shape[1]) for flag in flags)
            for flags in row_flags
        ]
        flag_slot_width = max(row_flag_widths, default=0)
        bar_width = max(32, round(120 * scale))
        bar_height = max(4, round(text_height * 0.38))
        desired_make_width = max(
            cv2.getTextSize(row.make, font, scale, thickness)[0][0]
            for row in visible_rows
        )
        max_panel_width = frame_width - padding * 2
        fixed_width = (
            padding * 2
            + flag_slot_width
            + (flag_gap if flag_slot_width else 0)
            + column_gap
            + count_width
            + column_gap
            + bar_width
        )
        make_width = min(desired_make_width, max(0, max_panel_width - fixed_width))
        if make_width <= 0:
            return
        panel_width = min(max_panel_width, fixed_width + make_width)
        panel_height = (
            padding * 2
            + len(visible_rows) * row_height
            + (len(visible_rows) - 1) * row_gap
            + total_block_height
        )
        if panel_width <= 0 or panel_height <= 0:
            return

        x1, y1 = _counter_origin(
            frame_width=frame_width,
            frame_height=frame_height,
            box_width=panel_width,
            box_height=panel_height,
            padding=padding,
            position=self.config.render.counter_position,
        )
        x2 = min(frame_width, x1 + panel_width)
        y2 = min(frame_height, y1 + panel_height)
        if x2 <= x1 or y2 <= y1:
            return

        panel_region = frame[y1:y2, x1:x2]
        overlay = panel_region.copy()
        cv2.rectangle(
            overlay,
            (0, 0),
            (x2 - x1, y2 - y1),
            _hex_to_bgr(self.config.render.label_bg_color),
            -1,
        )
        cv2.addWeighted(
            overlay,
            self.config.render.label_bg_alpha,
            panel_region,
            1.0 - self.config.render.label_bg_alpha,
            0,
            dst=panel_region,
        )
        panel = panel_region.copy()

        row_stride = row_height + row_gap
        visible_labels = {row.make for row in visible_rows}
        for label in set(self._make_row_y_by_make) - visible_labels:
            self._make_row_y_by_make.pop(label, None)
            self._make_progress_by_make.pop(label, None)
            self._make_count_by_make.pop(label, None)
            self._make_pulse_frames_by_make.pop(label, None)

        animated_rows: list[
            tuple[float, MakeStatisticRow, float, int, tuple[str, ...]]
        ] = []
        for index, row in enumerate(visible_rows):
            label = row.make
            target_y = float(index * row_stride)
            previous_y = self._make_row_y_by_make.get(
                label,
                target_y + row_stride,
            )
            current_y = previous_y + (target_y - previous_y) * _MAKE_STATS_ROW_EASING
            self._make_row_y_by_make[label] = current_y

            previous_progress = self._make_progress_by_make.get(label, 0.0)
            target_progress = min(max(row.progress, 0.0), 1.0)
            current_progress = (
                previous_progress
                + (target_progress - previous_progress) * _MAKE_STATS_PROGRESS_EASING
            )
            self._make_progress_by_make[label] = current_progress

            previous_count = self._make_count_by_make.get(label)
            if previous_count is None or row.count > previous_count:
                self._make_pulse_frames_by_make[label] = _MAKE_STATS_PULSE_FRAMES
            self._make_count_by_make[label] = row.count

            pulse_frames = self._make_pulse_frames_by_make.get(label, 0)
            animated_rows.append(
                (current_y, row, current_progress, pulse_frames, row_flags[index])
            )

        text_color = _hex_to_bgr(self.config.render.label_text_color)
        bar_color = _hex_to_bgr(_MAKE_STATS_BAR_COLOR)
        pulse_bar_color = _hex_to_bgr(_MAKE_STATS_PULSE_BAR_COLOR)
        track_color = _hex_to_bgr(_MAKE_STATS_TRACK_COLOR)
        separator_color = _hex_to_bgr(_MAKE_STATS_SEPARATOR_COLOR)

        for animated_y, row, animated_progress, pulse_frames, flags in sorted(
            animated_rows, key=lambda item: item[0]
        ):
            row_top = padding + round(animated_y)
            baseline_y = row_top + int(text_height)
            if row_top + row_height < 0 or row_top >= panel.shape[0]:
                continue

            text_x = padding
            for flag in flags:
                flag_image = _render_flag_emoji(flag, flag_height)
                _alpha_composite_rgba(
                    panel,
                    flag_image,
                    text_x,
                    baseline_y - int(flag_image.shape[0]),
                )
                text_x += int(flag_image.shape[1])
            text_x += flag_slot_width + (flag_gap if flag_slot_width else 0)

            display_make = _truncate_text_to_width(
                row.make,
                max_width=make_width,
                font=font,
                scale=scale,
                thickness=thickness,
            )
            cv2.putText(
                panel,
                display_make,
                (text_x, baseline_y),
                font,
                scale,
                text_color,
                thickness,
                cv2.LINE_AA,
            )

            count_text = str(row.count)
            count_text_width = cv2.getTextSize(
                count_text,
                font,
                scale,
                thickness,
            )[0][0]
            count_x = text_x + make_width + column_gap + count_width - count_text_width
            cv2.putText(
                panel,
                count_text,
                (count_x, baseline_y),
                font,
                scale,
                text_color,
                thickness,
                cv2.LINE_AA,
            )

            bar_x = text_x + make_width + column_gap + count_width + column_gap
            bar_y = row_top + max(0, (row_height - bar_height) // 2)
            cv2.rectangle(
                panel,
                (bar_x, bar_y),
                (bar_x + bar_width, bar_y + bar_height),
                track_color,
                -1,
            )
            fill_width = max(1, round(bar_width * animated_progress))
            cv2.rectangle(
                panel,
                (bar_x, bar_y),
                (bar_x + fill_width, bar_y + bar_height),
                pulse_bar_color if pulse_frames > 0 else bar_color,
                -1,
            )
            cv2.rectangle(
                panel,
                (bar_x, bar_y),
                (bar_x + bar_width, bar_y + bar_height),
                bar_color,
                max(1, thickness),
            )
            if pulse_frames > 0:
                self._make_pulse_frames_by_make[row.make] = pulse_frames - 1

        total_separator_y = (
            padding
            + len(visible_rows) * row_height
            + (len(visible_rows) - 1) * row_gap
            + total_gap
        )
        cv2.line(
            panel,
            (padding, total_separator_y),
            (max(padding, panel.shape[1] - padding), total_separator_y),
            separator_color,
            1,
            cv2.LINE_AA,
        )
        total_text = str(counter_value or 0)
        total_text_size = cv2.getTextSize(
            total_text,
            font,
            total_scale,
            thickness,
        )[0]
        total_x = max(0, (panel.shape[1] - total_text_size[0]) // 2)
        total_baseline_y = total_separator_y + total_gap + int(total_text_height)
        cv2.putText(
            panel,
            total_text,
            (total_x, total_baseline_y),
            font,
            total_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

        panel_region[:] = panel


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    normalized = hex_color.lstrip("#")
    if len(normalized) != 6:
        raise ValueError(f"Expected #RRGGBB color, got {hex_color!r}")
    red = int(normalized[0:2], 16)
    green = int(normalized[2:4], 16)
    blue = int(normalized[4:6], 16)
    return blue, green, red


def _truncate_text_to_width(
    text: str,
    *,
    max_width: int,
    font: int,
    scale: float,
    thickness: int,
) -> str:
    if max_width <= 0:
        return ""
    if cv2.getTextSize(text, font, scale, thickness)[0][0] <= max_width:
        return text
    ellipsis = "..."
    if cv2.getTextSize(ellipsis, font, scale, thickness)[0][0] > max_width:
        return ""
    truncated = text.rstrip()
    while truncated:
        candidate = f"{truncated}{ellipsis}"
        if cv2.getTextSize(candidate, font, scale, thickness)[0][0] <= max_width:
            return candidate
        truncated = truncated[:-1].rstrip()
    return ellipsis


def _is_flag_emoji(value: str) -> bool:
    return len(value) == 2 and all(
        "\U0001f1e6" <= character <= "\U0001f1ff" for character in value
    )


def _split_flag_emojis(value: str) -> tuple[str, ...]:
    flags: list[str] = []
    index = 0
    while index + 2 <= len(value) and _is_flag_emoji(value[index : index + 2]):
        flags.append(value[index : index + 2])
        index += 2
    return tuple(flags)


def _split_leading_flags(line: str) -> tuple[tuple[str, ...], str]:
    flags: list[str] = []
    i = 0
    while i + 1 < len(line) and _is_flag_emoji(line[i : i + 2]):
        flags.append(line[i : i + 2])
        i += 2
    if i < len(line) and line[i] == " " and flags:
        return tuple(flags), line[i + 1 :]
    return (), line


@lru_cache(maxsize=None)
def _render_native_color_emoji(emoji: str) -> Image.Image:
    try:
        font = ImageFont.truetype(_EMOJI_FONT_NAME, _EMOJI_FONT_NATIVE_SIZE)
    except OSError as error:
        raise RuntimeError(
            "Noto Color Emoji is required to render label emoji. "
            "Install NotoColorEmoji.ttf so Pillow can find it."
        ) from error

    canvas = Image.new("RGBA", (160, 140), (0, 0, 0, 0))
    ImageDraw.Draw(canvas).text(
        (0, 0),
        emoji,
        font=font,
        embedded_color=True,
    )
    bounding_box = canvas.getbbox()
    if bounding_box is None:
        raise RuntimeError(f"Noto Color Emoji did not render emoji {emoji!r}")
    return canvas.crop(bounding_box)


@lru_cache(maxsize=512)
def _render_color_emoji(emoji: str, target_height: int) -> np.ndarray:
    if target_height <= 0:
        raise ValueError(f"Expected positive emoji height, got {target_height}")
    cropped = _render_native_color_emoji(emoji)
    target_width = max(1, round(cropped.width * target_height / cropped.height))
    resized = cropped.resize(
        (target_width, target_height),
        resample=Image.Resampling.LANCZOS,
    )
    return np.asarray(resized, dtype=np.uint8)


@lru_cache(maxsize=None)
def _render_native_flag_emoji(flag: str) -> Image.Image:
    if not _is_flag_emoji(flag):
        raise ValueError(f"Expected flag emoji, got {flag!r}")
    return _render_native_color_emoji(flag)


@lru_cache(maxsize=512)
def _render_flag_emoji(flag: str, target_height: int) -> np.ndarray:
    if target_height <= 0:
        raise ValueError(f"Expected positive flag height, got {target_height}")
    if not _is_flag_emoji(flag):
        raise ValueError(f"Expected flag emoji, got {flag!r}")
    return _render_color_emoji(flag, target_height)


def _split_trailing_label_emojis(line: str) -> tuple[str, tuple[str, ...]]:
    text = line.rstrip()
    trailing_emojis: list[str] = []
    while text:
        emoji = next(
            (candidate for candidate in TAG_EMOJI_ORDER if text.endswith(candidate)),
            None,
        )
        if emoji is None:
            break
        text = text[: -len(emoji)].rstrip()
        trailing_emojis.append(emoji)
    trailing_emojis.reverse()
    return text, tuple(trailing_emojis)


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
        flags, text = _split_leading_flags(line)
        text, trailing_emojis = _split_trailing_label_emojis(text)
        thickness = title_thickness if index == 0 else normal_thickness
        size, baseline = cv2.getTextSize(
            text,
            font,
            scale,
            thickness,
        )
        flag_widths = tuple(
            int(_render_flag_emoji(flag, max(1, int(size[1]))).shape[1])
            for flag in flags
        )
        trailing_emoji_widths = tuple(
            int(_render_color_emoji(emoji, max(1, int(size[1]))).shape[1])
            for emoji in trailing_emojis
        )
        layouts.append(
            _LabelLineLayout(
                text=text,
                flags=flags,
                text_size=(int(size[0]), int(size[1])),
                baseline=int(baseline),
                flag_widths=flag_widths,
                flag_gap_width=flag_gap_width,
                trailing_emojis=trailing_emojis,
                trailing_emoji_widths=trailing_emoji_widths,
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
        for flag, flag_width in zip(layout.flags, layout.flag_widths, strict=True):
            flag_image = _render_flag_emoji(flag, max(1, layout.text_size[1]))
            _alpha_composite_rgba(
                frame,
                flag_image,
                text_x,
                baseline_y - int(flag_image.shape[0]),
            )
            text_x += flag_width
        if layout.flags:
            text_x += layout.flag_gap_width
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
        text_x += layout.text_size[0]
        for emoji, emoji_width in zip(
            layout.trailing_emojis,
            layout.trailing_emoji_widths,
            strict=True,
        ):
            text_x += layout.flag_gap_width
            emoji_image = _render_color_emoji(
                emoji,
                max(1, layout.text_size[1]),
            )
            _alpha_composite_rgba(
                frame,
                emoji_image,
                text_x,
                baseline_y - int(emoji_image.shape[0]),
            )
            text_x += emoji_width
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
    scale = config.render.label_font_scale
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
