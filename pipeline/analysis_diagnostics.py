from __future__ import annotations

from collections.abc import Iterable, Mapping
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field
from typing import Any

from detectors.base import Detector

CONFIDENCE_BINS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01)
BOX_WIDTH_BINS = (
    0.0,
    40.0,
    80.0,
    120.0,
    160.0,
    200.0,
    300.0,
    400.0,
    600.0,
    800.0,
    float("inf"),
)


@dataclass(slots=True)
class HistogramAccumulator:
    bins: tuple[float, ...]
    counts: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.counts:
            self.counts = [0 for _ in range(max(0, len(self.bins) - 1))]

    def observe(self, value: float) -> None:
        for index, (lower, upper) in enumerate(
            zip(self.bins, self.bins[1:], strict=True)
        ):
            if lower <= value < upper or (
                index == len(self.counts) - 1 and value == upper
            ):
                self.counts[index] += 1
                break

    def extend(self, values: Iterable[float]) -> None:
        for value in values:
            self.observe(value)

    def payload(self) -> list[dict[str, Any]]:
        return [
            {
                "min": lower,
                "max": upper if upper != float("inf") else None,
                "count": count,
            }
            for lower, upper, count in zip(
                self.bins[:-1], self.bins[1:], self.counts, strict=True
            )
        ]


@dataclass(slots=True)
class AnalysisDiagnostics:
    total_sampled_frames: int = 0
    detections_passed_to_tracker: int = 0
    tracker_outputs: int = 0
    tracks_discarded_edge_contact: int = 0
    edge_observations_skipped: int = 0
    tracks_discarded_min_track_frames: int = 0
    tracks_without_crop_candidates: int = 0
    tracks_without_crop_due_to_width: int = 0
    tracks_without_crop_due_to_short_lifetime: int = 0
    tracks_hidden_from_render_crop_eligibility: int = 0
    duplicate_track_observations_suppressed: int = 0
    duplicate_track_ids_dropped: int = 0
    duplicate_track_suppression_blocked_counted: int = 0
    stale_reassociation_observations_suppressed: int = 0
    stale_reassociation_track_ids_dropped: int = 0
    tracker_confidence_histogram: HistogramAccumulator = field(
        default_factory=lambda: HistogramAccumulator(CONFIDENCE_BINS)
    )
    tracker_box_width_histogram: HistogramAccumulator = field(
        default_factory=lambda: HistogramAccumulator(BOX_WIDTH_BINS)
    )


def histogram(
    values: SequenceABC[float], bins: SequenceABC[float]
) -> list[dict[str, Any]]:
    counts = [0 for _ in range(max(0, len(bins) - 1))]
    for value in values:
        for index, (lower, upper) in enumerate(zip(bins, bins[1:], strict=True)):
            if lower <= value < upper or (index == len(counts) - 1 and value == upper):
                counts[index] += 1
                break
    return [
        {
            "min": lower,
            "max": upper if upper != float("inf") else None,
            "count": count,
        }
        for lower, upper, count in zip(bins[:-1], bins[1:], counts, strict=True)
    ]


def detector_diagnostics(detector: Detector) -> dict[str, Any]:
    snapshot = getattr(detector, "detection_diagnostics", None)
    if not callable(snapshot):
        return {}
    raw = snapshot()
    return raw if isinstance(raw, dict) else {}


def diagnostic_count(
    detector_counts: Mapping[str, object], key: str, fallback: int
) -> int:
    value = detector_counts.get(key)
    return int(value) if isinstance(value, int | float) else fallback


def diagnostic_float_values(value: object) -> list[float]:
    if not isinstance(value, list):
        return []
    return [float(item) for item in value if isinstance(item, int | float)]


def analysis_diagnostics_payload(
    diagnostics: AnalysisDiagnostics, detector: Detector
) -> dict[str, Any]:
    detector_snapshot = detector_diagnostics(detector)
    detector_counts_raw = detector_snapshot.get("counts", {})
    detector_counts = (
        detector_counts_raw if isinstance(detector_counts_raw, Mapping) else {}
    )
    detector_confidences = diagnostic_float_values(
        detector_snapshot.get("confidence_values")
    )
    detector_confidence_histogram = detector_snapshot.get("confidence_histogram")
    confidence_histogram = (
        detector_confidence_histogram
        if isinstance(detector_confidence_histogram, list)
        else (
            histogram(detector_confidences, CONFIDENCE_BINS)
            if detector_confidences
            else diagnostics.tracker_confidence_histogram.payload()
        )
    )

    return {
        "total_sampled_frames": diagnostics.total_sampled_frames,
        "raw_detections_before_class_filtering": diagnostic_count(
            detector_counts,
            "raw_candidate_rows",
            diagnostics.detections_passed_to_tracker,
        ),
        "detections_after_confidence_filtering": diagnostic_count(
            detector_counts,
            "detections_after_confidence_filtering",
            diagnostics.detections_passed_to_tracker,
        ),
        "detections_after_class_filtering": diagnostic_count(
            detector_counts,
            "detections_after_class_filtering",
            diagnostics.detections_passed_to_tracker,
        ),
        "detections_after_nms": diagnostic_count(
            detector_counts,
            "detections_after_nms",
            diagnostics.detections_passed_to_tracker,
        ),
        "detections_suppressed_by_nms": diagnostic_count(
            detector_counts,
            "detections_suppressed_by_nms",
            0,
        ),
        "detections_passed_to_tracker": diagnostics.detections_passed_to_tracker,
        "tracker_outputs": diagnostics.tracker_outputs,
        "tracks_discarded_edge_contact": diagnostics.tracks_discarded_edge_contact,
        "edge_observations_skipped": diagnostics.edge_observations_skipped,
        "duplicate_track_observations_suppressed": (
            diagnostics.duplicate_track_observations_suppressed
        ),
        "duplicate_track_ids_dropped": diagnostics.duplicate_track_ids_dropped,
        "duplicate_track_suppression_blocked_counted": (
            diagnostics.duplicate_track_suppression_blocked_counted
        ),
        "stale_reassociation_observations_suppressed": (
            diagnostics.stale_reassociation_observations_suppressed
        ),
        "stale_reassociation_track_ids_dropped": (
            diagnostics.stale_reassociation_track_ids_dropped
        ),
        "tracks_discarded_min_track_frames": diagnostics.tracks_discarded_min_track_frames,
        "tracks_without_crop_candidates": diagnostics.tracks_without_crop_candidates,
        "tracks_without_crop_due_to_width": (
            diagnostics.tracks_without_crop_due_to_width
        ),
        "tracks_without_crop_due_to_short_lifetime": (
            diagnostics.tracks_without_crop_due_to_short_lifetime
        ),
        "tracks_hidden_from_render_due_to_crop_eligibility": (
            diagnostics.tracks_hidden_from_render_crop_eligibility
        ),
        "confidence_histogram": confidence_histogram,
        "box_width_histogram": diagnostics.tracker_box_width_histogram.payload(),
        "detector": detector_snapshot,
    }
