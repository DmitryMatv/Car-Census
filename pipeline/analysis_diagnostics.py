from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field
from typing import Any

from detectors.base import Detector


@dataclass(slots=True)
class AnalysisDiagnostics:
    total_sampled_frames: int = 0
    detections_passed_to_tracker: int = 0
    tracker_outputs: int = 0
    tracks_discarded_edge_contact: int = 0
    edge_observations_skipped: int = 0
    tracks_discarded_min_track_frames: int = 0
    tracks_without_crop_candidates: int = 0
    tracks_without_crop_due_to_height: int = 0
    tracks_without_crop_due_to_short_lifetime: int = 0
    tracks_hidden_from_render_crop_eligibility: int = 0
    tracker_confidence_values: list[float] = field(default_factory=list)
    tracker_box_height_values: list[float] = field(default_factory=list)


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
    confidence_values = detector_confidences or diagnostics.tracker_confidence_values

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
        "detections_passed_to_tracker": diagnostics.detections_passed_to_tracker,
        "tracker_outputs": diagnostics.tracker_outputs,
        "tracks_discarded_edge_contact": diagnostics.tracks_discarded_edge_contact,
        "edge_observations_skipped": diagnostics.edge_observations_skipped,
        "tracks_discarded_min_track_frames": diagnostics.tracks_discarded_min_track_frames,
        "tracks_without_crop_candidates": diagnostics.tracks_without_crop_candidates,
        "tracks_without_crop_due_to_height": (
            diagnostics.tracks_without_crop_due_to_height
        ),
        "tracks_without_crop_due_to_short_lifetime": (
            diagnostics.tracks_without_crop_due_to_short_lifetime
        ),
        "tracks_hidden_from_render_due_to_crop_eligibility": (
            diagnostics.tracks_hidden_from_render_crop_eligibility
        ),
        "confidence_histogram": histogram(
            confidence_values,
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01],
        ),
        "box_height_histogram": histogram(
            diagnostics.tracker_box_height_values,
            [
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
            ],
        ),
        "detector": detector_snapshot,
    }
