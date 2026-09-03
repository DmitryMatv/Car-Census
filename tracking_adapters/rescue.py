"""Rescue layer for tracks whose IoU association failed.

When BoT-SORT spawns a fresh tracklet for a detection that no existing track
claimed (the "car reappears after a gap with a new ID" failure), the rescue
layer predicts the missing track's position from its own recent trajectory
and takes the identity over when the handoff is physically plausible.

Two operating modes exist:

- **World mode** (a calibrated camera homography is available): positions
  are road-plane meters, the prediction assumes constant world velocity
  (the camera is static), and displacement/lateral/longitudinal gates run
  in meters. Appearance evidence optionally confirms the takeover.
- **Pixel mode** (no homography, ``pixel_fallback_enabled``): positions are
  image-space bottom-centers. Displacement limits are expressed in
  candidate box-heights so they scale with object size instead of raw
  pixels. The appearance gate is MANDATORY with a stricter floor, because
  with no road-plane geometry the embedding similarity is the only
  identity evidence.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from config import RescueConfig
from reid import TrackAppearanceMemory
from roi.transform import ViewTransformer

MODE_WORLD = "world"
MODE_PIXEL = "pixel"


@dataclass
class _TrajectoryPoint:
    """A track observation. World mode stores meters, pixel mode pixels."""

    timestamp: float
    x: float
    y: float


@dataclass
class RescueMatch:
    """A successful takeover proposal for a fresh spawn."""

    old_track_id: int
    mode: str
    gap_seconds: float
    distance: float
    lateral: float | None
    implied_speed: float
    appearance_similarity: float | None


@dataclass
class RescueRejection:
    """A rejected takeover proposal with the failing metrics for the audit."""

    old_track_id: int
    mode: str
    gap_seconds: float
    distance: float
    lateral: float | None
    implied_speed: float
    appearance_similarity: float | None
    reason: str


@dataclass
class _Trajectory:
    points: deque[_TrajectoryPoint] = field(default_factory=deque)

    def velocity(self, max_points: int) -> tuple[float, float] | None:
        """Least-squares velocity ``(vx, vy)`` in trajectory units per second.

        Returns ``None`` when fewer than two distinct-time points exist.
        """
        recent = list(self.points)[-max_points:]
        if len(recent) < 2:
            return None
        t0 = recent[0].timestamp
        times = np.array([p.timestamp - t0 for p in recent], dtype=np.float64)
        xs = np.array([p.x for p in recent], dtype=np.float64)
        ys = np.array([p.y for p in recent], dtype=np.float64)
        span = float(times[-1] - times[0])
        if span <= 0.0:
            return None
        # Least-squares slope through the origin of the centered window.
        t_mean = float(times.mean())
        denom = float(((times - t_mean) ** 2).sum())
        if denom <= 0.0:
            return None
        vx = float(((times - t_mean) * (xs - xs.mean())).sum() / denom)
        vy = float(((times - t_mean) * (ys - ys.mean())).sum() / denom)
        return vx, vy


class RescueEngine:
    """Tracks trajectories and proposes identity takeovers."""

    def __init__(
        self,
        config: RescueConfig,
        view_transformer: ViewTransformer | None,
    ) -> None:
        self.config = config
        self.view_transformer = view_transformer
        self.trajectories: dict[int, _Trajectory] = {}
        self.events: list[dict[str, Any]] = []

    @property
    def active(self) -> bool:
        if not self.config.enabled:
            return False
        return self.view_transformer is not None or self.config.pixel_fallback_enabled

    @property
    def mode(self) -> str:
        return MODE_WORLD if self.view_transformer is not None else MODE_PIXEL

    @property
    def world_gate_active(self) -> bool:
        """True when the engine runs on a calibrated homography."""
        return self.config.enabled and self.view_transformer is not None

    @property
    def pixel_fallback_active(self) -> bool:
        """True when the engine runs in image space without a homography."""
        return (
            self.config.enabled
            and self.view_transformer is None
            and self.config.pixel_fallback_enabled
        )

    def observe(
        self,
        track_id: int,
        timestamp: float,
        bottom_center: tuple[float, float],
    ) -> None:
        if not self.active:
            return
        if not all(math.isfinite(value) for value in bottom_center):
            return
        self._prune(timestamp)
        x, y = bottom_center
        if self.view_transformer is not None:
            world = self.view_transformer.transform_point(bottom_center)
            if not all(math.isfinite(value) for value in world):
                return
            x, y = world
        self.trajectories.setdefault(track_id, _Trajectory()).points.append(
            _TrajectoryPoint(timestamp, x, y)
        )

    def forget(self, track_id: int) -> None:
        self.trajectories.pop(track_id, None)

    def _prune(self, now: float) -> None:
        """Drop trajectories too old to ever be a takeover source.

        Without pruning, every trajectory ever seen stays in memory for the
        whole video and is re-evaluated (and re-rejected as ``gap_exceeded``)
        against every fresh spawn, inflating the audit and memory on long
        runs. A trajectory whose last observation is older than the rescue
        gap ceiling can no longer be matched, so it is removed.
        """
        cutoff = now - self.config.max_gap_seconds
        stale = [
            track_id
            for track_id, trajectory in self.trajectories.items()
            if trajectory.points and trajectory.points[-1].timestamp < cutoff
        ]
        for track_id in stale:
            del self.trajectories[track_id]

    def match(
        self,
        candidate_bbox: tuple[float, float, float, float],
        candidate_time: float,
        busy_ids: set[int],
        memory: TrackAppearanceMemory | None = None,
        candidate_vector: np.ndarray | None = None,
        min_appearance_similarity: float | None = None,
    ) -> tuple[RescueMatch | None, list[RescueRejection]]:
        """Find the best old track for a fresh spawn, or None.

        ``busy_ids`` lists track IDs already seen in this frame's output —
        those tracks are alive and matched, so they are never takeover
        sources. ``memory``/``candidate_vector`` carry the appearance
        evidence; in world mode it optionally confirms a geometry-plausible
        handoff against ``min_appearance_similarity``, in pixel mode it is
        mandatory against ``pixel_fallback_min_appearance_similarity``.
        Returns the accepted match (possibly None) plus every evaluated
        rejection for the audit trail.
        """
        rejections: list[RescueRejection] = []
        if not self.active:
            return None, rejections
        if not all(math.isfinite(value) for value in candidate_bbox):
            return None, rejections
        candidate_point = _bottom_center(candidate_bbox)
        if self.view_transformer is not None:
            projected = self.view_transformer.transform_point(candidate_point)
            if not all(math.isfinite(value) for value in projected):
                return None, rejections
            candidate_point = projected

        best: RescueMatch | None = None
        best_key: tuple[float, float] | None = None
        for old_id, trajectory in self.trajectories.items():
            if old_id in busy_ids:
                continue
            decision = self._evaluate(
                old_id,
                trajectory,
                candidate_point,
                candidate_time,
                candidate_bbox,
                memory,
                candidate_vector,
                min_appearance_similarity,
            )
            if isinstance(decision, RescueRejection):
                rejections.append(decision)
                continue
            key = (decision.distance, -(decision.appearance_similarity or 0.0))
            if best_key is None or key < best_key:
                best_key = key
                best = decision
        return best, rejections

    def _evaluate(
        self,
        old_id: int,
        trajectory: _Trajectory,
        candidate_point: tuple[float, float],
        candidate_time: float,
        candidate_bbox: tuple[float, float, float, float],
        memory: TrackAppearanceMemory | None,
        candidate_vector: np.ndarray | None,
        min_appearance_similarity: float | None,
    ) -> RescueMatch | RescueRejection:
        config = self.config
        mode = self.mode
        points = trajectory.points
        last = points[-1]
        gap_seconds = candidate_time - last.timestamp
        metrics: dict[str, Any] = {
            "gap_seconds": gap_seconds,
            "distance": math.inf,
            "lateral": None,
            "implied_speed": math.inf,
            "appearance_similarity": None,
            "mode": mode,
        }
        if gap_seconds <= 0.0:
            return self._reject(old_id, metrics, "not_after_last_observation")
        if gap_seconds > config.max_gap_seconds:
            return self._reject(old_id, metrics, "gap_exceeded")
        if mode == MODE_PIXEL:
            return self._evaluate_pixel(
                old_id,
                trajectory,
                candidate_point,
                candidate_bbox,
                gap_seconds,
                metrics,
                memory,
                candidate_vector,
            )
        velocity = trajectory.velocity(config.velocity_fit_points)
        if velocity is None:
            return self._reject(old_id, metrics, "insufficient_history")
        predicted = (
            last.x + velocity[0] * gap_seconds,
            last.y + velocity[1] * gap_seconds,
        )
        dx = candidate_point[0] - predicted[0]
        dy = candidate_point[1] - predicted[1]
        distance = math.hypot(dx, dy)
        implied_speed = (
            math.hypot(candidate_point[0] - last.x, candidate_point[1] - last.y)
            / gap_seconds
        )
        metrics["distance"] = distance
        metrics["implied_speed"] = implied_speed
        if implied_speed > config.max_speed_mps:
            return self._reject(old_id, metrics, "speed_exceeded")
        if distance > config.max_distance_m:
            return self._reject(old_id, metrics, "distance_exceeded")
        speed = math.hypot(*velocity)
        if speed >= config.min_direction_speed_mps:
            axis = (velocity[0] / speed, velocity[1] / speed)
            longitudinal = dx * axis[0] + dy * axis[1]
            lateral = abs(dx * -axis[1] + dy * axis[0])
            metrics["lateral"] = float(lateral)
            if lateral > config.lateral_tolerance_m:
                return self._reject(old_id, metrics, "lateral_exceeded")
            if longitudinal < -config.max_behind_prediction_m:
                return self._reject(old_id, metrics, "behind_prediction")

        similarity = _appearance_similarity(old_id, memory, candidate_vector)
        metrics["appearance_similarity"] = (
            float(similarity) if similarity is not None else None
        )
        if (
            similarity is not None
            and min_appearance_similarity is not None
            and similarity < min_appearance_similarity
        ):
            return self._reject(old_id, metrics, "appearance_mismatch")
        return RescueMatch(
            old_track_id=old_id,
            mode=mode,
            gap_seconds=gap_seconds,
            distance=distance,
            lateral=metrics["lateral"],
            implied_speed=implied_speed,
            appearance_similarity=similarity,
        )

    def _evaluate_pixel(
        self,
        old_id: int,
        trajectory: _Trajectory,
        candidate_point: tuple[float, float],
        candidate_bbox: tuple[float, float, float, float],
        gap_seconds: float,
        metrics: dict[str, Any],
        memory: TrackAppearanceMemory | None,
        candidate_vector: np.ndarray | None,
    ) -> RescueMatch | RescueRejection:
        """Pixel-mode gates: box-height-scaled displacement plus a mandatory
        appearance gate. Pixel velocity fits are dominated by image-scale
        change (a car approaching the camera grows without moving on the
        road), so the world-mode direction gates have no image-space analogue
        and are intentionally skipped."""
        if len(trajectory.points) < 2:
            return self._reject(old_id, metrics, "insufficient_history")
        candidate_height = candidate_bbox[3] - candidate_bbox[1]
        if candidate_height <= 0.0 or not math.isfinite(candidate_height):
            return self._reject(old_id, metrics, "degenerate_scale")
        last = trajectory.points[-1]
        displacement = math.hypot(
            candidate_point[0] - last.x, candidate_point[1] - last.y
        )
        implied_speed = displacement / gap_seconds
        metrics["distance"] = displacement
        metrics["implied_speed"] = implied_speed
        max_speed = self.config.pixel_max_speed_box_heights_per_s * candidate_height
        if implied_speed > max_speed:
            return self._reject(old_id, metrics, "speed_exceeded")
        allowed_distance = candidate_height * min(
            self.config.pixel_max_distance_box_heights,
            self.config.pixel_max_speed_box_heights_per_s * gap_seconds,
        )
        if displacement > allowed_distance:
            return self._reject(old_id, metrics, "distance_exceeded")

        similarity = _appearance_similarity(old_id, memory, candidate_vector)
        metrics["appearance_similarity"] = (
            float(similarity) if similarity is not None else None
        )
        if similarity is None:
            return self._reject(old_id, metrics, "appearance_required")
        if similarity < self.config.pixel_fallback_min_appearance_similarity:
            return self._reject(old_id, metrics, "appearance_mismatch")
        return RescueMatch(
            old_track_id=old_id,
            mode=MODE_PIXEL,
            gap_seconds=gap_seconds,
            distance=displacement,
            lateral=None,
            implied_speed=implied_speed,
            appearance_similarity=similarity,
        )

    def _reject(
        self, old_id: int, metrics: dict[str, Any], reason: str
    ) -> RescueRejection:
        return RescueRejection(
            old_track_id=old_id,
            mode=metrics["mode"],
            gap_seconds=float(metrics["gap_seconds"]),
            distance=float(metrics["distance"]),
            lateral=(
                float(metrics["lateral"]) if metrics["lateral"] is not None else None
            ),
            implied_speed=float(metrics["implied_speed"]),
            appearance_similarity=metrics["appearance_similarity"],
            reason=reason,
        )


def _appearance_similarity(
    old_id: int,
    memory: TrackAppearanceMemory | None,
    candidate_vector: np.ndarray | None,
) -> float | None:
    if memory is None or candidate_vector is None:
        return None
    similarity = memory.similarity(old_id, candidate_vector)
    return float(similarity) if similarity is not None else None


def _bottom_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, _y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)
