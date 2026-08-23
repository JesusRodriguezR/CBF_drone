"""Local diffusion-style trajectory generation.

This module provides a deterministic conditional sampler that mirrors the role
of a diffusion trajectory policy: sample multiple local futures, score them, and
return a feasible path. A trained denoising model can replace `_sample_candidate`
without changing the planner interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import random


class MotionIntent(str, Enum):
    FOLLOW_ROW = "follow_row"
    SHIFT_LEFT = "shift_left"
    SHIFT_RIGHT = "shift_right"
    HOLD = "hold"


@dataclass(frozen=True)
class TrajectoryConfig:
    horizon_s: float = 2.0
    dt_s: float = 0.2
    row_spacing_m: float = 0.75
    cruise_speed_mps: float = 0.6
    max_lateral_speed_mps: float = 0.45
    max_yaw_rate_radps: float = 0.9
    candidate_count: int = 32
    noise_scale_m: float = 0.03


@dataclass(frozen=True)
class TrajectoryPoint:
    t_s: float
    x_m: float
    y_m: float
    yaw_rad: float
    vx_mps: float
    vy_mps: float


@dataclass(frozen=True)
class ObservationHypothesis:
    """One local interpretation of an ambiguous visual observation."""

    row_center_offset_m: float = 0.0
    heading_error_rad: float = 0.0
    weight: float = 1.0


@dataclass(frozen=True)
class GeneratorObservation:
    """Local-only conditioning for trajectory generation.

    This deliberately contains no row index, no target row, and no visited-row
    memory. Those are mission-level planner responsibilities.
    """

    row_center_offset_m: float = 0.0
    heading_error_rad: float = 0.0
    hypotheses: tuple[ObservationHypothesis, ...] = ()
    obstacle_risk: float = 0.0
    localization_confidence: float = 1.0


@dataclass(frozen=True)
class TrajectorySample:
    trajectory: list[TrajectoryPoint]
    score: float
    hypothesis_index: int


class DiffusionTrajectoryGenerator:
    """Generates local, smooth, dynamically bounded trajectories."""

    def __init__(self, config: TrajectoryConfig | None = None, *, seed: int = 7) -> None:
        self.config = config or TrajectoryConfig()
        self._rng = random.Random(seed)
        if self.config.dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        if self.config.horizon_s <= 0.0:
            raise ValueError("horizon_s must be positive")

    def generate(
        self,
        observation: GeneratorObservation,
        intent: MotionIntent,
    ) -> list[TrajectoryPoint]:
        """Return the best local trajectory for the current observation and intent."""

        return self.generate_modes(observation, intent, mode_count=1)[0].trajectory

    def generate_modes(
        self,
        observation: GeneratorObservation,
        intent: MotionIntent,
        *,
        mode_count: int = 3,
    ) -> list[TrajectorySample]:
        """Return diverse local trajectory modes for a fixed planner intent.

        Multi-modality is local: alternate row-center or heading interpretations
        may produce different feasible paths, but the generator never selects a
        mission row or updates coverage progress.
        """

        if mode_count <= 0:
            raise ValueError("mode_count must be positive")

        hypotheses = self._expanded_hypotheses(observation)
        candidates = []
        samples_per_hypothesis = max(
            1,
            math.ceil(max(1, self.config.candidate_count) / len(hypotheses)),
        )
        for hypothesis_index, hypothesis in enumerate(hypotheses):
            local_observation = GeneratorObservation(
                row_center_offset_m=hypothesis.row_center_offset_m,
                heading_error_rad=hypothesis.heading_error_rad,
                obstacle_risk=observation.obstacle_risk,
                localization_confidence=observation.localization_confidence,
            )
            for _ in range(samples_per_hypothesis):
                trajectory = self._sample_candidate(local_observation, intent)
                score = self._score(
                    trajectory,
                    local_observation,
                    intent,
                    hypothesis_weight=hypothesis.weight,
                )
                candidates.append(
                    TrajectorySample(
                        trajectory=trajectory,
                        score=score,
                        hypothesis_index=hypothesis_index,
                    )
                )

        ranked = sorted(candidates, key=lambda sample: sample.score)
        return self._select_diverse_modes(ranked, mode_count)

    def _sample_candidate(
        self,
        observation: GeneratorObservation,
        intent: MotionIntent,
    ) -> list[TrajectoryPoint]:
        steps = max(2, int(round(self.config.horizon_s / self.config.dt_s)) + 1)
        target_y = self._target_lateral_offset(intent)
        target_x = 0.0 if intent == MotionIntent.HOLD else self.config.cruise_speed_mps * self.config.horizon_s

        confidence = max(0.0, min(1.0, observation.localization_confidence))
        noise_scale = self.config.noise_scale_m * (1.2 - confidence)
        points: list[TrajectoryPoint] = []
        previous_x = 0.0
        previous_y = observation.row_center_offset_m
        previous_yaw = observation.heading_error_rad

        for step in range(steps):
            tau = step / (steps - 1)
            smooth = self._smoothstep(tau)
            t_s = step * self.config.dt_s
            x_m = target_x * tau
            y_m = observation.row_center_offset_m + (target_y - observation.row_center_offset_m) * smooth
            yaw_rad = observation.heading_error_rad * (1.0 - smooth)

            if 0 < step < steps - 1:
                x_m += self._rng.gauss(0.0, noise_scale)
                y_m += self._rng.gauss(0.0, noise_scale)
                yaw_rad += self._rng.gauss(0.0, noise_scale * 0.5)

            if step == 0:
                vx_mps = 0.0 if intent == MotionIntent.HOLD else self.config.cruise_speed_mps
                vy_mps = 0.0
            else:
                vx_mps = (x_m - previous_x) / self.config.dt_s
                vy_mps = (y_m - previous_y) / self.config.dt_s
                yaw_rate = (yaw_rad - previous_yaw) / self.config.dt_s
                yaw_rad = previous_yaw + max(
                    -self.config.max_yaw_rate_radps,
                    min(self.config.max_yaw_rate_radps, yaw_rate),
                ) * self.config.dt_s
                vy_mps = max(
                    -self.config.max_lateral_speed_mps,
                    min(self.config.max_lateral_speed_mps, vy_mps),
                )
                y_m = previous_y + vy_mps * self.config.dt_s

            previous_x = x_m
            previous_y = y_m
            previous_yaw = yaw_rad
            points.append(
                TrajectoryPoint(
                    t_s=t_s,
                    x_m=x_m,
                    y_m=y_m,
                    yaw_rad=yaw_rad,
                    vx_mps=vx_mps,
                    vy_mps=vy_mps,
                )
            )

        return points

    def _target_lateral_offset(self, intent: MotionIntent) -> float:
        if intent == MotionIntent.SHIFT_LEFT:
            return -self.config.row_spacing_m
        if intent == MotionIntent.SHIFT_RIGHT:
            return self.config.row_spacing_m
        return 0.0

    def _score(
        self,
        trajectory: list[TrajectoryPoint],
        observation: GeneratorObservation,
        intent: MotionIntent,
        *,
        hypothesis_weight: float = 1.0,
    ) -> float:
        target_y = self._target_lateral_offset(intent)
        endpoint = trajectory[-1]
        endpoint_error = abs(endpoint.y_m - target_y) + abs(endpoint.yaw_rad)
        smoothness = 0.0
        limit_penalty = 0.0

        for prev, point in zip(trajectory, trajectory[1:]):
            dvx = point.vx_mps - prev.vx_mps
            dvy = point.vy_mps - prev.vy_mps
            smoothness += dvx * dvx + dvy * dvy
            if abs(point.vy_mps) > self.config.max_lateral_speed_mps + 1e-9:
                limit_penalty += 10.0

        obstacle_penalty = observation.obstacle_risk * sum(
            math.hypot(point.x_m, point.y_m) for point in trajectory
        )
        likelihood_bonus = -math.log(max(hypothesis_weight, 1e-6))
        return endpoint_error * 8.0 + smoothness + limit_penalty + obstacle_penalty + likelihood_bonus

    def _expanded_hypotheses(
        self,
        observation: GeneratorObservation,
    ) -> tuple[ObservationHypothesis, ...]:
        if observation.hypotheses:
            total_weight = sum(max(0.0, hypothesis.weight) for hypothesis in observation.hypotheses)
            if total_weight <= 0.0:
                uniform = 1.0 / len(observation.hypotheses)
                return tuple(
                    ObservationHypothesis(
                        row_center_offset_m=hypothesis.row_center_offset_m,
                        heading_error_rad=hypothesis.heading_error_rad,
                        weight=uniform,
                    )
                    for hypothesis in observation.hypotheses
                )
            return tuple(
                ObservationHypothesis(
                    row_center_offset_m=hypothesis.row_center_offset_m,
                    heading_error_rad=hypothesis.heading_error_rad,
                    weight=max(0.0, hypothesis.weight) / total_weight,
                )
                for hypothesis in observation.hypotheses
            )

        return (
            ObservationHypothesis(
                row_center_offset_m=observation.row_center_offset_m,
                heading_error_rad=observation.heading_error_rad,
                weight=1.0,
            ),
        )

    def _select_diverse_modes(
        self,
        ranked_samples: list[TrajectorySample],
        mode_count: int,
    ) -> list[TrajectorySample]:
        selected: list[TrajectorySample] = []
        for sample in ranked_samples:
            if not selected:
                selected.append(sample)
                continue
            if len(selected) >= mode_count:
                break
            if self._is_distinct_mode(sample, selected):
                selected.append(sample)

        for sample in ranked_samples:
            if len(selected) >= mode_count:
                break
            if sample not in selected:
                selected.append(sample)
        return selected

    def _is_distinct_mode(
        self,
        sample: TrajectorySample,
        selected: list[TrajectorySample],
    ) -> bool:
        endpoint = sample.trajectory[-1]
        start = sample.trajectory[0]
        for existing in selected:
            existing_endpoint = existing.trajectory[-1]
            existing_start = existing.trajectory[0]
            endpoint_distance = abs(endpoint.y_m - existing_endpoint.y_m)
            start_distance = abs(start.y_m - existing_start.y_m)
            if endpoint_distance < 0.15 and start_distance < 0.15:
                return False
        return True

    @staticmethod
    def _smoothstep(value: float) -> float:
        return value * value * (3.0 - 2.0 * value)
