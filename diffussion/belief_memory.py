"""Belief and memory module for row-index localization.

Rows can be visually identical, so this module treats vision as one cue rather
than an identity oracle. Topological odometry and inspection memory are kept as
first-class state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


def _normalize(values: list[float]) -> list[float]:
    total = sum(values)
    if total <= 0.0:
        uniform = 1.0 / len(values)
        return [uniform for _ in values]
    return [value / total for value in values]


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass(frozen=True)
class PerceptionInput:
    """Perception evidence for the current row.

    `row_likelihoods` may be uniform when rows look identical. The belief update
    still remains useful because the prediction step carries topological motion.
    """

    row_likelihoods: tuple[float, ...] | None = None
    row_center_confidence: float = 0.5
    row_visible: bool = True
    obstacle_risk: float = 0.0


@dataclass(frozen=True)
class OdometryDelta:
    """Topological and metric motion since the previous update."""

    row_shift: int = 0
    forward_m: float = 0.0
    lateral_m: float = 0.0
    confidence: float = 0.8


@dataclass(frozen=True)
class BeliefState:
    """State consumed by the high-level planner."""

    row_probabilities: tuple[float, ...]
    visited_rows: frozenset[int] = field(default_factory=frozenset)
    localization_confidence: float = 0.0
    last_observation_confidence: float = 0.0

    @property
    def num_rows(self) -> int:
        return len(self.row_probabilities)

    @property
    def estimated_row(self) -> int:
        return max(range(self.num_rows), key=lambda idx: self.row_probabilities[idx])

    @property
    def unvisited_rows(self) -> tuple[int, ...]:
        return tuple(idx for idx in range(self.num_rows) if idx not in self.visited_rows)

    @property
    def coverage_complete(self) -> bool:
        return len(self.visited_rows) == self.num_rows

    def probability_at(self, row: int) -> float:
        return self.row_probabilities[row]


class BeliefMemory:
    """Bayes-like row belief filter with persistent inspection memory."""

    def __init__(
        self,
        num_rows: int,
        *,
        initial_row: int | None = None,
        odometry_slip: float = 0.08,
    ) -> None:
        if num_rows <= 0:
            raise ValueError("num_rows must be positive")
        self.num_rows = num_rows
        self.odometry_slip = _clamp(odometry_slip, 0.0, 0.45)

        if initial_row is None:
            probabilities = [1.0 / num_rows for _ in range(num_rows)]
        else:
            if not 0 <= initial_row < num_rows:
                raise ValueError("initial_row is outside row range")
            probabilities = [0.0 for _ in range(num_rows)]
            probabilities[initial_row] = 1.0

        self._state = BeliefState(
            row_probabilities=tuple(probabilities),
            localization_confidence=max(probabilities),
        )

    @property
    def state(self) -> BeliefState:
        return self._state

    def update(
        self,
        perception: PerceptionInput,
        odometry: OdometryDelta,
        *,
        inspected_row: int | None = None,
        inspection_complete: bool = False,
    ) -> BeliefState:
        predicted = self._predict(self._state.row_probabilities, odometry)
        corrected = self._correct(predicted, perception)
        confidence = max(corrected) * _clamp(odometry.confidence, 0.0, 1.0)
        confidence *= 0.5 + 0.5 * _clamp(perception.row_center_confidence, 0.0, 1.0)

        visited = set(self._state.visited_rows)
        if inspection_complete:
            row = inspected_row if inspected_row is not None else max(
                range(self.num_rows),
                key=lambda idx: corrected[idx],
            )
            if 0 <= row < self.num_rows:
                visited.add(row)

        self._state = BeliefState(
            row_probabilities=tuple(corrected),
            visited_rows=frozenset(visited),
            localization_confidence=_clamp(confidence, 0.0, 1.0),
            last_observation_confidence=_clamp(perception.row_center_confidence, 0.0, 1.0),
        )
        return self._state

    def mark_visited(self, rows: Iterable[int]) -> BeliefState:
        visited = set(self._state.visited_rows)
        for row in rows:
            if not 0 <= row < self.num_rows:
                raise ValueError(f"row {row} is outside row range")
            visited.add(row)
        self._state = BeliefState(
            row_probabilities=self._state.row_probabilities,
            visited_rows=frozenset(visited),
            localization_confidence=self._state.localization_confidence,
            last_observation_confidence=self._state.last_observation_confidence,
        )
        return self._state

    def _predict(self, probabilities: tuple[float, ...], odometry: OdometryDelta) -> list[float]:
        shifted = [0.0 for _ in range(self.num_rows)]
        confidence = _clamp(odometry.confidence, 0.0, 1.0)
        stay_weight = 1.0 - self.odometry_slip * (1.0 - confidence)
        neighbor_weight = (1.0 - stay_weight) / 2.0

        for row, probability in enumerate(probabilities):
            target = _clamp(row + odometry.row_shift, 0, self.num_rows - 1)
            shifted[int(target)] += probability * stay_weight
            if target > 0:
                shifted[int(target) - 1] += probability * neighbor_weight
            else:
                shifted[int(target)] += probability * neighbor_weight
            if target < self.num_rows - 1:
                shifted[int(target) + 1] += probability * neighbor_weight
            else:
                shifted[int(target)] += probability * neighbor_weight

        return _normalize(shifted)

    def _correct(self, predicted: list[float], perception: PerceptionInput) -> list[float]:
        if not perception.row_visible or perception.row_likelihoods is None:
            return predicted
        if len(perception.row_likelihoods) != self.num_rows:
            raise ValueError("row_likelihoods length must match num_rows")

        likelihoods = _normalize([max(0.0, value) for value in perception.row_likelihoods])
        visual_weight = _clamp(perception.row_center_confidence, 0.0, 1.0)
        blended = []
        for prior, likelihood in zip(predicted, likelihoods):
            evidence = (1.0 - visual_weight) * (1.0 / self.num_rows) + visual_weight * likelihood
            blended.append(prior * evidence)
        return _normalize(blended)
