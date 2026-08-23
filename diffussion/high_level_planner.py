"""High-level row coverage planner.

The planner owns global inspection decisions. The trajectory generator only
receives local motion intent derived from these decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .belief_memory import BeliefState
from .trajectory_generator import MotionIntent


class TaskMode(str, Enum):
    INSPECT_ROW = "inspect_row"
    TRANSITION_BETWEEN_ROWS = "transition_between_rows"
    COMPLETE = "complete"


@dataclass(frozen=True)
class PlannerConfig:
    inspect_confidence_threshold: float = 0.55
    transition_confidence_threshold: float = 0.35


@dataclass(frozen=True)
class PlannerDecision:
    target_row: int | None
    task_mode: TaskMode
    motion_intent: MotionIntent
    reason: str


class HighLevelPlanner:
    """Chooses the next unvisited row without redundant revisits."""

    def __init__(self, config: PlannerConfig | None = None) -> None:
        self.config = config or PlannerConfig()

    def decide(self, belief_state: BeliefState) -> PlannerDecision:
        if belief_state.coverage_complete:
            return PlannerDecision(
                target_row=None,
                task_mode=TaskMode.COMPLETE,
                motion_intent=MotionIntent.HOLD,
                reason="all rows inspected",
            )

        current = belief_state.estimated_row
        current_probability = belief_state.probability_at(current)

        if (
            current not in belief_state.visited_rows
            and current_probability >= self.config.inspect_confidence_threshold
        ):
            return PlannerDecision(
                target_row=current,
                task_mode=TaskMode.INSPECT_ROW,
                motion_intent=MotionIntent.FOLLOW_ROW,
                reason="current row is unvisited and localized",
            )

        target = self._nearest_unvisited_row(belief_state, current)
        intent = self._transition_intent(current, target)
        return PlannerDecision(
            target_row=target,
            task_mode=TaskMode.TRANSITION_BETWEEN_ROWS,
            motion_intent=intent,
            reason="move to nearest unvisited row",
        )

    def _nearest_unvisited_row(self, belief_state: BeliefState, current: int) -> int:
        return min(
            belief_state.unvisited_rows,
            key=lambda row: (abs(row - current), row),
        )

    @staticmethod
    def _transition_intent(current: int, target: int) -> MotionIntent:
        if target < current:
            return MotionIntent.SHIFT_LEFT
        if target > current:
            return MotionIntent.SHIFT_RIGHT
        return MotionIntent.FOLLOW_ROW
