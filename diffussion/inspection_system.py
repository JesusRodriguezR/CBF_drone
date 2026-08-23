"""End-to-end coordinator for crop-row inspection."""

from __future__ import annotations

from dataclasses import dataclass

from .belief_memory import BeliefMemory, BeliefState, OdometryDelta, PerceptionInput
from .high_level_planner import HighLevelPlanner, PlannerDecision, TaskMode
from .trajectory_generator import (
    DiffusionTrajectoryGenerator,
    GeneratorObservation,
    TrajectorySample,
    TrajectoryPoint,
)


@dataclass(frozen=True)
class InspectionStep:
    belief_state: BeliefState
    decision: PlannerDecision
    trajectory: list[TrajectoryPoint]
    trajectory_modes: list[TrajectorySample]


class CropRowInspectionSystem:
    """Wires belief, planning, and local trajectory generation."""

    def __init__(
        self,
        num_rows: int,
        *,
        initial_row: int | None = None,
        belief: BeliefMemory | None = None,
        planner: HighLevelPlanner | None = None,
        trajectory_generator: DiffusionTrajectoryGenerator | None = None,
    ) -> None:
        self.belief = belief or BeliefMemory(num_rows, initial_row=initial_row)
        self.planner = planner or HighLevelPlanner()
        self.trajectory_generator = trajectory_generator or DiffusionTrajectoryGenerator()

    def step(
        self,
        perception: PerceptionInput,
        odometry: OdometryDelta,
        *,
        inspection_complete: bool = False,
    ) -> InspectionStep:
        previous_decision = self.planner.decide(self.belief.state)
        inspected_row = (
            previous_decision.target_row
            if previous_decision.task_mode == TaskMode.INSPECT_ROW
            else None
        )
        belief_state = self.belief.update(
            perception,
            odometry,
            inspected_row=inspected_row,
            inspection_complete=inspection_complete,
        )
        decision = self.planner.decide(belief_state)
        generator_observation = GeneratorObservation(
            row_center_offset_m=0.0,
            heading_error_rad=0.0,
            obstacle_risk=perception.obstacle_risk,
            localization_confidence=belief_state.localization_confidence,
        )
        trajectory_modes = self.trajectory_generator.generate_modes(
            generator_observation,
            decision.motion_intent,
        )
        return InspectionStep(
            belief_state=belief_state,
            decision=decision,
            trajectory=trajectory_modes[0].trajectory,
            trajectory_modes=trajectory_modes,
        )
