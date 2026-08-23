"""Small deterministic demo of the crop-row inspection loop."""

from __future__ import annotations

from .belief_memory import OdometryDelta, PerceptionInput
from .high_level_planner import TaskMode
from .inspection_system import CropRowInspectionSystem


def main() -> None:
    num_rows = 5
    system = CropRowInspectionSystem(num_rows=num_rows, initial_row=0)

    print("step,current_estimate,visited,target,mode,intent")
    step_index = 0
    while not system.belief.state.coverage_complete:
        decision = system.planner.decide(system.belief.state)
        inspection_complete = decision.task_mode == TaskMode.INSPECT_ROW
        if decision.target_row is None:
            row_shift = 0
        else:
            row_shift = decision.target_row - system.belief.state.estimated_row
            row_shift = max(-1, min(1, row_shift))

        likelihoods = tuple([1.0 / num_rows for _ in range(num_rows)])
        step = system.step(
            PerceptionInput(
                row_likelihoods=likelihoods,
                row_center_confidence=0.2,
                row_visible=True,
            ),
            OdometryDelta(row_shift=row_shift, forward_m=1.0, confidence=0.95),
            inspection_complete=inspection_complete,
        )
        print(
            f"{step_index},"
            f"{step.belief_state.estimated_row},"
            f"{sorted(step.belief_state.visited_rows)},"
            f"{step.decision.target_row},"
            f"{step.decision.task_mode.value},"
            f"{step.decision.motion_intent.value}"
        )
        step_index += 1


if __name__ == "__main__":
    main()
