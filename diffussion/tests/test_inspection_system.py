import math
from dataclasses import fields

from diffussion.belief_memory import BeliefMemory, OdometryDelta, PerceptionInput
from diffussion.high_level_planner import HighLevelPlanner, TaskMode
from diffussion.trajectory_generator import (
    DiffusionTrajectoryGenerator,
    GeneratorObservation,
    MotionIntent,
    ObservationHypothesis,
    TrajectoryConfig,
)


def test_planner_covers_all_rows_without_redundant_revisits():
    memory = BeliefMemory(num_rows=4, initial_row=0)
    planner = HighLevelPlanner()
    inspected = []

    for _ in range(12):
        decision = planner.decide(memory.state)
        if decision.task_mode == TaskMode.COMPLETE:
            break

        if decision.task_mode == TaskMode.INSPECT_ROW:
            assert decision.target_row not in inspected
            inspected.append(decision.target_row)
            row_shift = 0
            inspection_complete = True
        else:
            current = memory.state.estimated_row
            row_shift = max(-1, min(1, decision.target_row - current))
            inspection_complete = False

        memory.update(
            PerceptionInput(row_likelihoods=(0.25, 0.25, 0.25, 0.25), row_center_confidence=0.2),
            OdometryDelta(row_shift=row_shift, confidence=0.98),
            inspected_row=decision.target_row if inspection_complete else None,
            inspection_complete=inspection_complete,
        )

    assert inspected == [0, 1, 2, 3]
    assert memory.state.coverage_complete


def test_identical_visual_rows_still_localize_from_odometry():
    memory = BeliefMemory(num_rows=3, initial_row=0)
    identical_visual_likelihood = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

    memory.update(
        PerceptionInput(row_likelihoods=identical_visual_likelihood, row_center_confidence=0.1),
        OdometryDelta(row_shift=1, confidence=0.99),
    )

    assert memory.state.estimated_row == 1
    assert memory.state.probability_at(1) > memory.state.probability_at(0)


def test_trajectory_generator_respects_motion_intent_and_limits():
    config = TrajectoryConfig(
        horizon_s=1.0,
        dt_s=0.1,
        row_spacing_m=0.8,
        max_lateral_speed_mps=1.4,
        candidate_count=8,
        noise_scale_m=0.0,
    )
    generator = DiffusionTrajectoryGenerator(config=config, seed=3)
    trajectory = generator.generate(
        GeneratorObservation(localization_confidence=1.0),
        MotionIntent.SHIFT_RIGHT,
    )

    assert math.isclose(trajectory[-1].y_m, 0.8, abs_tol=0.05)
    assert all(abs(point.vy_mps) <= config.max_lateral_speed_mps + 1e-9 for point in trajectory)
    assert all(next_point.t_s > point.t_s for point, next_point in zip(trajectory, trajectory[1:]))


def test_diffusion_interface_has_no_mission_progress_fields():
    field_names = {field.name for field in fields(GeneratorObservation)}

    assert "target_row" not in field_names
    assert "visited_rows" not in field_names
    assert "row_probabilities" not in field_names


def test_diffusion_returns_multiple_modes_for_ambiguous_observations():
    config = TrajectoryConfig(
        horizon_s=1.0,
        dt_s=0.1,
        row_spacing_m=0.8,
        max_lateral_speed_mps=1.4,
        candidate_count=8,
        noise_scale_m=0.0,
    )
    generator = DiffusionTrajectoryGenerator(config=config, seed=5)
    modes = generator.generate_modes(
        GeneratorObservation(
            hypotheses=(
                ObservationHypothesis(row_center_offset_m=-0.35, heading_error_rad=0.08, weight=0.5),
                ObservationHypothesis(row_center_offset_m=0.35, heading_error_rad=-0.08, weight=0.5),
            ),
            localization_confidence=0.4,
        ),
        MotionIntent.FOLLOW_ROW,
        mode_count=2,
    )

    assert len(modes) == 2
    assert {mode.hypothesis_index for mode in modes} == {0, 1}
    assert abs(modes[0].trajectory[0].y_m - modes[1].trajectory[0].y_m) > 0.5
    for mode in modes:
        assert abs(mode.trajectory[-1].y_m) < 0.05
        assert all(abs(point.vy_mps) <= config.max_lateral_speed_mps + 1e-9 for point in mode.trajectory)
