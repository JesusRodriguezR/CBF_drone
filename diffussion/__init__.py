"""Crop-row inspection under partial observability.

The package is intentionally small and dependency-light. It separates global
inspection decisions from local trajectory generation so a learned diffusion
policy can be swapped in behind the same interface later.
"""

from .belief_memory import BeliefMemory, BeliefState, OdometryDelta, PerceptionInput
from .high_level_planner import HighLevelPlanner, PlannerConfig, PlannerDecision, TaskMode
from .trajectory_generator import (
    DiffusionTrajectoryGenerator,
    MotionIntent,
    ObservationHypothesis,
    TrajectoryConfig,
    TrajectoryPoint,
    TrajectorySample,
)

__all__ = [
    "BeliefMemory",
    "BeliefState",
    "DiffusionTrajectoryGenerator",
    "HighLevelPlanner",
    "MotionIntent",
    "ObservationHypothesis",
    "OdometryDelta",
    "PerceptionInput",
    "PlannerConfig",
    "PlannerDecision",
    "TaskMode",
    "TrajectoryConfig",
    "TrajectoryPoint",
    "TrajectorySample",
]
