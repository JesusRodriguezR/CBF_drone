"""
Finite State Machine Controller for Greenhouse Lane Navigation
================================================================

Implements the 6-state FSM that governs the drone's behavior:

    TAKEOFF → ENTER_LANE → IN_LANE → END_OF_LANE → TURNING → (repeat or LAND)

Each state produces a velocity command [vx, vy] based on:
  - Current drone position (from Flow Deck odometry)
  - CNN perception outputs (lateral_offset, end_of_row_prob)
  - Coverage planner state (current lane, direction, next target)
"""

import numpy as np
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Tuple
from greenhouse_env import GreenhouseEnv
from coverage_planner import CoveragePlanner
from simulated_perception import SimulatedPerception


class FSMState(Enum):
    TAKEOFF = auto()
    ENTER_LANE = auto()
    IN_LANE = auto()
    END_OF_LANE = auto()
    TURNING = auto()
    LAND = auto()
    DONE = auto()


@dataclass
class FSMConfig:
    """FSM controller parameters."""
    cruise_speed: float = 0.4          # Forward speed in lane [m/s]
    headland_speed: float = 0.35       # Speed during headland maneuvers [m/s]
    approach_speed: float = 0.2        # Slow approach speed at end of lane [m/s]

    # Lane centering
    lateral_gain: float = 1.5          # P-gain for lateral correction [1/s]
    max_lateral_correction: float = 0.2  # Max lateral velocity correction [m/s]

    # End-of-row detection
    end_of_row_threshold: float = 0.7  # Probability threshold to trigger END_OF_LANE

    # Waypoint reaching
    position_tolerance: float = 0.15   # Distance to consider waypoint reached [m]

    # Takeoff
    takeoff_hover_time: float = 1.0    # Hover before starting [s]


class FSMController:
    """
    Finite State Machine controller for greenhouse lane coverage.

    The FSM takes perception inputs and drone state, and outputs
    velocity commands [vx, vy] that will be passed through the CBF
    safety filter before reaching the drone.
    """

    def __init__(self, env: GreenhouseEnv, planner: CoveragePlanner,
                 perception: SimulatedPerception, config: FSMConfig = None):
        self.env = env
        self.planner = planner
        self.perception = perception
        self.config = config or FSMConfig()

        self.state = FSMState.TAKEOFF
        self.prev_state = None
        self.state_start_time = 0.0

        # Current lane tracking
        self.current_lane_idx = 0
        self.current_lane_direction = 1  # +1 = top→bot, -1 = bot→top
        self.lanes_visited = set()

        # Turning state
        self._turn_target_x = 0.0
        self._turn_target_y = 0.0
        self._turn_phase = 0  # 0=lateral shift, 1=done

        # Log
        self.log = {
            'time': [],
            'state': [],
            'vx_cmd': [],
            'vy_cmd': [],
            'lane_idx': [],
            'lateral_offset': [],
            'end_of_row_prob': [],
        }

    def reset(self):
        """Reset FSM to initial state."""
        self.state = FSMState.TAKEOFF
        self.prev_state = None
        self.state_start_time = 0.0
        self.current_lane_idx = 0
        self.current_lane_direction = 1
        self.lanes_visited = set()
        self._turn_phase = 0
        for key in self.log:
            self.log[key] = []

    def _setup_lane_traversal(self):
        """Configure direction for the current lane."""
        # Boustrophedon: alternate direction each lane
        if self.current_lane_idx % 2 == 0:
            self.current_lane_direction = 1   # top → bottom (decreasing y)
        else:
            self.current_lane_direction = -1  # bottom → top (increasing y)

    def step(self, drone_x: float, drone_y: float, drone_theta: float,
             time: float) -> Tuple[float, float]:
        """
        Compute velocity command based on current FSM state.

        Parameters
        ----------
        drone_x, drone_y : float — current drone position [m]
        drone_theta : float — current heading [rad]
        time : float — current simulation time [s]

        Returns
        -------
        vx_cmd, vy_cmd : float — velocity command [m/s]
        """
        cfg = self.config
        env = self.env

        # Track state transitions
        if self.state != self.prev_state:
            self.state_start_time = time
            self.prev_state = self.state

        # Get perception
        perception = self.perception.perceive(
            drone_x, drone_y,
            self.current_lane_idx,
            self.current_lane_direction,
            time
        )
        lat_offset = perception['lateral_offset']
        end_prob = perception['end_of_row_prob']

        vx_cmd = 0.0
        vy_cmd = 0.0

        # ════════════════════════════════════════════
        if self.state == FSMState.TAKEOFF:
            # Hover in place for takeoff_hover_time
            if time - self.state_start_time >= cfg.takeoff_hover_time:
                self._setup_lane_traversal()
                self.state = FSMState.ENTER_LANE
            vx_cmd, vy_cmd = 0.0, 0.0

        # ════════════════════════════════════════════
        elif self.state == FSMState.ENTER_LANE:
            # Navigate to the entry point of the current lane
            lane_cx = env.lane_centers_x[self.current_lane_idx]

            if self.current_lane_direction == 1:
                # Entering from top
                entry_y = env.lane_y_max + 0.2
            else:
                # Entering from bottom
                entry_y = env.lane_y_min - 0.2

            dx = lane_cx - drone_x
            dy = entry_y - drone_y
            dist = np.hypot(dx, dy)

            if dist < cfg.position_tolerance:
                # Aligned with lane entry, start flying through
                self.state = FSMState.IN_LANE
            else:
                speed = cfg.headland_speed
                vx_cmd = speed * dx / max(dist, 0.01)
                vy_cmd = speed * dy / max(dist, 0.01)

        # ════════════════════════════════════════════
        elif self.state == FSMState.IN_LANE:
            # Fly along lane with lateral correction from CNN

            # Forward velocity (along lane direction)
            if self.current_lane_direction == 1:
                vy_cmd = -cfg.cruise_speed  # top → bottom = -y
            else:
                vy_cmd = cfg.cruise_speed   # bottom → top = +y

            # Lateral correction from CNN lateral_offset
            lateral_correction = -cfg.lateral_gain * lat_offset
            lateral_correction = np.clip(
                lateral_correction,
                -cfg.max_lateral_correction,
                cfg.max_lateral_correction
            )
            vx_cmd = lateral_correction

            # Check end-of-row
            if end_prob > cfg.end_of_row_threshold:
                self.state = FSMState.END_OF_LANE

        # ════════════════════════════════════════════
        elif self.state == FSMState.END_OF_LANE:
            # Decelerate and stop at lane end

            # Slow approach toward lane boundary
            if self.current_lane_direction == 1:
                dist_to_end = drone_y - env.lane_y_min
            else:
                dist_to_end = env.lane_y_max - drone_y

            if dist_to_end < 0.1:
                # Stopped, mark lane as visited
                self.lanes_visited.add(self.current_lane_idx)

                # Check if all lanes done
                if len(self.lanes_visited) >= env.config.n_lanes:
                    self.state = FSMState.LAND
                else:
                    # Setup turning to next lane
                    next_lane = self.current_lane_idx + 1
                    self._turn_target_x = env.lane_centers_x[next_lane]

                    if self.current_lane_direction == 1:
                        # We're at the bottom, stay in bottom headland
                        self._turn_target_y = env.lane_y_min - 0.5
                    else:
                        # We're at the top, stay in top headland
                        self._turn_target_y = env.lane_y_max + 0.5

                    self._turn_phase = 0
                    self.state = FSMState.TURNING
            else:
                # Slow approach
                if self.current_lane_direction == 1:
                    vy_cmd = -cfg.approach_speed
                else:
                    vy_cmd = cfg.approach_speed

                # Still correct laterally
                vx_cmd = -cfg.lateral_gain * lat_offset * 0.5
                vx_cmd = np.clip(vx_cmd,
                                 -cfg.max_lateral_correction,
                                 cfg.max_lateral_correction)

        # ════════════════════════════════════════════
        elif self.state == FSMState.TURNING:
            # Two-phase turn in headland:
            #   Phase 0: Move to the headland position at next lane's x
            #   Phase 1: (done, transition to ENTER_LANE)

            dx = self._turn_target_x - drone_x
            dy = self._turn_target_y - drone_y
            dist = np.hypot(dx, dy)

            if dist < cfg.position_tolerance:
                # Reached turn target
                self.current_lane_idx += 1
                self._setup_lane_traversal()
                self.state = FSMState.ENTER_LANE
            else:
                speed = cfg.headland_speed
                vx_cmd = speed * dx / max(dist, 0.01)
                vy_cmd = speed * dy / max(dist, 0.01)

        # ════════════════════════════════════════════
        elif self.state == FSMState.LAND:
            # Hover / stop
            vx_cmd, vy_cmd = 0.0, 0.0
            if time - self.state_start_time > 1.0:
                self.state = FSMState.DONE

        # ════════════════════════════════════════════
        elif self.state == FSMState.DONE:
            vx_cmd, vy_cmd = 0.0, 0.0

        # ── Log ──
        self.log['time'].append(time)
        self.log['state'].append(self.state.name)
        self.log['vx_cmd'].append(vx_cmd)
        self.log['vy_cmd'].append(vy_cmd)
        self.log['lane_idx'].append(self.current_lane_idx)
        self.log['lateral_offset'].append(lat_offset)
        self.log['end_of_row_prob'].append(end_prob)

        return vx_cmd, vy_cmd

    def is_done(self) -> bool:
        return self.state == FSMState.DONE

    def get_state_name(self) -> str:
        return self.state.name

    def get_log_arrays(self) -> dict:
        result = {}
        for key, val in self.log.items():
            if key == 'state':
                result[key] = val
            else:
                result[key] = np.array(val)
        return result
