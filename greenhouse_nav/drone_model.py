"""
Drone Kinematic Model — 2D Velocity-Controlled Point Mass
==========================================================

Simple unicycle-like model for the 2D lane-navigation simulation.
Assumes the low-level attitude controller is already solved (by the
existing digital twin / PID). This model takes velocity commands
and integrates them.

State: [x, y, theta]   (position + heading)
Input: [v, omega]      (forward speed, yaw rate)

Also supports direct velocity commands [vx, vy] for headland maneuvers.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class DroneConfig:
    """Drone kinematic parameters."""
    max_speed: float = 0.5        # Max forward speed [m/s]
    max_yaw_rate: float = 2.0     # Max yaw rate [rad/s]
    max_accel: float = 1.0        # Max acceleration [m/s^2]


class DroneModel:
    """2D kinematic drone model with state [x, y, theta]."""

    def __init__(self, config: DroneConfig = None):
        self.config = config or DroneConfig()
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0  # heading in radians, 0 = +x, pi/2 = +y
        self.vx = 0.0
        self.vy = 0.0
        self.speed = 0.0

        # Data log
        self.log = {
            'time': [],
            'x': [],
            'y': [],
            'theta': [],
            'vx': [],
            'vy': [],
            'speed': [],
            'state_label': [],
        }
        self.time = 0.0

    def reset(self, x: float, y: float, theta: float):
        """Reset drone to initial position and heading."""
        self.x = x
        self.y = y
        self.theta = theta
        self.vx = 0.0
        self.vy = 0.0
        self.speed = 0.0
        self.time = 0.0
        for key in self.log:
            self.log[key] = []

    def step_velocity(self, vx_cmd: float, vy_cmd: float, dt: float):
        """
        Step with direct velocity commands (used after CBF filtering).

        Parameters
        ----------
        vx_cmd : float   — commanded x-velocity [m/s]
        vy_cmd : float   — commanded y-velocity [m/s]
        dt     : float   — timestep [s]
        """
        # Apply speed limit
        speed = np.hypot(vx_cmd, vy_cmd)
        if speed > self.config.max_speed:
            scale = self.config.max_speed / speed
            vx_cmd *= scale
            vy_cmd *= scale

        self.vx = vx_cmd
        self.vy = vy_cmd
        self.speed = np.hypot(self.vx, self.vy)

        # Update heading to match velocity direction (if moving)
        if self.speed > 0.01:
            self.theta = np.arctan2(self.vy, self.vx)

        # Integrate position
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.time += dt

    def log_state(self, state_label: str = ''):
        """Record current state to the log."""
        self.log['time'].append(self.time)
        self.log['x'].append(self.x)
        self.log['y'].append(self.y)
        self.log['theta'].append(self.theta)
        self.log['vx'].append(self.vx)
        self.log['vy'].append(self.vy)
        self.log['speed'].append(self.speed)
        self.log['state_label'].append(state_label)

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def get_log_arrays(self) -> dict:
        """Convert log to numpy arrays."""
        result = {}
        for key, val in self.log.items():
            if key == 'state_label':
                result[key] = val  # keep as list of strings
            else:
                result[key] = np.array(val)
        return result
