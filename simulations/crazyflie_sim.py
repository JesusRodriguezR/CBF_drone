"""
Crazyflie Digital Twin — Main Simulation Class
================================================

Orchestrates all subsystems into a single coherent simulation:

    Command → Motor Mixing → Motor Dynamics → Forces/Torques
            → Rigid Body Integration → State Update → Sensor Output

Usage:
    from crazyflie_sim import CrazyflieSim

    sim = CrazyflieSim()
    sim.reset()

    for _ in range(10000):  # 10 seconds at 1 kHz
        state, sensors = sim.step(command=[0.5, 0.0, 0.0, 0.0])

Coordinate convention: NED (North-East-Down)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

from crazyflie_params import CrazyflieParams
from rigid_body_dynamics import (
    RigidBodyDynamics,
    make_state,
    quat_to_euler,
    quat_to_rotation_matrix,
    POS, VEL, QUAT, OMEGA, STATE_DIM,
)
from motor_model import MotorGroup
from motor_mixing import MotorMixer
from aerodynamics_model import AerodynamicsModel
from sensor_model import IMUModel, BarometerModel


@dataclass
class SimConfig:
    """Simulation configuration."""
    dt: float = 0.001             # Simulation timestep [s] (1 kHz)
    sub_steps: int = 1            # Sub-steps per step() call for stability
    enable_aerodynamics: bool = True
    enable_ground_effect: bool = True
    enable_sensors: bool = True
    enable_ground_collision: bool = True   # Prevent going below z=0 (ground in NED is z≤0 altitude)
    ground_z: float = 0.0        # Ground plane Z position in NED (z=0 means altitude=0)
    sensor_seed: int = 42
    log_data: bool = True


class CrazyflieSim:
    """
    High-fidelity Crazyflie 2.1 digital twin simulation.

    Ties together dynamics, motors, mixing, aerodynamics, and sensors
    into a single step() call.
    """

    def __init__(
        self,
        params: CrazyflieParams = None,
        config: SimConfig = None,
    ):
        self.params = params or CrazyflieParams()
        self.config = config or SimConfig()

        # ── Subsystems ──
        self.dynamics = RigidBodyDynamics(
            mass=self.params.mass,
            inertia=self.params.inertia_matrix,
            gravity=self.params.g,
        )
        self.motors = MotorGroup(self.params)
        self.mixer = MotorMixer(self.params)
        self.aero = AerodynamicsModel(self.params)
        self.imu = IMUModel(self.params, seed=self.config.sensor_seed)
        self.baro = BarometerModel(seed=self.config.sensor_seed + 1)

        # ── State ──
        self.state = make_state()
        self.time = 0.0

        # ── Data logging ──
        self.log = {
            'time': [],
            'state': [],
            'motor_speeds': [],
            'motor_thrusts': [],
            'motor_pwms': [],
            'command': [],
            'euler_angles': [],
            'position': [],
            'velocity': [],
        }

    def reset(
        self,
        initial_state: np.ndarray = None,
        position: np.ndarray = None,
        euler_angles: np.ndarray = None,
    ) -> np.ndarray:
        """
        Reset the simulation to an initial state.

        Parameters
        ----------
        initial_state : ndarray (13,), optional
            Full state vector. If provided, overrides position/euler_angles.
        position : ndarray (3,), optional
            Initial position [x, y, z] in NED [m]. Default: origin.
        euler_angles : ndarray (3,), optional
            Initial [roll, pitch, yaw] in radians. Default: level.

        Returns
        -------
        state : ndarray (13,)
            Initial state vector.
        """
        if initial_state is not None:
            self.state = initial_state.copy()
        else:
            from rigid_body_dynamics import euler_to_quat
            quat = np.array([1, 0, 0, 0])
            if euler_angles is not None:
                quat = euler_to_quat(*euler_angles)
            pos = position if position is not None else np.zeros(3)
            self.state = make_state(position=pos, quaternion=quat)

        self.time = 0.0
        self.motors.reset()
        self.imu.reset(seed=self.config.sensor_seed)
        self.baro.reset(seed=self.config.sensor_seed + 1)

        # Clear log
        for key in self.log:
            self.log[key] = []

        return self.state.copy()

    def step(
        self,
        command: np.ndarray,
        command_type: str = 'attitude',
    ) -> Tuple[np.ndarray, dict]:
        """
        Advance the simulation by one timestep.

        Parameters
        ----------
        command : ndarray (4,)
            Control command. Interpretation depends on command_type:
            - 'attitude': [thrust, roll, pitch, yaw] where thrust ∈ [0,1],
              roll/pitch/yaw ∈ [-1,1]
            - 'pwm': [m1, m2, m3, m4] direct motor PWMs ∈ [0,1]
        command_type : str
            'attitude' or 'pwm'.

        Returns
        -------
        state : ndarray (13,)
            Updated state vector.
        sensors : dict
            Sensor readings (if sensors enabled), else empty dict.
        """
        command = np.asarray(command, dtype=float)
        dt_sub = self.config.dt / self.config.sub_steps

        for _ in range(self.config.sub_steps):
            self._step_inner(command, command_type, dt_sub)

        self.time += self.config.dt

        # ── Sensor readings ──
        sensors = {}
        if self.config.enable_sensors:
            # Compute body force for accelerometer
            motor_thrusts = self.motors.get_thrusts()
            yaw_torques = self.motors.get_torques_z()
            force_body, _ = self.mixer.compute_body_wrench(motor_thrusts, yaw_torques)

            if self.config.enable_aerodynamics:
                aero_f, _ = self.aero.compute_forces_and_torques(
                    self.state[VEL], self.state[OMEGA], self.state[QUAT]
                )
                force_body = force_body + aero_f

            sensors['imu'] = self.imu.measure(
                self.state, force_body, self.params.mass
            )
            altitude = -self.state[2]  # NED: z down, altitude = -z
            sensors['baro'] = self.baro.measure(altitude)

        # ── Logging ──
        if self.config.log_data:
            self._log_step(command)

        return self.state.copy(), sensors

    def _step_inner(
        self,
        command: np.ndarray,
        command_type: str,
        dt: float,
    ):
        """Single integration sub-step."""

        # ── 1. Motor mixing ──
        if command_type == 'attitude':
            motor_pwms = self.mixer.mix_forward(command)
        elif command_type == 'pwm':
            motor_pwms = np.clip(command, 0.0, 1.0)
        else:
            raise ValueError(f"Unknown command_type: {command_type}")

        # ── 2. Motor dynamics ──
        self.motors.step(motor_pwms, dt)

        # ── 3. Compute forces and torques ──
        motor_thrusts = self.motors.get_thrusts()
        yaw_torques = self.motors.get_torques_z()

        # Apply ground effect
        if self.config.enable_ground_effect:
            altitude = -self.state[2]  # NED z is down
            for i in range(4):
                motor_thrusts[i] = self.aero.compute_ground_effect(
                    altitude, motor_thrusts[i]
                )

        # Motor wrench
        force_body, torque_body = self.mixer.compute_body_wrench(
            motor_thrusts, yaw_torques
        )

        # Aerodynamic forces
        if self.config.enable_aerodynamics:
            aero_force, aero_torque = self.aero.compute_forces_and_torques(
                self.state[VEL], self.state[OMEGA], self.state[QUAT]
            )
            force_body = force_body + aero_force
            torque_body = torque_body + aero_torque

        # ── 4. Rigid body integration ──
        self.state = self.dynamics.step_rk4(
            self.state, force_body, torque_body, dt
        )

        # ── 5. Ground collision ──
        if self.config.enable_ground_collision:
            # In NED, ground is at z = ground_z (positive z = below ground)
            # The drone starts at z <= 0 (above ground) typically
            # But we can also think of it as: if altitude < 0, clamp
            altitude = -self.state[2]
            if altitude < 0:
                self.state[2] = -0.0  # Put on ground
                # Kill downward velocity (in NED, downward = positive vz)
                if self.state[5] > 0:
                    self.state[5] = 0.0
                # Damp angular velocity on ground contact
                self.state[OMEGA] *= 0.5

    def _log_step(self, command: np.ndarray):
        """Record current state to the log."""
        self.log['time'].append(self.time)
        self.log['state'].append(self.state.copy())
        self.log['motor_speeds'].append(self.motors.get_speeds().copy())
        self.log['motor_thrusts'].append(self.motors.get_thrusts().copy())
        self.log['command'].append(command.copy())
        self.log['euler_angles'].append(quat_to_euler(self.state[QUAT]))
        self.log['position'].append(self.state[POS].copy())
        self.log['velocity'].append(self.state[VEL].copy())

    def get_state(self) -> np.ndarray:
        """Return current state vector."""
        return self.state.copy()

    def get_euler_angles(self) -> np.ndarray:
        """Return current [roll, pitch, yaw] in radians."""
        return quat_to_euler(self.state[QUAT])

    def get_position(self) -> np.ndarray:
        """Return current position [x, y, z] in NED."""
        return self.state[POS].copy()

    def get_velocity(self) -> np.ndarray:
        """Return current velocity [vx, vy, vz] in world frame."""
        return self.state[VEL].copy()

    def get_log_arrays(self) -> dict:
        """Convert log lists to numpy arrays for analysis."""
        return {
            key: np.array(val) for key, val in self.log.items()
        }
