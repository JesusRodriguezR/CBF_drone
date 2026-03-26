"""
Motor Dynamics Model
====================

Models the Crazyflie 2.1 brushed coreless DC motors (7×16 mm) with
45 mm propellers.

Each motor is modelled as:
    1. First-order lag:  τ_m · ω̇ = ω_cmd - ω
    2. Thrust:           F = k_f · ω²
    3. Reactive torque:  M = k_m · ω²  (sign depends on spin direction)

The motor accepts normalised PWM commands [0, 1] and converts them to
angular velocity setpoints via the empirical thrust curve.

Units: SI (rad/s, N, N·m)
"""

import numpy as np
from crazyflie_params import CrazyflieParams


class Motor:
    """Single brushed DC motor with first-order dynamics."""

    def __init__(self, params: CrazyflieParams, spin_direction: int = 1):
        """
        Parameters
        ----------
        params : CrazyflieParams
            Physical parameters.
        spin_direction : int
            +1 for CW, -1 for CCW (determines reactive torque sign).
        """
        self.params = params
        self.spin_direction = spin_direction

        # Current motor angular velocity [rad/s]
        self.speed = 0.0

    def reset(self):
        """Reset motor speed to zero."""
        self.speed = 0.0

    def pwm_to_speed(self, pwm: float) -> float:
        """
        Convert normalised PWM [0, 1] to target motor speed [rad/s].

        Uses the empirical thrust curve inverted:
            thrust = a·pwm² + b·pwm
            omega = sqrt(thrust / k_f)
        """
        pwm = np.clip(pwm, 0.0, 1.0)
        p = self.params

        # Thrust from empirical curve [N]
        thrust = p.pwm_thrust_a * pwm**2 + p.pwm_thrust_b * pwm

        # Clamp to non-negative
        thrust = max(thrust, 0.0)

        # Invert F = k_f · ω² → ω = sqrt(F / k_f)
        speed = np.sqrt(thrust / p.k_f)

        # Clamp to motor limits
        return np.clip(speed, p.motor_min_speed, p.motor_max_speed)

    def speed_to_thrust(self, speed: float = None) -> float:
        """Compute thrust [N] from motor speed [rad/s]."""
        if speed is None:
            speed = self.speed
        return self.params.k_f * speed**2

    def speed_to_torque(self, speed: float = None) -> float:
        """
        Compute reactive torque [N·m] on the body from this motor.

        Sign convention: CW motors (+1) produce -Z torque on body,
        CCW motors (-1) produce +Z torque on body (reaction torque
        opposes motor spin direction).
        """
        if speed is None:
            speed = self.speed
        return -self.spin_direction * self.params.k_m * speed**2

    def step(self, pwm_command: float, dt: float) -> float:
        """
        Advance motor dynamics by one timestep.

        Parameters
        ----------
        pwm_command : float
            Normalised PWM command [0, 1].
        dt : float
            Timestep [s].

        Returns
        -------
        speed : float
            Updated motor speed [rad/s].
        """
        target_speed = self.pwm_to_speed(pwm_command)
        tau = self.params.motor_time_constant

        # First-order lag: exponential step
        # ω(t+dt) = ω_target + (ω(t) - ω_target) * exp(-dt/τ)
        alpha = 1.0 - np.exp(-dt / tau)
        self.speed += alpha * (target_speed - self.speed)

        # Clamp
        self.speed = np.clip(
            self.speed, self.params.motor_min_speed, self.params.motor_max_speed
        )

        return self.speed


class MotorGroup:
    """
    Group of 4 motors for the Crazyflie quadrotor.

    Motor numbering and spin directions (top view, nose = +X, NED):

        M4 (CCW, -1)    M1 (CW, +1)
             \\          /
              \\        /
               center
              /        \\
             /          \\
        M3 (CW, +1)     M2 (CCW, -1)

    CW  = clockwise when viewed from above
    CCW = counter-clockwise when viewed from above
    """

    # Spin directions: +1 = CW, -1 = CCW
    SPIN_DIRECTIONS = [+1, -1, +1, -1]  # M1, M2, M3, M4

    def __init__(self, params: CrazyflieParams):
        self.params = params
        self.motors = [
            Motor(params, spin_dir)
            for spin_dir in self.SPIN_DIRECTIONS
        ]

    def reset(self):
        """Reset all motors to zero speed."""
        for motor in self.motors:
            motor.reset()

    def step(self, pwm_commands: np.ndarray, dt: float) -> np.ndarray:
        """
        Step all motors with their respective PWM commands.

        Parameters
        ----------
        pwm_commands : ndarray (4,)
            Normalised PWM commands [0, 1] for [M1, M2, M3, M4].
        dt : float
            Timestep [s].

        Returns
        -------
        speeds : ndarray (4,)
            Updated motor speeds [rad/s].
        """
        speeds = np.array([
            motor.step(pwm, dt)
            for motor, pwm in zip(self.motors, pwm_commands)
        ])
        return speeds

    def get_speeds(self) -> np.ndarray:
        """Get current motor speeds [rad/s]."""
        return np.array([m.speed for m in self.motors])

    def get_thrusts(self) -> np.ndarray:
        """Get current thrust per motor [N]."""
        return np.array([m.speed_to_thrust() for m in self.motors])

    def get_torques_z(self) -> np.ndarray:
        """Get reactive yaw torques per motor [N·m]."""
        return np.array([m.speed_to_torque() for m in self.motors])

    def set_speeds_directly(self, speeds: np.ndarray):
        """Set motor speeds directly (bypassing dynamics), for testing."""
        for motor, speed in zip(self.motors, speeds):
            motor.speed = speed
