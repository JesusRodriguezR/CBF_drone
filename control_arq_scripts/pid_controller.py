"""
PID Controller for Crazyflie 2.1
==================================

Cascaded controller matching the real Crazyflie firmware architecture:

    User input → desired angles/altitude
                     ↓
    Outer loop:  angle error → desired angular rate
    Inner loop:  rate error  → desired torque
                     ↓
    Altitude PI: altitude error → desired thrust (in Newtons)
                     ↓
    Force allocation:  [F_total, τx, τy, τz] → [F1, F2, F3, F4]
                     ↓
    Thrust-to-PWM:  per-motor force → per-motor PWM

This architecture provides critical angular rate damping that prevents
the drone from spinning out of control despite its tiny inertia.
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simulations'))
from crazyflie_params import CrazyflieParams
from rigid_body_dynamics import quat_to_euler, POS, VEL, QUAT, OMEGA


class PID:
    """Minimal PID with anti-windup."""

    def __init__(self, kp, ki, kd, i_limit=1.0, out_limit=None):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i_limit = i_limit
        self.out_limit = out_limit
        self.integral = 0.0
        self.prev_error = None

    def reset(self):
        self.integral = 0.0
        self.prev_error = None

    def update(self, error, dt):
        p = self.kp * error
        self.integral = np.clip(self.integral + error * dt,
                                -self.i_limit, self.i_limit)
        i = self.ki * self.integral
        if self.prev_error is not None and dt > 0:
            d = self.kd * (error - self.prev_error) / dt
        else:
            d = 0.0
        self.prev_error = error
        out = p + i + d
        if self.out_limit is not None:
            out = np.clip(out, -self.out_limit, self.out_limit)
        return out


class CrazyflieController:
    """
    Cascaded PID flight controller.

    Outer loop (angle):  angle_error → desired_rate
    Inner loop (rate):   rate_error  → torque
    Altitude loop:       alt_error   → thrust force
    """

    def __init__(self, params: CrazyflieParams = None):
        self.params = params or CrazyflieParams()

        # ── Attitude outer loop (angle → desired rate) ──
        # Low gains: we want gentle rate commands
        self.roll_angle_pid = PID(kp=5.0, ki=0.5, kd=0.0,
                                  i_limit=0.5, out_limit=3.0)  # max 3 rad/s
        self.pitch_angle_pid = PID(kp=5.0, ki=0.5, kd=0.0,
                                   i_limit=0.5, out_limit=3.0)

        # ── Attitude inner loop (rate → torque) ──
        # Higher gains for fast damping
        self.roll_rate_pid = PID(kp=0.0004, ki=0.00002, kd=0.000002,
                                 i_limit=0.001, out_limit=0.002)
        self.pitch_rate_pid = PID(kp=0.0004, ki=0.00002, kd=0.000002,
                                  i_limit=0.001, out_limit=0.002)

        # ── Yaw (rate only) ──
        self.yaw_rate_pid = PID(kp=0.0006, ki=0.00003, kd=0.000002,
                                i_limit=0.001, out_limit=0.002)

        # ── Altitude ──
        self.alt_pid = PID(kp=1.5, ki=0.3, kd=0.8,
                           i_limit=0.5, out_limit=2.0)

        # Motor allocation geometry
        d = self.params.arm_length * np.sqrt(2) / 2
        km_kf = self.params.k_m / self.params.k_f  # torque-to-thrust ratio

        # Allocation matrix: [F_total, τ_roll, τ_pitch, τ_yaw] → [F1, F2, F3, F4]
        # Inverse of: B @ [F1,F2,F3,F4] = [F_total, τ_roll, τ_pitch, τ_yaw]
        # Motor config: M1(FR,CW), M2(BR,CCW), M3(BL,CW), M4(FL,CCW)
        self.B = np.array([
            [1.0,    1.0,     1.0,    1.0    ],   # F_total = sum
            [-d,     -d,      +d,     +d     ],   # τ_roll
            [+d,     -d,      -d,     +d     ],   # τ_pitch
            [-km_kf, +km_kf,  -km_kf, +km_kf ],   # τ_yaw (reactive)
        ])
        self.B_inv = np.linalg.pinv(self.B)

    def reset(self):
        for pid in [self.roll_angle_pid, self.pitch_angle_pid,
                    self.roll_rate_pid, self.pitch_rate_pid,
                    self.yaw_rate_pid, self.alt_pid]:
            pid.reset()

    def force_to_pwm(self, force):
        """Convert per-motor force [N] to normalised PWM [0, 1]."""
        force = max(force, 0.0)
        # F = k_f * omega^2, omega = sqrt(F/k_f)
        omega = np.sqrt(force / self.params.k_f)
        # PWM from empirical inverse: F = a*pwm^2 + b*pwm
        # Solve quadratic: pwm = (-b + sqrt(b^2 + 4aF)) / 2a
        a = self.params.pwm_thrust_a
        b = self.params.pwm_thrust_b
        disc = b**2 + 4 * a * force
        if disc < 0:
            return 0.0
        pwm = (-b + np.sqrt(disc)) / (2 * a)
        return np.clip(pwm, 0.0, 1.0)

    def compute(
        self,
        state: np.ndarray,
        desired_roll: float = 0.0,
        desired_pitch: float = 0.0,
        desired_yaw_rate: float = 0.0,
        desired_altitude: float = 1.0,
        dt: float = 0.001,
    ) -> np.ndarray:
        """Compute motor PWM commands [M1, M2, M3, M4]."""
        euler = quat_to_euler(state[QUAT])
        roll, pitch, yaw = euler
        omega = state[OMEGA]   # body angular velocity [rad/s]
        altitude = -state[2]   # NED z is down

        # ── 1. Altitude → total thrust force ──
        alt_error = desired_altitude - altitude
        accel_cmd = self.alt_pid.update(alt_error, dt)
        # F = m * (g + a_cmd)
        f_total = self.params.mass * (self.params.g + accel_cmd)
        f_total = np.clip(f_total, 0.0, self.params.mass * self.params.g * 3.0)

        # ── 2. Angle → desired rate (outer loop) ──
        roll_error = desired_roll - roll
        desired_roll_rate = self.roll_angle_pid.update(roll_error, dt)

        pitch_error = desired_pitch - pitch
        desired_pitch_rate = self.pitch_angle_pid.update(pitch_error, dt)

        # ── 3. Rate → torque (inner loop) ──
        roll_rate_error = desired_roll_rate - omega[0]
        tau_roll = self.roll_rate_pid.update(roll_rate_error, dt)

        pitch_rate_error = desired_pitch_rate - omega[1]
        tau_pitch = self.pitch_rate_pid.update(pitch_rate_error, dt)

        yaw_rate_error = desired_yaw_rate - omega[2]
        tau_yaw = self.yaw_rate_pid.update(yaw_rate_error, dt)

        # ── 4. Allocate to per-motor forces ──
        wrench = np.array([f_total, tau_roll, tau_pitch, tau_yaw])
        motor_forces = self.B_inv @ wrench
        motor_forces = np.clip(motor_forces, 0.0, self.params.max_thrust_per_motor)

        # ── 5. Convert forces to PWMs ──
        motor_pwms = np.array([self.force_to_pwm(f) for f in motor_forces])
        return motor_pwms
