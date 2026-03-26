"""
Motor Mixing Matrix
====================

Firmware-accurate X-configuration motor mixing for the Crazyflie 2.1.

Maps high-level commands [thrust, roll, pitch, yaw] to individual motor
PWM values, and computes net body forces/torques from motor states.

Motor layout (top view, nose = +X, NED frame):

        M4 (CCW)     M1 (CW)
           \\         /
            \\       /
             center
            /       \\
           /         \\
        M3 (CW)      M2 (CCW)

    M1: front-right (+X, +Y)  → CW
    M2: back-right  (-X, +Y)  → CCW
    M3: back-left   (-X, -Y)  → CW
    M4: front-left  (+X, -Y)  → CCW

Firmware mixing (from crazyflie-firmware, QUAD_FORMATION_NORMAL):
    M1 = thrust + pitch + yaw
    M2 = thrust - roll  - yaw
    M3 = thrust - pitch + yaw
    M4 = thrust + roll  - yaw
"""

import numpy as np
from crazyflie_params import CrazyflieParams


class MotorMixer:
    """
    Motor mixing for Crazyflie X-configuration quadrotor.

    Provides forward mixing (commands → motor PWMs) and inverse computation
    (motor thrusts → net forces and torques on the body).
    """

    # Forward mixing matrix: [M1, M2, M3, M4]^T = MIX @ [thrust, roll, pitch, yaw]^T
    # Rows: M1, M2, M3, M4
    # Cols: thrust, roll, pitch, yaw
    FORWARD_MIX = np.array([
        [1.0,  0.0, +1.0, +1.0],   # M1 = T + pitch + yaw
        [1.0, -1.0,  0.0, -1.0],   # M2 = T - roll  - yaw
        [1.0,  0.0, -1.0, +1.0],   # M3 = T - pitch + yaw
        [1.0, +1.0,  0.0, -1.0],   # M4 = T + roll  - yaw
    ])

    def __init__(self, params: CrazyflieParams):
        self.params = params
        self.arm = params.arm_length

        # Motor positions in body frame (NED, X-config)
        # Each motor is at 45° from body axes
        d = self.arm * np.sqrt(2.0) / 2.0  # projection onto X and Y axes
        self.motor_positions = np.array([
            [+d, +d, 0.0],    # M1: front-right
            [-d, +d, 0.0],    # M2: back-right
            [-d, -d, 0.0],    # M3: back-left
            [+d, -d, 0.0],    # M4: front-left
        ])

    def mix_forward(self, command: np.ndarray) -> np.ndarray:
        """
        Convert [thrust, roll, pitch, yaw] command to motor PWMs.

        Parameters
        ----------
        command : ndarray (4,)
            [thrust, roll, pitch, yaw] — all normalised [0, 1] for thrust,
            [-1, 1] for roll/pitch/yaw.

        Returns
        -------
        motor_pwms : ndarray (4,)
            Normalised PWM values [0, 1] for [M1, M2, M3, M4].
        """
        raw = self.FORWARD_MIX @ command
        # Clamp to [0, 1]
        return np.clip(raw, 0.0, 1.0)

    def compute_body_wrench(
        self,
        thrusts: np.ndarray,
        yaw_torques: np.ndarray,
    ) -> tuple:
        """
        Compute net force and torque on the body from individual motor
        thrusts and reactive yaw torques.

        Parameters
        ----------
        thrusts : ndarray (4,)
            Thrust from each motor [N] (always positive).
        yaw_torques : ndarray (4,)
            Reactive torque about Z from each motor [N·m]
            (sign reflects CW/CCW).

        Returns
        -------
        force_body : ndarray (3,)
            Net force in body frame [N].
            Thrust acts in -Z direction (up in NED).
        torque_body : ndarray (3,)
            Net torque in body frame [N·m].
        """
        # ── Net thrust force ──
        # All motors produce thrust in the -Z_body direction (upward in NED)
        total_thrust = np.sum(thrusts)
        force_body = np.array([0.0, 0.0, -total_thrust])

        # ── Torques from thrust differential ──
        # Roll  (about X_body): right motors push down, left motors push up
        #   τ_x = arm_y * (F_right - F_left)
        #   Right: M1, M2 (positive Y)   Left: M3, M4 (negative Y)
        # Pitch (about Y_body): front motors vs back motors
        #   τ_y = arm_x * (F_front - F_back)
        #   Front: M1, M4 (positive X)   Back: M2, M3 (negative X)
        torque_body = np.zeros(3)

        for i, (thrust, pos) in enumerate(zip(thrusts, self.motor_positions)):
            # Torque = r × F, where F = [0, 0, -thrust] in body frame
            # r × [0, 0, -T] = [-T*ry, T*rx, 0]... wait, let me be careful
            # r = [rx, ry, 0], F = [0, 0, -T]
            # r × F = [ry*(-T) - 0, 0 - rx*(-T), 0] = [-T*ry, T*rx, 0]
            torque_body[0] += -thrust * pos[1]   # roll torque
            torque_body[1] += thrust * pos[0]     # pitch torque

        # Yaw torque (about Z_body): sum of reactive torques
        torque_body[2] = np.sum(yaw_torques)

        return force_body, torque_body

    def get_allocation_matrix(self) -> np.ndarray:
        """
        Return the 4×4 control allocation matrix that maps motor thrusts
        [F1, F2, F3, F4] to body wrench [Fz, τx, τy, τz].

        Useful for control design and analysis.
        """
        d = self.arm * np.sqrt(2.0) / 2.0
        kf = self.params.k_f
        km = self.params.k_m
        ratio = km / kf  # torque-to-thrust ratio

        # Spin directions: M1=CW(+1), M2=CCW(-1), M3=CW(+1), M4=CCW(-1)
        spins = np.array([+1, -1, +1, -1])

        B = np.array([
            [1.0,          1.0,          1.0,          1.0         ],  # Fz (sum of thrusts)
            [-d,           -d,            d,            d          ],  # τx (roll)
            [d,            -d,           -d,            d          ],  # τy (pitch)
            [-spins[0]*ratio, -spins[1]*ratio, -spins[2]*ratio, -spins[3]*ratio],  # τz (yaw)
        ])

        return B
