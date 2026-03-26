"""
6-DOF Rigid Body Dynamics
=========================

Quaternion-based rigid body equations of motion for a quadrotor UAV.
Integrates translational and rotational dynamics using 4th-order
Runge-Kutta (RK4).

Coordinate convention: NED (North-East-Down)
    +X = forward, +Y = right, +Z = down
    Gravity acts in +Z direction.

State vector (13 elements):
    [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
     position    velocity    quaternion        angular vel (body)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple


# ── Quaternion Utilities ────────────────────────────────────────────────

def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize a quaternion [qw, qx, qy, qz] to unit length."""
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions [qw, qx, qy, qz]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate of a unit quaternion (inverse for unit quaternions)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotate vector v by quaternion q: v' = q ⊗ [0,v] ⊗ q*

    Parameters
    ----------
    q : array (4,) — unit quaternion [qw, qx, qy, qz]
    v : array (3,) — vector in the original frame

    Returns
    -------
    v_rotated : array (3,) — vector in the rotated frame
    """
    v_quat = np.array([0.0, v[0], v[1], v[2]])
    rotated = quat_multiply(quat_multiply(q, v_quat), quat_conjugate(q))
    return rotated[1:4]


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [qw, qx, qy, qz] to a 3×3 rotation matrix.

    The matrix R transforms body-frame vectors to world-frame:
        v_world = R @ v_body
    """
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),       1 - 2*(qx**2 + qz**2),  2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),       2*(qy*qz + qx*qw),      1 - 2*(qx**2 + qy**2)],
    ])


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [qw, qx, qy, qz] to Euler angles [roll, pitch, yaw] in radians.

    Uses ZYX (yaw-pitch-roll) convention.
    """
    qw, qx, qy, qz = q

    # Roll (rotation about X)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx**2 + qy**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (rotation about Y)
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # Yaw (rotation about Z)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy**2 + qz**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert Euler angles [roll, pitch, yaw] (ZYX convention) to
    quaternion [qw, qx, qy, qz].
    """
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)

    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


# ── State Indexing ──────────────────────────────────────────────────────

POS   = slice(0, 3)    # x, y, z
VEL   = slice(3, 6)    # vx, vy, vz
QUAT  = slice(6, 10)   # qw, qx, qy, qz
OMEGA = slice(10, 13)  # wx, wy, wz (body frame)
STATE_DIM = 13


def make_state(
    position: np.ndarray = None,
    velocity: np.ndarray = None,
    quaternion: np.ndarray = None,
    angular_velocity: np.ndarray = None,
) -> np.ndarray:
    """Create a state vector from components (defaults to hover at origin)."""
    state = np.zeros(STATE_DIM)
    state[POS] = position if position is not None else np.zeros(3)
    state[VEL] = velocity if velocity is not None else np.zeros(3)
    state[QUAT] = quaternion if quaternion is not None else np.array([1, 0, 0, 0])
    state[OMEGA] = angular_velocity if angular_velocity is not None else np.zeros(3)
    return state


# ── Rigid Body Dynamics ────────────────────────────────────────────────

class RigidBodyDynamics:
    """
    6-DOF rigid body dynamics with quaternion attitude.

    Integrates Newton-Euler equations using RK4.
    """

    def __init__(self, mass: float, inertia: np.ndarray, gravity: float = 9.81):
        """
        Parameters
        ----------
        mass : float
            Total mass [kg].
        inertia : ndarray (3, 3)
            Inertia tensor in body frame [kg·m²].
        gravity : float
            Gravitational acceleration [m/s²].
        """
        self.mass = mass
        self.inertia = inertia.copy()
        self.inertia_inv = np.linalg.inv(inertia)
        self.gravity = gravity

        # Gravity vector in NED: +Z is down
        self.gravity_world = np.array([0.0, 0.0, self.gravity])

    def state_derivative(
        self,
        state: np.ndarray,
        force_body: np.ndarray,
        torque_body: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the time derivative of the state vector.

        Parameters
        ----------
        state : ndarray (13,)
            Current state vector.
        force_body : ndarray (3,)
            Net force in body frame [N].
        torque_body : ndarray (3,)
            Net torque in body frame [N·m].

        Returns
        -------
        state_dot : ndarray (13,)
            Time derivative of state.
        """
        pos = state[POS]
        vel = state[VEL]
        q = quat_normalize(state[QUAT])
        omega = state[OMEGA]

        state_dot = np.zeros(STATE_DIM)

        # ── Translational dynamics ──
        # Transform body force to world frame and add gravity
        R = quat_to_rotation_matrix(q)
        force_world = R @ force_body + self.mass * self.gravity_world

        state_dot[POS] = vel
        state_dot[VEL] = force_world / self.mass

        # ── Quaternion kinematics ──
        # q_dot = 0.5 * q ⊗ [0, omega]
        omega_quat = np.array([0.0, omega[0], omega[1], omega[2]])
        q_dot = 0.5 * quat_multiply(q, omega_quat)
        state_dot[QUAT] = q_dot

        # ── Rotational dynamics (Euler's equation) ──
        # I·ω̇ = τ - ω × (I·ω)
        omega_cross_I_omega = np.cross(omega, self.inertia @ omega)
        omega_dot = self.inertia_inv @ (torque_body - omega_cross_I_omega)
        state_dot[OMEGA] = omega_dot

        return state_dot

    def step_rk4(
        self,
        state: np.ndarray,
        force_body: np.ndarray,
        torque_body: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Advance the state by one timestep using 4th-order Runge-Kutta.

        Forces and torques are assumed constant over the timestep
        (zero-order hold).

        Parameters
        ----------
        state : ndarray (13,)
            Current state.
        force_body : ndarray (3,)
            Net force in body frame [N].
        torque_body : ndarray (3,)
            Net torque in body frame [N·m].
        dt : float
            Timestep [s].

        Returns
        -------
        new_state : ndarray (13,)
            State after integration.
        """
        k1 = self.state_derivative(state, force_body, torque_body)
        k2 = self.state_derivative(state + 0.5 * dt * k1, force_body, torque_body)
        k3 = self.state_derivative(state + 0.5 * dt * k2, force_body, torque_body)
        k4 = self.state_derivative(state + dt * k3, force_body, torque_body)

        new_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Renormalize quaternion to prevent drift
        new_state[QUAT] = quat_normalize(new_state[QUAT])

        return new_state
