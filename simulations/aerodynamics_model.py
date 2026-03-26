"""
Aerodynamics Model
==================

Aerodynamic effects for the Crazyflie 2.1 digital twin:
    1. Linear body drag (translational)
    2. Rotational drag (damping)
    3. Ground effect (thrust amplification near the floor)

These are applied as additional forces/torques in the body frame
before integration.

For a micro-UAV flying indoors at low speeds, a linear drag model
is a reasonable approximation (Reynolds number regime).
"""

import numpy as np
from crazyflie_params import CrazyflieParams
from rigid_body_dynamics import quat_to_rotation_matrix, quat_conjugate, quat_rotate_vector


class AerodynamicsModel:
    """Aerodynamic force and torque model."""

    def __init__(self, params: CrazyflieParams):
        self.params = params

    def compute_drag_force(
        self,
        velocity_world: np.ndarray,
        quaternion: np.ndarray,
    ) -> np.ndarray:
        """
        Compute translational drag force in body frame.

        Uses a linear drag model: F_drag = -K_drag · v_body
        where K_drag is a diagonal matrix with different coefficients
        for lateral (xy) and vertical (z) motion.

        Parameters
        ----------
        velocity_world : ndarray (3,)
            Velocity in world frame [m/s].
        quaternion : ndarray (4,)
            Current attitude quaternion [qw, qx, qy, qz].

        Returns
        -------
        drag_force_body : ndarray (3,)
            Drag force in body frame [N].
        """
        # Transform velocity to body frame
        R = quat_to_rotation_matrix(quaternion)
        v_body = R.T @ velocity_world

        # Diagonal drag coefficients
        drag = np.array([
            -self.params.k_drag_xy * v_body[0],
            -self.params.k_drag_xy * v_body[1],
            -self.params.k_drag_z * v_body[2],
        ])

        return drag

    def compute_rotational_drag(
        self,
        angular_velocity: np.ndarray,
    ) -> np.ndarray:
        """
        Compute rotational drag torque in body frame.

        Models aerodynamic damping on angular velocity:
            τ_drag = -k_rot · ω

        Parameters
        ----------
        angular_velocity : ndarray (3,)
            Angular velocity in body frame [rad/s].

        Returns
        -------
        drag_torque : ndarray (3,)
            Rotational drag torque in body frame [N·m].
        """
        return -self.params.k_rot_drag * angular_velocity

    def compute_ground_effect(
        self,
        altitude: float,
        base_thrust: float,
    ) -> float:
        """
        Compute ground-effect thrust amplification.

        When the drone is within a few rotor diameters of the ground,
        the induced flow recirculates and increases effective thrust.

        Uses a simple linear model:
            multiplier = 1 + (max_boost - 1) * (1 - h / h_ceiling)
        when h < h_ceiling, else multiplier = 1.

        Parameters
        ----------
        altitude : float
            Height above ground [m]. In NED, this is -z.
        base_thrust : float
            Thrust without ground effect [N].

        Returns
        -------
        modified_thrust : float
            Thrust with ground effect applied [N].
        """
        p = self.params
        if altitude >= p.ground_effect_ceiling or altitude < 0:
            return base_thrust

        # Linear interpolation of boost
        fraction = 1.0 - altitude / p.ground_effect_ceiling
        multiplier = 1.0 + (p.ground_effect_max_multiplier - 1.0) * fraction

        return base_thrust * multiplier

    def compute_forces_and_torques(
        self,
        velocity_world: np.ndarray,
        angular_velocity: np.ndarray,
        quaternion: np.ndarray,
    ) -> tuple:
        """
        Compute all aerodynamic forces and torques.

        Parameters
        ----------
        velocity_world : ndarray (3,)
            Velocity in world frame [m/s].
        angular_velocity : ndarray (3,)
            Angular velocity in body frame [rad/s].
        quaternion : ndarray (4,)
            Attitude quaternion [qw, qx, qy, qz].

        Returns
        -------
        aero_force_body : ndarray (3,)
            Total aerodynamic force in body frame [N].
        aero_torque_body : ndarray (3,)
            Total aerodynamic torque in body frame [N·m].
        """
        force = self.compute_drag_force(velocity_world, quaternion)
        torque = self.compute_rotational_drag(angular_velocity)
        return force, torque
