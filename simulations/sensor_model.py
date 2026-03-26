"""
Sensor Model
=============

Simulated IMU (accelerometer + gyroscope) matching the BMI088 on the
Crazyflie 2.1, with realistic noise characteristics.

Noise model per axis:
    measurement = true_value + bias + white_noise

    white_noise ~ N(0, σ²)     where σ = noise_density * √(sample_rate)
    bias: random walk          Δbias ~ N(0, σ_bias²·dt)

Optionally includes a barometer model (BMP388) for altitude estimation.
"""

import numpy as np
from crazyflie_params import CrazyflieParams
from rigid_body_dynamics import (
    quat_to_rotation_matrix,
    POS, VEL, QUAT, OMEGA,
)


class IMUModel:
    """
    Simulated 6-axis IMU (3-axis accelerometer + 3-axis gyroscope).
    """

    def __init__(self, params: CrazyflieParams, seed: int = None):
        """
        Parameters
        ----------
        params : CrazyflieParams
            Physical parameters (noise densities, sample rate, etc.).
        seed : int, optional
            Random seed for reproducibility.
        """
        self.params = params
        self.rng = np.random.default_rng(seed)

        # Sample rate
        self.dt_sensor = 1.0 / params.imu_sample_rate

        # Noise standard deviations (per sample)
        self.accel_noise_std = params.accel_noise_density * np.sqrt(params.imu_sample_rate)
        self.gyro_noise_std = params.gyro_noise_density * np.sqrt(params.imu_sample_rate)

        # Bias random walk rate
        self.accel_bias_walk_std = params.accel_bias_instability * np.sqrt(self.dt_sensor)
        self.gyro_bias_walk_std = params.gyro_bias_instability * np.sqrt(self.dt_sensor)

        # Current biases [3-axis each]
        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)

    def reset(self, seed: int = None):
        """Reset biases and optionally reseed."""
        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def _update_bias(self):
        """Random-walk update of sensor biases."""
        self.accel_bias += self.rng.normal(0, self.accel_bias_walk_std, 3)
        self.gyro_bias += self.rng.normal(0, self.gyro_bias_walk_std, 3)

        # Clamp biases to reasonable range
        max_accel_bias = self.params.accel_bias_instability * 3.0
        max_gyro_bias = self.params.gyro_bias_instability * 3.0
        self.accel_bias = np.clip(self.accel_bias, -max_accel_bias, max_accel_bias)
        self.gyro_bias = np.clip(self.gyro_bias, -max_gyro_bias, max_gyro_bias)

    def measure(
        self,
        state: np.ndarray,
        force_body: np.ndarray,
        mass: float,
    ) -> dict:
        """
        Generate noisy IMU measurements from the true state.

        Parameters
        ----------
        state : ndarray (13,)
            True state vector.
        force_body : ndarray (3,)
            Net non-gravitational force in body frame [N].
            (Needed for proper accelerometer reading.)
        mass : float
            Vehicle mass [kg].

        Returns
        -------
        imu_data : dict
            'accel' : ndarray (3,) — accelerometer reading [m/s²]
            'gyro'  : ndarray (3,) — gyroscope reading [rad/s]
        """
        # ── True values ──

        # Accelerometer measures specific force (non-gravitational acceleration)
        # in body frame. This is what a real accelerometer reads:
        #   a_measured = (F_body / m) — or equivalently, total_accel - g_body
        # In hover, accelerometer reads -g in body Z (pointing up in NED).
        q = state[QUAT]
        R = quat_to_rotation_matrix(q)
        gravity_world = np.array([0.0, 0.0, self.params.g])
        gravity_body = R.T @ gravity_world

        # Specific force = total_body_force / mass
        # The accelerometer reads specific force = a_body - g_body
        # where a_body is the body-frame acceleration from all forces except gravity.
        true_accel = force_body / mass - gravity_body

        # Gyroscope measures angular velocity in body frame
        true_gyro = state[OMEGA].copy()

        # ── Apply noise ──
        self._update_bias()

        noisy_accel = (
            true_accel
            + self.accel_bias
            + self.rng.normal(0, self.accel_noise_std, 3)
        )

        noisy_gyro = (
            true_gyro
            + self.gyro_bias
            + self.rng.normal(0, self.gyro_noise_std, 3)
        )

        return {
            'accel': noisy_accel,
            'gyro': noisy_gyro,
        }


class BarometerModel:
    """
    Simple barometer model (BMP388 equivalent).

    Converts altitude to pressure with noise.
    """

    def __init__(
        self,
        noise_std_pa: float = 1.2,     # Pressure noise [Pa]
        bias_pa: float = 0.0,          # Static bias [Pa]
        sample_rate: float = 50.0,     # [Hz]
        seed: int = None,
    ):
        self.noise_std = noise_std_pa
        self.bias = bias_pa
        self.sample_rate = sample_rate
        self.rng = np.random.default_rng(seed)

        # ISA sea-level reference
        self.P0 = 101325.0      # [Pa]
        self.T0 = 288.15        # [K]
        self.L = 0.0065         # Temperature lapse rate [K/m]
        self.R = 287.05         # Gas constant [J/(kg·K)]
        self.g = 9.81

    def reset(self, seed: int = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def altitude_to_pressure(self, altitude: float) -> float:
        """Convert geometric altitude [m] to pressure [Pa] using ISA model."""
        return self.P0 * (1 - self.L * altitude / self.T0) ** (self.g / (self.R * self.L))

    def measure(self, altitude: float) -> dict:
        """
        Generate noisy barometer reading.

        Parameters
        ----------
        altitude : float
            True altitude above sea level [m]. In NED, altitude = -z.

        Returns
        -------
        baro_data : dict
            'pressure' : float — pressure [Pa]
            'altitude_est' : float — altitude estimate from pressure [m]
        """
        true_pressure = self.altitude_to_pressure(altitude)
        noisy_pressure = (
            true_pressure
            + self.bias
            + self.rng.normal(0, self.noise_std)
        )

        # Invert barometric formula for altitude estimate
        alt_est = (self.T0 / self.L) * (1 - (noisy_pressure / self.P0) ** (self.R * self.L / self.g))

        return {
            'pressure': noisy_pressure,
            'altitude_est': alt_est,
        }
