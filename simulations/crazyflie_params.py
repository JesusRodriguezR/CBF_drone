"""
Crazyflie 2.1 Physical Parameters
==================================

Centralized dataclass containing all physical constants for the stock
Crazyflie 2.1 quadrotor (brushed motors, 45 mm propellers).

Sources:
    - Bitcraze official specs (bitcraze.io)
    - System identification: Förster (2015), Snider (2023)
    - Crazyflie firmware (github.com/bitcraze/crazyflie-firmware)

Coordinate convention: NED (North-East-Down)
    +X = forward (nose)
    +Y = right
    +Z = down

Units: SI throughout (kg, m, s, rad, N)
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class CrazyflieParams:
    """All physical parameters for the Crazyflie 2.1 digital twin."""

    # ── Mass & Geometry ─────────────────────────────────────────────────
    mass: float = 0.033          # Total mass with battery [kg]
    arm_length: float = 0.046   # Center-to-motor distance [m] (92 mm diagonal / 2)

    # Inertia tensor (body frame, diagonal-dominant) [kg·m²]
    # From system identification (Förster 2015)
    Ixx: float = 16.571710e-6
    Iyy: float = 16.655602e-6
    Izz: float = 29.261652e-6
    # Cross-terms (small but included for fidelity)
    Ixy: float = 0.830806e-6
    Ixz: float = 0.718277e-6
    Iyz: float = 1.800197e-6

    # ── Motor & Propeller ───────────────────────────────────────────────
    # Stock: 7×16 mm coreless brushed DC motors + 45 mm propellers

    # Thrust coefficient: F = k_f * omega^2  [N / (rad/s)^2]
    k_f: float = 3.16e-10 * (60.0 / (2.0 * np.pi))**2  # converted from N/RPM² to N/(rad/s)²
    # NOTE: 3.16e-10 N/RPM² → multiply by (60/2π)² to get N/(rad/s)²

    # Torque coefficient: M = k_m * omega^2  [N·m / (rad/s)^2]
    # Empirical ratio k_m/k_f ≈ 0.005–0.025 m for small rotors
    k_m: float = 7.94e-12 * (60.0 / (2.0 * np.pi))**2  # converted from N·m/RPM²

    # First-order motor time constant [s]
    motor_time_constant: float = 0.02

    # Motor speed limits [rad/s]
    motor_min_rpm: float = 0.0
    motor_max_rpm: float = 21666.0   # ~21,666 RPM for stock motors

    @property
    def motor_min_speed(self) -> float:
        """Minimum motor angular velocity [rad/s]."""
        return self.motor_min_rpm * 2.0 * np.pi / 60.0

    @property
    def motor_max_speed(self) -> float:
        """Maximum motor angular velocity [rad/s]."""
        return self.motor_max_rpm * 2.0 * np.pi / 60.0

    # PWM-to-thrust empirical quadratic (PWM as fraction 0–1, thrust in N)
    # Original: thrust_N = 0.091492681 * pwm_16bit^2 + 0.067673604 * pwm_16bit
    # where pwm_16bit ∈ [0, 65535]. Normalised to pwm ∈ [0, 1]:
    pwm_thrust_a: float = 0.091492681 * 65535.0**2  # ≈ 392956.0
    pwm_thrust_b: float = 0.067673604 * 65535.0     # ≈ 4434.7

    # Maximum thrust per motor [N]  (~15 g per motor, 60 g total)
    max_thrust_per_motor: float = 0.1471   # 4 * 15g * 9.81 / 4

    # ── Aerodynamics ────────────────────────────────────────────────────
    # Linear drag coefficient (body frame) [N·s/m]
    # Small for indoor micro-UAV; tunable
    k_drag_xy: float = 0.01     # lateral drag
    k_drag_z: float = 0.005     # vertical drag

    # Rotational drag coefficient [N·m·s/rad]
    k_rot_drag: float = 1.0e-5

    # Ground effect: thrust multiplier when altitude < ground_effect_ceiling
    ground_effect_ceiling: float = 0.10   # [m] ≈ 2× rotor radius
    ground_effect_max_multiplier: float = 1.15  # 15 % boost at ground level

    # ── Sensor Noise (BMI088 equivalent) ────────────────────────────────
    # Accelerometer
    accel_noise_density: float = 150e-6 * 9.81  # 150 µg/√Hz → m/s²/√Hz
    accel_bias_instability: float = 40e-3 * 9.81  # ±40 mg → m/s²

    # Gyroscope
    gyro_noise_density: float = 0.014 * np.pi / 180.0  # 0.014 °/s/√Hz → rad/s/√Hz
    gyro_bias_instability: float = 1.0 * np.pi / 180.0  # ±1 °/s → rad/s

    # IMU sample rate [Hz]
    imu_sample_rate: float = 1000.0

    # ── Environment ─────────────────────────────────────────────────────
    g: float = 9.81       # gravitational acceleration [m/s²]
    rho: float = 1.225    # air density at sea level [kg/m³]

    # ── Derived Properties ──────────────────────────────────────────────
    @property
    def inertia_matrix(self) -> np.ndarray:
        """Full 3×3 inertia tensor [kg·m²]."""
        return np.array([
            [self.Ixx,  self.Ixy, self.Ixz],
            [self.Ixy,  self.Iyy, self.Iyz],
            [self.Ixz,  self.Iyz, self.Izz],
        ])

    @property
    def inertia_matrix_inv(self) -> np.ndarray:
        """Inverse of the inertia tensor."""
        return np.linalg.inv(self.inertia_matrix)

    @property
    def weight(self) -> float:
        """Weight force magnitude [N]."""
        return self.mass * self.g

    @property
    def hover_thrust_per_motor(self) -> float:
        """Thrust required per motor for hover [N]."""
        return self.weight / 4.0

    @property
    def hover_motor_speed(self) -> float:
        """Motor speed for hover [rad/s]."""
        return np.sqrt(self.hover_thrust_per_motor / self.k_f)

    def __post_init__(self):
        """Validate parameters after initialization."""
        assert self.mass > 0, "Mass must be positive"
        assert self.arm_length > 0, "Arm length must be positive"
        assert self.k_f > 0, "Thrust coefficient must be positive"
        assert self.k_m > 0, "Torque coefficient must be positive"
        assert self.motor_time_constant > 0, "Motor time constant must be positive"
