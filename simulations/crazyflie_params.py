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

    @classmethod
    def brushless(cls) -> 'CrazyflieParams':
        """
        Factory for Crazyflie 2.1 Brushless + Flow deck v2.

        Hardware: 08028-10000KV motors, 55mm/35mm-pitch propellers,
                  1S 5A ESC (BLHeli_S/Bluejay), Flow deck v2.

        Key differences from stock brushed CF 2.1:
            - 55 mm propellers (vs 45 mm) → higher k_f
            - Brushless ESC response ~10 ms (vs ~20 ms brushed)
            - ~50% more thrust per motor (30 g vs ~15 g max)
            - Heavier (~34–37 g vs 27–33 g)

        Sources:
            - Bitcraze product specs (bitcraze.io/products/crazyflie-2-1-brushless)
            - Gräfe, Scherer, Hönig, Trimpe, "How to Model Your Crazyflie
              Brushless", ICRA 2026 (arXiv:2603.05944)
            - Bitcraze 08028-10000KV motor datasheet: 30 g max thrust,
              peak 1.8 A / 7.2 W, ~5 g/W efficiency at 10 g thrust
            - Frame: 100 mm diagonal (50 mm center-to-motor)
        """
        # ── Thrust coefficient derivation ──
        # Max thrust = 30 g = 0.2943 N per motor at ~25000 RPM
        # F = k_f * omega^2  →  k_f = F / omega^2
        # omega_max = 25000 * 2π/60 ≈ 2618 rad/s
        # k_f = 0.2943 / (2618^2) ≈ 4.30e-8  [N/(rad/s)^2]
        # In RPM units: k_f_rpm = 0.2943 / 25000^2 ≈ 4.71e-10 [N/RPM^2]
        _k_f_rpm = 4.71e-10                # N/RPM²
        _k_m_rpm = _k_f_rpm * 0.0060       # k_m/k_f ≈ 0.006 m for 55 mm props
        _rpm_to_rad = (60.0 / (2.0 * np.pi))**2  # conversion factor RPM²→(rad/s)²

        return cls(
            # ── Mass & Geometry ──
            mass=0.037,               # ~37 g: base 32g + Flow deck ~3.5g + battery
            arm_length=0.050,         # 100 mm diagonal / 2 (Bitcraze spec)

            # Inertia: scaled from Förster 2015 values by mass ratio
            # (m_bl/m_stock = 37/33 = 1.12) and arm length ratio
            # (50/46 = 1.087)². Heavier motor tips increase Ixx/Iyy.
            Ixx=20.5e-6,
            Iyy=20.6e-6,
            Izz=35.5e-6,
            Ixy=0.9e-6,
            Ixz=0.8e-6,
            Iyz=1.9e-6,

            # ── Motor & Propeller ──
            # 08028-10000KV motors + 55mm/35mm-pitch propellers
            k_f=_k_f_rpm * _rpm_to_rad,
            k_m=_k_m_rpm * _rpm_to_rad,

            # Brushless ESC response is faster than brushed DC
            motor_time_constant=0.010,    # ~10 ms (BLHeli_S)

            motor_min_rpm=0.0,
            motor_max_rpm=25000.0,        # 10000 KV × 4.2 V ≈ 42000 RPM unloaded
                                          # ~25000 RPM loaded with 55mm props

            # PWM-to-thrust: quadratic fit for brushless
            # F = a * pwm^2 + b * pwm  where pwm ∈ [0, 1] (normalised)
            # Anchor points (derived from real hover flight data):
            #   pwm = 0.0  → F = 0 N       (motors off)
            #   pwm ≈ 0.57 → F = 0.0907 N  (hover at 37 g, from logged M1≈37280/65535)
            #   pwm = 1.0  → F = 0.2943 N  (max thrust: 30 g per motor)
            # Solving: a·p² + b·p = F_hover  and  a + b = F_max
            pwm_thrust_a=0.3126,    # N (quadratic term for pwm ∈ [0,1])
            pwm_thrust_b=-0.0183,   # N (linear term for pwm ∈ [0,1])

            max_thrust_per_motor=0.2943,  # 30 g × 9.81 m/s²

            # ── Aerodynamics ──
            k_drag_xy=0.012,              # slightly higher with 55mm props
            k_drag_z=0.006,
            k_rot_drag=1.5e-5,            # larger props → more rot drag
            ground_effect_ceiling=0.12,    # ~2× rotor radius (55mm/2)
            ground_effect_max_multiplier=1.15,

            # ── Sensor Noise (same IMU: BMI088) ──
            accel_noise_density=150e-6 * 9.81,
            accel_bias_instability=40e-3 * 9.81,
            gyro_noise_density=0.014 * np.pi / 180.0,
            gyro_bias_instability=1.0 * np.pi / 180.0,
            imu_sample_rate=1000.0,

            # ── Environment ──
            g=9.81,
            rho=1.225,
        )
