"""
Digital Twin Unit & Integration Tests
=======================================

Validates correctness of the Crazyflie 2.1 digital twin:
    1. Motor mixing matrix signs
    2. Hover equilibrium
    3. Quaternion normalization stability
    4. Symmetry (equal motors → zero angular accel)
    5. Free-fall gravity validation
    6. Motor dynamics response
    7. Sensor noise characteristics
"""

import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from crazyflie_params import CrazyflieParams
from rigid_body_dynamics import (
    RigidBodyDynamics, make_state, quat_normalize,
    quat_to_rotation_matrix, quat_to_euler, euler_to_quat,
    quat_multiply, quat_rotate_vector,
    POS, VEL, QUAT, OMEGA,
)
from motor_model import Motor, MotorGroup
from motor_mixing import MotorMixer
from aerodynamics_model import AerodynamicsModel
from sensor_model import IMUModel
from crazyflie_sim import CrazyflieSim, SimConfig


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def params():
    return CrazyflieParams()

@pytest.fixture
def sim():
    config = SimConfig(dt=0.001, enable_sensors=False, log_data=False)
    s = CrazyflieSim(config=config)
    s.reset(position=np.array([0.0, 0.0, -1.0]))
    return s


# ── 1. Motor Mixing Matrix ─────────────────────────────────────────────

class TestMotorMixing:
    """Verify mixing matrix matches Crazyflie firmware conventions."""

    def test_pure_thrust_equal_motors(self, params):
        """Pure thrust command → all motors equal."""
        mixer = MotorMixer(params)
        cmd = np.array([0.5, 0.0, 0.0, 0.0])  # thrust only
        pwms = mixer.mix_forward(cmd)
        assert np.allclose(pwms, 0.5), f"Expected all 0.5, got {pwms}"

    def test_positive_roll_increases_m4(self, params):
        """Positive roll → M4 increases, M2 decreases (right roll in NED)."""
        mixer = MotorMixer(params)
        cmd = np.array([0.5, 0.1, 0.0, 0.0])  # thrust + roll
        pwms = mixer.mix_forward(cmd)
        # M4 = T + roll, M2 = T - roll
        assert pwms[3] > pwms[1], f"M4={pwms[3]} should > M2={pwms[1]}"

    def test_positive_pitch_increases_m1(self, params):
        """Positive pitch → M1 increases, M3 decreases."""
        mixer = MotorMixer(params)
        cmd = np.array([0.5, 0.0, 0.1, 0.0])  # thrust + pitch
        pwms = mixer.mix_forward(cmd)
        # M1 = T + pitch, M3 = T - pitch
        assert pwms[0] > pwms[2], f"M1={pwms[0]} should > M3={pwms[2]}"

    def test_positive_yaw_increases_m1_m3(self, params):
        """Positive yaw → M1,M3 increase (CW motors), M2,M4 decrease (CCW)."""
        mixer = MotorMixer(params)
        cmd = np.array([0.5, 0.0, 0.0, 0.1])  # thrust + yaw
        pwms = mixer.mix_forward(cmd)
        # M1,M3 = T + yaw; M2,M4 = T - yaw
        assert pwms[0] > pwms[1], f"M1={pwms[0]} should > M2={pwms[1]}"
        assert pwms[2] > pwms[3], f"M3={pwms[2]} should > M4={pwms[3]}"

    def test_pwm_clamping(self, params):
        """Motor PWMs should be clamped to [0, 1]."""
        mixer = MotorMixer(params)
        cmd = np.array([0.9, 0.0, 0.5, 0.0])  # thrust + large pitch
        pwms = mixer.mix_forward(cmd)
        assert np.all(pwms >= 0.0) and np.all(pwms <= 1.0)

    def test_body_wrench_pure_thrust(self, params):
        """Equal thrusts → net force in -Z, zero torque."""
        mixer = MotorMixer(params)
        thrusts = np.array([0.1, 0.1, 0.1, 0.1])
        # For zero yaw torque, need CW and CCW to cancel
        # M1(CW), M2(CCW), M3(CW), M4(CCW) → with equal thrusts,
        # yaw torques should nearly cancel
        yaw_torques = np.array([
            -params.k_m * 100**2,   # M1 CW
            +params.k_m * 100**2,   # M2 CCW
            -params.k_m * 100**2,   # M3 CW
            +params.k_m * 100**2,   # M4 CCW
        ])
        force, torque = mixer.compute_body_wrench(thrusts, yaw_torques)

        assert force[2] < 0, "Thrust should be in -Z direction (upward in NED)"
        assert np.isclose(force[2], -0.4, atol=1e-6)
        assert np.isclose(torque[0], 0.0, atol=1e-6), "Roll torque should be zero"
        assert np.isclose(torque[1], 0.0, atol=1e-6), "Pitch torque should be zero"
        assert np.isclose(torque[2], 0.0, atol=1e-6), "Yaw torque should cancel"


# ── 2. Hover Equilibrium ───────────────────────────────────────────────

class TestHoverEquilibrium:
    """Verify that thrust = weight produces stable hover."""

    def test_hover_acceleration_near_zero(self, sim):
        """At hover thrust, vertical acceleration should be ≈ 0."""
        params = sim.params

        # Compute hover PWM (per motor: weight/4)
        a = params.pwm_thrust_a
        b = params.pwm_thrust_b
        hover_thrust_per_motor = params.weight / 4.0
        discriminant = b**2 + 4 * a * hover_thrust_per_motor
        hover_pwm = (-b + np.sqrt(discriminant)) / (2 * a)

        command = np.array([hover_pwm] * 4)

        # Pre-spin motors to steady state (5× time constant)
        spinup_steps = int(5 * params.motor_time_constant / sim.config.dt)
        for _ in range(spinup_steps):
            sim.step(command, command_type='pwm')

        # Now zero out any velocity accumulated during spinup
        sim.state[VEL] = np.zeros(3)

        # Run for 2 seconds of hover
        for _ in range(2000):
            state, _ = sim.step(command, command_type='pwm')

        altitude = -state[2]
        # Should still be near 1.0 m (started at 1.0 m)
        assert abs(altitude - 1.0) < 0.05, \
            f"Altitude drifted to {altitude:.4f} m (expected ~1.0)"

    def test_hover_zero_angular_rates(self, sim):
        """In hover with equal motors, angular rates should stay ≈ 0."""
        params = sim.params
        a = params.pwm_thrust_a
        b = params.pwm_thrust_b
        hover_pwm = (-b + np.sqrt(b**2 + 4*a*(params.weight/4.0))) / (2*a)
        command = np.array([hover_pwm] * 4)

        for _ in range(1000):
            state, _ = sim.step(command, command_type='pwm')

        omega = state[OMEGA]
        assert np.all(np.abs(omega) < 0.01), \
            f"Angular rates should be ~0, got {omega}"


# ── 3. Quaternion Stability ────────────────────────────────────────────

class TestQuaternion:
    """Quaternion should remain unit-norm over long simulations."""

    def test_unit_norm_preservation(self, sim):
        """Quaternion norm should stay ≈ 1.0 over 10 seconds."""
        params = sim.params
        a = params.pwm_thrust_a
        b = params.pwm_thrust_b
        hover_pwm = (-b + np.sqrt(b**2 + 4*a*(params.weight/4.0))) / (2*a)

        # Add slight asymmetry to exercise rotation
        command = np.array([hover_pwm*1.01, hover_pwm*0.99,
                           hover_pwm*1.01, hover_pwm*0.99])

        for i in range(10000):  # 10 seconds
            state, _ = sim.step(command, command_type='pwm')

        quat_norm = np.linalg.norm(state[QUAT])
        assert abs(quat_norm - 1.0) < 1e-6, \
            f"Quaternion norm = {quat_norm}, expected 1.0"

    def test_euler_roundtrip(self):
        """Euler → quaternion → Euler should be identity."""
        angles = np.array([0.1, -0.2, 0.3])  # roll, pitch, yaw
        q = euler_to_quat(*angles)
        recovered = quat_to_euler(q)
        assert np.allclose(angles, recovered, atol=1e-10), \
            f"Roundtrip failed: {angles} → {recovered}"

    def test_rotation_matrix_orthogonal(self):
        """Rotation matrix from quaternion should be orthogonal."""
        q = euler_to_quat(0.5, -0.3, 1.2)
        R = quat_to_rotation_matrix(q)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)


# ── 4. Symmetry ────────────────────────────────────────────────────────

class TestSymmetry:
    """Equal motor commands should produce zero angular acceleration."""

    def test_zero_torque_from_equal_thrust(self, params):
        """Equal thrusts on all motors → zero roll and pitch torque."""
        mixer = MotorMixer(params)
        arm = params.arm_length
        equal_thrust = 0.08  # N per motor
        thrusts = np.array([equal_thrust] * 4)

        # Compute yaw torques (should cancel for equal speeds)
        speed = np.sqrt(equal_thrust / params.k_f)
        yaw_torques = np.array([
            -params.k_m * speed**2,   # M1 CW
            +params.k_m * speed**2,   # M2 CCW
            -params.k_m * speed**2,   # M3 CW
            +params.k_m * speed**2,   # M4 CCW
        ])

        _, torque = mixer.compute_body_wrench(thrusts, yaw_torques)
        assert np.allclose(torque, 0.0, atol=1e-10), \
            f"Torque should be zero, got {torque}"


# ── 5. Free Fall ───────────────────────────────────────────────────────

class TestFreeFall:
    """Zero thrust should produce free-fall at g."""

    def test_freefall_acceleration(self):
        """Vertical acceleration in free fall should be g (downward in NED)."""
        config = SimConfig(
            dt=0.001,
            enable_sensors=False,
            enable_ground_collision=False,
            enable_aerodynamics=False,
            log_data=False,
        )
        sim = CrazyflieSim(config=config)
        sim.reset(position=np.array([0.0, 0.0, -10.0]))

        command = np.array([0.0, 0.0, 0.0, 0.0])

        # Run 0.5 seconds
        for _ in range(500):
            state, _ = sim.step(command, command_type='pwm')

        # Velocity should be g * t = 9.81 * 0.5 ≈ 4.905 m/s downward (+ in NED)
        vz = state[5]
        expected_vz = 9.81 * 0.5
        assert abs(vz - expected_vz) < 0.01, \
            f"Vz = {vz:.4f}, expected {expected_vz:.4f}"

        # Position: z = z0 + 0.5*g*t^2 = -10 + 0.5*9.81*0.25 = -8.77
        z = state[2]
        expected_z = -10.0 + 0.5 * 9.81 * 0.5**2
        assert abs(z - expected_z) < 0.01, \
            f"z = {z:.4f}, expected {expected_z:.4f}"


# ── 6. Motor Dynamics ──────────────────────────────────────────────────

class TestMotorDynamics:
    """Motor first-order response characteristics."""

    def test_motor_reaches_target(self, params):
        """Motor should converge to target speed after several time constants."""
        motor = Motor(params, spin_direction=1)
        target_pwm = 0.5

        # Run for 5× time constant (should reach >99%)
        dt = 0.001
        n_steps = int(5 * params.motor_time_constant / dt)
        for _ in range(n_steps):
            motor.step(target_pwm, dt)

        target_speed = motor.pwm_to_speed(target_pwm)
        assert abs(motor.speed - target_speed) / target_speed < 0.01, \
            f"Motor speed {motor.speed:.1f} didn't converge to {target_speed:.1f}"

    def test_motor_starts_at_zero(self, params):
        """Motor should start at zero speed."""
        motor = Motor(params)
        assert motor.speed == 0.0

    def test_thrust_positive(self, params):
        """Thrust should always be non-negative."""
        motor = Motor(params)
        motor.speed = 1000.0
        assert motor.speed_to_thrust() > 0


# ── 7. Sensor Noise ────────────────────────────────────────────────────

class TestSensors:
    """Verify sensor noise characteristics."""

    def test_imu_noise_magnitude(self, params):
        """IMU noise should be reasonable (not gigantic or zero)."""
        imu = IMUModel(params, seed=42)
        state = make_state(position=np.array([0, 0, -1]))

        # Body force at hover
        hover_force = np.array([0.0, 0.0, -params.weight])

        readings = []
        for _ in range(1000):
            r = imu.measure(state, hover_force, params.mass)
            readings.append(r['accel'])

        readings = np.array(readings)
        noise_std = np.std(readings, axis=0)

        # Should have some noise but not crazy
        assert np.all(noise_std > 0.0), "Sensor noise should be non-zero"
        assert np.all(noise_std < 5.0), f"Noise too large: {noise_std}"

    def test_gyro_zero_when_stationary(self, params):
        """Gyro mean should be ≈ 0 for a stationary drone."""
        imu = IMUModel(params, seed=42)
        state = make_state()  # stationary

        hover_force = np.array([0.0, 0.0, -params.weight])
        readings = [imu.measure(state, hover_force, params.mass)['gyro']
                    for _ in range(1000)]
        mean_gyro = np.mean(readings, axis=0)

        # Mean should be near zero (bias walks slowly)
        assert np.all(np.abs(mean_gyro) < 0.1), \
            f"Mean gyro too large: {mean_gyro}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
