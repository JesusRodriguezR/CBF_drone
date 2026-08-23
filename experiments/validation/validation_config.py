"""
Validation Configuration
========================

Centralised settings for the sim-to-real digital twin validation.

Platform: Crazyflie 2.1 Brushless + Flow deck v2 (no OptiTrack).
"""

import numpy as np

# ── Crazyflie Radio ────────────────────────────────────────────────────
URI = "radio://0/80/2M/E7E7E7E7E7"

# ── Flight Parameters ──────────────────────────────────────────────────
FLIGHT_HEIGHT = 0.5          # default hover height [m]
CONTROL_RATE_HZ = 100        # position setpoint send rate
LOG_PERIOD_MS = 10           # 100 Hz onboard logging

# ── Safety ─────────────────────────────────────────────────────────────
MIN_BATTERY_V = 3.0          # land if vbat drops below
MAX_HEIGHT_M = 1.5           # absolute ceiling

# ── Data Directory ─────────────────────────────────────────────────────
DATA_DIR = "data"            # subfolder under experiments/validation/

# ── Logging Groups ─────────────────────────────────────────────────────
# Extended logging for full 6-DOF validation
LOG_GROUPS = {
    "pose": {
        "period_ms": LOG_PERIOD_MS,
        "variables": [
            ("stateEstimate.x",  "float"),
            ("stateEstimate.y",  "float"),
            ("stateEstimate.z",  "float"),
            ("stateEstimate.vx", "float"),
            ("stateEstimate.vy", "float"),
            ("stateEstimate.vz", "float"),
        ],
    },
    "attitude": {
        "period_ms": LOG_PERIOD_MS,
        "variables": [
            ("stabilizer.roll",  "float"),
            ("stabilizer.pitch", "float"),
            ("stabilizer.yaw",   "float"),
            ("gyro.x",           "float"),
            ("gyro.y",           "float"),
            ("gyro.z",           "float"),
        ],
    },
    "motors": {
        "period_ms": LOG_PERIOD_MS,
        "variables": [
            ("motor.m1", "uint16_t"),
            ("motor.m2", "uint16_t"),
            ("motor.m3", "uint16_t"),
            ("motor.m4", "uint16_t"),
            ("pm.vbat",  "float"),
        ],
    },
    "accel": {
        "period_ms": LOG_PERIOD_MS,
        "variables": [
            ("acc.x", "float"),
            ("acc.y", "float"),
            ("acc.z", "float"),
        ],
    },
}

# CSV column order (flat, for writing)
CSV_COLUMNS = [
    "t",
    "x", "y", "z",
    "vx", "vy", "vz",
    "roll", "pitch", "yaw",
    "gyro_x", "gyro_y", "gyro_z",
    "m1", "m2", "m3", "m4",
    "vbat",
    "acc_x", "acc_y", "acc_z",
    "x_ref", "y_ref", "z_ref", "yaw_ref",
    "manoeuvre",
]

# ── Validation Manoeuvres ──────────────────────────────────────────────
# Each manoeuvre is a callable: manoeuvre(t) → (x, y, z, yaw)
# Duration is in seconds.

MANOEUVRES = {
    "hover": {
        "duration": 10.0,
        "description": "Stationary hover — validates altitude PID + gravity model",
    },
    "step_z": {
        "duration": 15.0,
        "description": "Altitude steps 0.3→0.7→0.3 m — validates thrust model",
    },
    "step_xy": {
        "duration": 15.0,
        "description": "Position steps ±0.3 m in X then Y — validates lateral dynamics",
    },
    "circle": {
        "duration": 20.0,
        "description": "0.4 m radius circle — validates coupled dynamics",
    },
    "impulse": {
        "duration": 10.0,
        "description": "Quick position impulse + recovery — validates inertia & damping",
    },
}


# ── Manoeuvre Trajectory Generators ────────────────────────────────────

def traj_hover(t, *, x=0.0, y=0.0, z=0.5, yaw=0.0):
    """Stationary hover."""
    return np.array([x, y, z, yaw], dtype=np.float64)


def traj_step_z(t, *, z_low=0.3, z_high=0.7, period=5.0, yaw=0.0):
    """Altitude step: low → high → low every *period* seconds."""
    phase = (t % (2.0 * period)) / (2.0 * period)
    z = z_high if phase < 0.5 else z_low
    return np.array([0.0, 0.0, z, yaw], dtype=np.float64)


def traj_step_xy(t, *, offset=0.3, period=5.0, z=0.5, yaw=0.0):
    """Position step: origin → +X → origin → +Y → origin, repeating."""
    cycle = 4.0 * period
    phase = (t % cycle) / period
    seg = int(phase) % 4
    if seg == 0:
        return np.array([offset, 0.0, z, yaw], dtype=np.float64)
    elif seg == 1:
        return np.array([0.0, 0.0, z, yaw], dtype=np.float64)
    elif seg == 2:
        return np.array([0.0, offset, z, yaw], dtype=np.float64)
    else:
        return np.array([0.0, 0.0, z, yaw], dtype=np.float64)


def traj_circle(t, *, radius=0.4, period=10.0, z=0.5, yaw=0.0):
    """Horizontal circle at constant altitude."""
    w = 2.0 * np.pi / period
    return np.array([radius * np.cos(w * t),
                     radius * np.sin(w * t),
                     z, yaw], dtype=np.float64)


def traj_impulse(t, *, amplitude=0.4, pulse_start=2.0, pulse_end=3.0,
                 z=0.5, yaw=0.0):
    """Quick step in X then return to origin — excites transient response."""
    if pulse_start <= t < pulse_end:
        return np.array([amplitude, 0.0, z, yaw], dtype=np.float64)
    return np.array([0.0, 0.0, z, yaw], dtype=np.float64)


TRAJECTORY_FNS = {
    "hover":   traj_hover,
    "step_z":  traj_step_z,
    "step_xy": traj_step_xy,
    "circle":  traj_circle,
    "impulse": traj_impulse,
}
