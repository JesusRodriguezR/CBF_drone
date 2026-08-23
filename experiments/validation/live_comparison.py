"""
Live Sim-to-Real Comparison
=============================

Runs the digital twin simulation in real-time alongside the physical
Crazyflie Brushless, displaying side-by-side telemetry for immediate
visual validation.

The script:
  1. Connects to the Crazyflie via Crazyradio + Flow deck v2
  2. Flies a chosen manoeuvre, sending position setpoints
  3. Reads motor PWMs from the physical drone at 100 Hz
  4. Feeds those PWMs into the sim in real-time
  5. Displays live comparison (terminal + CSV logging)

Usage
-----
    python live_comparison.py                     # hover (default)
    python live_comparison.py --manoeuvre circle   # circle trajectory
    python live_comparison.py --duration 15        # custom duration
"""

import argparse
import csv
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ── cflib imports ──────────────────────────────────────────────────────
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger

# ── Simulation imports ────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
SIM_DIR = SCRIPT_DIR.parent.parent / "simulations"
CTRL_DIR = SCRIPT_DIR.parent.parent / "control_arq_scripts"
sys.path.insert(0, str(SIM_DIR))
sys.path.insert(0, str(CTRL_DIR))

from crazyflie_params import CrazyflieParams
from crazyflie_sim import CrazyflieSim, SimConfig
from rigid_body_dynamics import quat_to_euler, quat_to_rotation_matrix, POS, VEL, QUAT, OMEGA
from pid_controller import CrazyflieController

import validation_config as cfg

# ── 3D visualisation (optional, enabled with --viz) ───────────────────
try:
    import pygame
    from pygame.locals import DOUBLEBUF, OPENGL, QUIT
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from realtime_sim import (draw_ground_grid, draw_crazyflie,
                              draw_trail, draw_altitude_line, COL_SKY)
    _HAS_VIZ = True
except ImportError:
    _HAS_VIZ = False

# Position control constants
KP_XY = 1.0                          # position → tilt gain [rad/m]
KD_XY = 0.4                          # velocity damping [rad/(m/s)]
MAX_TILT = 15.0 * math.pi / 180.0    # max roll/pitch command [rad]


# ═══════════════════════════════════════════════════════════════════════
# Drone State (async log accumulator)
# ═══════════════════════════════════════════════════════════════════════

class DroneState:
    """Thread-safe accumulator for async log callbacks."""

    def __init__(self):
        self._data = {
            "x": 0.0, "y": 0.0, "z": 0.0,
            "vx": 0.0, "vy": 0.0, "vz": 0.0,
            "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
            "gyro_x": 0.0, "gyro_y": 0.0, "gyro_z": 0.0,
            "m1": 0, "m2": 0, "m3": 0, "m4": 0,
            "vbat": 0.0,
            "acc_x": 0.0, "acc_y": 0.0, "acc_z": 0.0,
        }

    def update_pose(self, timestamp, data, logconf):
        self._data["x"]  = data["stateEstimate.x"]
        self._data["y"]  = data["stateEstimate.y"]
        self._data["z"]  = data["stateEstimate.z"]
        self._data["vx"] = data["stateEstimate.vx"]
        self._data["vy"] = data["stateEstimate.vy"]
        self._data["vz"] = data["stateEstimate.vz"]

    def update_attitude(self, timestamp, data, logconf):
        self._data["roll"]   = data["stabilizer.roll"]
        self._data["pitch"]  = data["stabilizer.pitch"]
        self._data["yaw"]    = data["stabilizer.yaw"]
        self._data["gyro_x"] = data["gyro.x"]
        self._data["gyro_y"] = data["gyro.y"]
        self._data["gyro_z"] = data["gyro.z"]

    def update_motors(self, timestamp, data, logconf):
        self._data["m1"]   = data["motor.m1"]
        self._data["m2"]   = data["motor.m2"]
        self._data["m3"]   = data["motor.m3"]
        self._data["m4"]   = data["motor.m4"]
        self._data["vbat"] = data["pm.vbat"]

    def update_accel(self, timestamp, data, logconf):
        self._data["acc_x"] = data["acc.x"]
        self._data["acc_y"] = data["acc.y"]
        self._data["acc_z"] = data["acc.z"]

    def snapshot(self):
        return dict(self._data)


# ═══════════════════════════════════════════════════════════════════════
# Helpers (reused from collect_validation_flight.py)
# ═══════════════════════════════════════════════════════════════════════

_LOG_CALLBACKS = {
    "pose":     "update_pose",
    "attitude": "update_attitude",
    "motors":   "update_motors",
    "accel":    "update_accel",
}


def setup_logging(scf, state):
    configs = []
    for group_name, group_info in cfg.LOG_GROUPS.items():
        lc = LogConfig(name=group_name, period_in_ms=group_info["period_ms"])
        for var_name, var_type in group_info["variables"]:
            lc.add_variable(var_name, var_type)
        scf.cf.log.add_config(lc)
        lc.data_received_cb.add_callback(getattr(state, _LOG_CALLBACKS[group_name]))
        lc.start()
        configs.append(lc)
    return configs


def reset_estimator(scf):
    cf = scf.cf
    cf.param.set_value("kalman.resetEstimation", "1")
    time.sleep(0.1)
    cf.param.set_value("kalman.resetEstimation", "0")
    print("  Waiting for estimator ...", end="", flush=True)
    time.sleep(5.0)

    log_config = LogConfig(name="z_check", period_in_ms=100)
    log_config.add_variable("stateEstimate.z", "float")
    z_samples = []
    with SyncLogger(scf, log_config) as logger:
        for log_entry in logger:
            z_samples.append(log_entry[1]["stateEstimate.z"])
            if len(z_samples) >= 10:
                break
    z_mean = sum(z_samples) / len(z_samples)
    print(f" z={z_mean:.3f} m")
    if z_mean < 0.001:
        print("  [WARN] Flow deck reading very low — check surface/deck")
    print(f"  [OK] Estimator ready (ground ~ {z_mean*100:.1f} cm)")


def arm_motors(scf):
    cf = scf.cf
    print("  Clearing crash flags ...", end="", flush=True)
    cf.platform.send_crash_recovery_request()
    time.sleep(0.5)
    print(" OK")
    print("  Arming brushless ...", end="", flush=True)
    cf.platform.send_arming_request(True)
    time.sleep(1.0)
    cf.commander.send_setpoint(0, 0, 0, 0)
    time.sleep(0.1)
    print(" ARMED")


# ═══════════════════════════════════════════════════════════════════════
# Live Display
# ═══════════════════════════════════════════════════════════════════════

def print_header():
    """Print the comparison table header."""
    print()
    print(f"  {'':>6}  {'--- PHYSICAL ---':^30}  {'--- SIMULATION ---':^30}  {'--- ERROR ---':^20}")
    print(f"  {'t':>6}  {'x':>8} {'y':>8} {'z':>8}  {'x':>8} {'y':>8} {'z':>8}  {'dx':>6} {'dy':>6} {'dz':>6}")
    print(f"  {'─'*6}  {'─'*8} {'─'*8} {'─'*8}  {'─'*8} {'─'*8} {'─'*8}  {'─'*6} {'─'*6} {'─'*6}")


def print_live_row(t, snap, sim_state):
    """Print one row of the live comparison."""
    sim_pos = sim_state[POS]
    sim_z = -sim_pos[2]  # NED to altitude

    dx = sim_pos[0] - snap["x"]
    dy = sim_pos[1] - snap["y"]
    dz = sim_z - snap["z"]

    print(f"  {t:6.2f}  "
          f"{snap['x']:8.4f} {snap['y']:8.4f} {snap['z']:8.4f}  "
          f"{sim_pos[0]:8.4f} {sim_pos[1]:8.4f} {sim_z:8.4f}  "
          f"{dx:+6.3f} {dy:+6.3f} {dz:+6.3f}", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# Main Live Comparison
# ═══════════════════════════════════════════════════════════════════════

def run_live_comparison(manoeuvre_name: str, duration: float = 10.0,
                        run_id: int = 1, viz: bool = False):
    """
    Fly the drone and run the sim simultaneously, comparing in real-time.
    """
    traj_fn = cfg.TRAJECTORY_FNS.get(manoeuvre_name, cfg.TRAJECTORY_FNS["hover"])

    # ── Initialise simulation ─────────────────────────────────────────
    params = CrazyflieParams.brushless()
    sim_config = SimConfig(
        dt=0.001, enable_sensors=False, enable_ground_collision=True,
        enable_ground_effect=True, enable_aerodynamics=True, log_data=False,
    )
    sim = CrazyflieSim(params, sim_config)
    sim.reset(position=np.array([0.0, 0.0, -cfg.FLIGHT_HEIGHT]))
    controller = CrazyflieController(params)

    # Pre-spin to hover (closed-loop)
    for _ in range(500):
        pwms = controller.compute(
            sim.state, desired_altitude=cfg.FLIGHT_HEIGHT, dt=0.001)
        sim.step(pwms, command_type='pwm')
    sim.state[VEL] = np.zeros(3)
    sim.state[OMEGA] = np.zeros(3)
    sim.time = 0.0

    # ── Prepare output ────────────────────────────────────────────────
    data_dir = SCRIPT_DIR / cfg.DATA_DIR
    data_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = data_dir / f"live_{manoeuvre_name}_run{run_id}_{stamp}.csv"

    state = DroneState()
    rows = []
    dt = 1.0 / cfg.CONTROL_RATE_HZ
    sim_dt = sim_config.dt

    # Running error accumulators
    err_sq_sum = np.zeros(3)  # x, y, z
    n_samples = 0

    print(f"\n{'='*70}")
    print(f"  LIVE SIM-TO-REAL COMPARISON (closed-loop)")
    print(f"  Manoeuvre: {manoeuvre_name}  |  Duration: {duration:.0f}s")
    print(f"  Platform: CF Brushless + Flow deck v2")
    print(f"{'='*70}")

    # ── 3D Visualisation setup ─────────────────────────────────────
    trail_sim, trail_real = [], []
    if viz and _HAS_VIZ:
        pygame.init()
        viz_w, viz_h = 960, 540
        pygame.display.set_mode((viz_w, viz_h), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Digital Twin — Live Comparison")
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(50, viz_w / viz_h, 0.1, 100.0)
        glEnable(GL_DEPTH_TEST)
    elif viz:
        print("  [WARN] --viz requested but pygame/OpenGL not available")
        viz = False

    cflib.crtp.init_drivers()

    with SyncCrazyflie(cfg.URI, cf=Crazyflie(rw_cache="./cache")) as scf:
        reset_estimator(scf)
        log_configs = setup_logging(scf, state)
        arm_motors(scf)
        commander = scf.cf.commander

        try:
            # ── Take-off ──────────────────────────────────────────
            print("  Taking off ...")
            for _ in range(int(2.0 * cfg.CONTROL_RATE_HZ)):
                commander.send_position_setpoint(0, 0, cfg.FLIGHT_HEIGHT, 0)
                time.sleep(dt)

            # ── Settle ────────────────────────────────────────────
            start_ref = traj_fn(0.0)
            print(f"  Settling at ({start_ref[0]:.2f}, {start_ref[1]:.2f}) ...")
            for _ in range(int(3.0 * cfg.CONTROL_RATE_HZ)):
                commander.send_position_setpoint(
                    start_ref[0], start_ref[1], start_ref[2], start_ref[3])
                time.sleep(dt)

            # ── Live tracking with comparison ─────────────────────
            print_header()
            t0 = time.monotonic()
            sim_t = 0.0
            display_counter = 0

            while True:
                t_now = time.monotonic() - t0
                if t_now >= duration:
                    break

                ref = traj_fn(t_now)
                commander.send_position_setpoint(ref[0], ref[1], ref[2], ref[3])

                snap = state.snapshot()

                # Battery safety
                if snap["vbat"] > 0.5 and snap["vbat"] < cfg.MIN_BATTERY_V:
                    print("\n  LOW BATTERY -- aborting!")
                    break

                # ── Run sim with closed-loop PID ──────────────────
                # Position outer loop: position error → desired tilt
                sim_pos_pre = sim.state[POS]
                sim_vel = sim.state[VEL]
                pos_err_x = ref[0] - sim_pos_pre[0]
                pos_err_y = ref[1] - sim_pos_pre[1]
                # Negative pitch = nose down = move in +X (NED convention)
                desired_pitch = np.clip(-(KP_XY * pos_err_x - KD_XY * sim_vel[0]), -MAX_TILT, MAX_TILT)
                desired_roll  = np.clip( (KP_XY * pos_err_y - KD_XY * sim_vel[1]), -MAX_TILT, MAX_TILT)

                frame_dt = dt
                sim_pwms = controller.compute(
                    sim.state,
                    desired_roll=desired_roll,
                    desired_pitch=desired_pitch,
                    desired_yaw_rate=0.0,
                    desired_altitude=ref[2],
                    dt=frame_dt,
                )

                # Advance sim to match real time
                steps = max(1, int(dt / sim_dt))
                for _ in range(steps):
                    sim.step(sim_pwms, command_type='pwm')
                    sim_t += sim_dt

                # ── Compute error ─────────────────────────────────
                sim_pos = sim.state[POS]
                sim_euler = quat_to_euler(sim.state[QUAT])
                sim_z = -sim_pos[2]

                err = np.array([
                    sim_pos[0] - snap["x"],
                    sim_pos[1] - snap["y"],
                    sim_z - snap["z"],
                ])
                err_sq_sum += err**2
                n_samples += 1

                # ── Display (every 10th sample = 10 Hz) ───────────
                display_counter += 1
                if display_counter % 10 == 0:
                    print_live_row(t_now, snap, sim.state)

                # ── Log row ───────────────────────────────────────
                rows.append({
                    "t":         round(t_now, 4),
                    "real_x":    snap["x"],   "real_y":    snap["y"],
                    "real_z":    snap["z"],
                    "real_vx":   snap["vx"],  "real_vy":   snap["vy"],
                    "real_vz":   snap["vz"],
                    "real_roll": snap["roll"], "real_pitch": snap["pitch"],
                    "real_yaw":  snap["yaw"],
                    "sim_x":     sim_pos[0],  "sim_y":     sim_pos[1],
                    "sim_z":     sim_z,
                    "sim_vx":    sim.state[3], "sim_vy":    sim.state[4],
                    "sim_vz":    sim.state[5],
                    "sim_roll":  math.degrees(sim_euler[0]),
                    "sim_pitch": math.degrees(sim_euler[1]),
                    "sim_yaw":   math.degrees(sim_euler[2]),
                    "m1": snap["m1"], "m2": snap["m2"],
                    "m3": snap["m3"], "m4": snap["m4"],
                    "vbat": snap["vbat"],
                    "x_ref": ref[0], "y_ref": ref[1],
                    "z_ref": ref[2], "yaw_ref": ref[3],
                    "err_x": err[0], "err_y": err[1], "err_z": err[2],
                })

                # ── 3D visualisation (10 Hz, not every tick) ──────
                if viz and display_counter % 10 == 0:
                    for ev in pygame.event.get():
                        if ev.type == QUIT:
                            break
                    gl_sim = (sim_pos[0], sim_pos[1], -sim_pos[2])
                    gl_real = (snap["x"], snap["y"], snap["z"])
                    trail_sim.append(gl_sim)
                    trail_real.append(gl_real)
                    if len(trail_sim) > 500: trail_sim.pop(0)
                    if len(trail_real) > 500: trail_real.pop(0)

                    glClearColor(*COL_SKY)
                    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                    glMatrixMode(GL_MODELVIEW)
                    glLoadIdentity()
                    mid_x = (gl_sim[0] + gl_real[0]) / 2
                    mid_y = (gl_sim[1] + gl_real[1]) / 2
                    gluLookAt(mid_x + 1.5, mid_y + 1.5, 1.8,
                              mid_x, mid_y, 0.5, 0, 0, 1)
                    glEnable(GL_BLEND)
                    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                    glEnable(GL_LINE_SMOOTH)
                    draw_ground_grid()

                    # Sim drone (full model)
                    draw_altitude_line(gl_sim[0], gl_sim[1], gl_sim[2])
                    draw_trail(trail_sim)
                    glPushMatrix()
                    glTranslatef(*gl_sim)
                    R = quat_to_rotation_matrix(sim.state[QUAT])
                    R_gl = R.copy()
                    R_gl[2, :] = -R_gl[2, :]
                    R_gl[:, 2] = -R_gl[:, 2]
                    m4 = np.eye(4); m4[:3, :3] = R_gl
                    glMultMatrixf(m4.T.flatten())
                    draw_crazyflie(params.arm_length,
                                   sim.motors.get_speeds(),
                                   params.motor_max_speed * 2 * math.pi / 60)
                    glPopMatrix()

                    # Real drone (green cross marker)
                    glColor4f(0.0, 0.7, 0.0, 1.0)
                    glLineWidth(4.0)
                    s = 0.04
                    glBegin(GL_LINES)
                    for axis in [(s,0,0),(0,s,0),(0,0,s)]:
                        glVertex3f(gl_real[0]-axis[0], gl_real[1]-axis[1], gl_real[2]-axis[2])
                        glVertex3f(gl_real[0]+axis[0], gl_real[1]+axis[1], gl_real[2]+axis[2])
                    glEnd()
                    glLineWidth(1.0)

                    # Real drone trail (green)
                    if len(trail_real) > 1:
                        glLineWidth(2.0)
                        glBegin(GL_LINE_STRIP)
                        for i, pt in enumerate(trail_real):
                            a = i / len(trail_real) * 0.6
                            glColor4f(0.0, 0.7, 0.0, a)
                            glVertex3f(*pt)
                        glEnd()
                        glLineWidth(1.0)

                    pygame.display.flip()

                # Maintain loop rate
                elapsed = time.monotonic() - t0 - t_now
                sleep_t = dt - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

            # ── Land ──────────────────────────────────────────────
            print("\n  Landing ...")
            snap = state.snapshot()
            for h in np.linspace(snap["z"], 0.05, 30):
                commander.send_position_setpoint(snap["x"], snap["y"], h, 0)
                time.sleep(0.1)
            commander.send_stop_setpoint()
            time.sleep(0.5)
            scf.cf.platform.send_arming_request(False)

        except KeyboardInterrupt:
            print("\n  Interrupted -- sending stop!")
            commander.send_stop_setpoint()
            scf.cf.platform.send_arming_request(False)

        finally:
            for lc in log_configs:
                lc.stop()

    # ── Cleanup viz ───────────────────────────────────────────────────
    if viz:
        pygame.quit()

    # ── Print summary ─────────────────────────────────────────────────
    if n_samples > 0:
        rmse = np.sqrt(err_sq_sum / n_samples)
        print(f"\n{'='*70}")
        print(f"  LIVE COMPARISON RESULTS ({n_samples} samples)")
        print(f"{'='*70}")
        print(f"  RMSE  X: {rmse[0]:.4f} m")
        print(f"  RMSE  Y: {rmse[1]:.4f} m")
        print(f"  RMSE  Z: {rmse[2]:.4f} m")
        print(f"  RMSE 3D: {np.linalg.norm(rmse):.4f} m")
        print(f"{'='*70}")

    # ── Save CSV ──────────────────────────────────────────────────────
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Saved {len(rows)} samples -> {csv_path}")

        # Offer to run offline analysis with plots
        print(f"\n  To generate plots, run:")
        print(f"    python validate_twin.py {csv_path} --plot")

    return csv_path


# ═══════════════════════════════════════════════════════════════════════
# Sim-Only Mode (for screen recording)
# ═══════════════════════════════════════════════════════════════════════

def run_sim_only(manoeuvre_name: str, duration: float = 20.0):
    """Run the simulation with 3D visualisation only (no physical drone)."""
    if not _HAS_VIZ:
        print("ERROR: pygame/OpenGL not available for --sim-only mode")
        return

    traj_fn = cfg.TRAJECTORY_FNS.get(manoeuvre_name, cfg.TRAJECTORY_FNS["hover"])
    params = CrazyflieParams.brushless()
    sim_config = SimConfig(
        dt=0.001, enable_sensors=False, enable_ground_collision=True,
        enable_ground_effect=True, enable_aerodynamics=True, log_data=False,
    )
    sim = CrazyflieSim(params, sim_config)
    sim.reset(position=np.array([0.0, 0.0, -cfg.FLIGHT_HEIGHT]))
    controller = CrazyflieController(params)

    # Pre-spin to stable hover
    for _ in range(1000):
        pwms = controller.compute(
            sim.state, desired_altitude=cfg.FLIGHT_HEIGHT, dt=0.001)
        sim.step(pwms, command_type='pwm')
    sim.state[VEL] = np.zeros(3)
    sim.state[OMEGA] = np.zeros(3)
    sim.time = 0.0

    # Pygame / OpenGL setup
    pygame.init()
    viz_w, viz_h = 1280, 720
    pygame.display.set_mode((viz_w, viz_h), DOUBLEBUF | OPENGL)
    pygame.display.set_caption(f"Digital Twin — {manoeuvre_name} (sim-only)")
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(50, viz_w / viz_h, 0.1, 100.0)
    glEnable(GL_DEPTH_TEST)

    clock = pygame.time.Clock()
    trail = []
    sim_dt = sim_config.dt
    ctrl_dt = 1.0 / cfg.CONTROL_RATE_HZ

    # Camera orbit
    cam_angle = 45.0
    cam_elev = 30.0
    cam_dist = 2.5

    print(f"\n  SIM-ONLY: {manoeuvre_name} | {duration:.0f}s | Press Esc to quit")
    print(f"  Starting in 10 seconds... (start your screen recorder)\n")

    # ── 10-second countdown (scene visible, drone hovering) ───────
    pygame.font.init()
    countdown_font = pygame.font.SysFont("Consolas", 72, bold=True)
    label_font = pygame.font.SysFont("Consolas", 24, bold=True)
    countdown_surface = pygame.Surface((viz_w, viz_h), pygame.SRCALPHA)

    countdown_start = time.monotonic()
    while time.monotonic() - countdown_start < 10.0:
        remaining = 10.0 - (time.monotonic() - countdown_start)
        clock.tick(60)
        for ev in pygame.event.get():
            if ev.type == QUIT:
                pygame.quit()
                return
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                pygame.quit()
                return

        # Render static scene with drone hovering
        gl_pos = (sim.state[POS][0], sim.state[POS][1], -sim.state[POS][2])
        glClearColor(*COL_SKY)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        elev_rad = math.radians(cam_elev)
        angle_rad = math.radians(cam_angle)
        cx = gl_pos[0] + cam_dist * math.cos(elev_rad) * math.cos(angle_rad)
        cy = gl_pos[1] + cam_dist * math.cos(elev_rad) * math.sin(angle_rad)
        cz = gl_pos[2] + cam_dist * math.sin(elev_rad)
        gluLookAt(cx, cy, cz, gl_pos[0], gl_pos[1], gl_pos[2], 0, 0, 1)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        draw_ground_grid()
        draw_altitude_line(gl_pos[0], gl_pos[1], gl_pos[2])
        glPushMatrix()
        glTranslatef(*gl_pos)
        R = quat_to_rotation_matrix(sim.state[QUAT])
        R_gl = R.copy(); R_gl[2,:] = -R_gl[2,:]; R_gl[:,2] = -R_gl[:,2]
        m4 = np.eye(4); m4[:3,:3] = R_gl
        glMultMatrixf(m4.T.flatten())
        draw_crazyflie(params.arm_length, sim.motors.get_speeds(),
                       params.motor_max_speed * 2 * math.pi / 60)
        glPopMatrix()

        # Countdown overlay
        countdown_surface.fill((0, 0, 0, 0))
        txt = countdown_font.render(f"{int(remaining) + 1}", True, (30, 30, 30))
        lbl = label_font.render(f"Starting {manoeuvre_name} in...", True, (80, 80, 80))
        countdown_surface.blit(lbl, ((viz_w - lbl.get_width()) // 2, viz_h // 2 - 60))
        countdown_surface.blit(txt, ((viz_w - txt.get_width()) // 2, viz_h // 2 - 10))

        # Overlay HUD via raster
        hud_data = pygame.image.tostring(countdown_surface, "RGBA", True)
        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
        glOrtho(0, viz_w, 0, viz_h, -1, 1)
        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
        glDisable(GL_DEPTH_TEST)
        glRasterPos2i(0, 0)
        glDrawPixels(viz_w, viz_h, GL_RGBA, GL_UNSIGNED_BYTE, hud_data)
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION); glPopMatrix()
        glMatrixMode(GL_MODELVIEW); glPopMatrix()

        pygame.display.flip()

    print("  GO!\n")

    t = 0.0
    running = True
    while running and t < duration:
        frame_dt = clock.tick(60) / 1000.0

        # Events
        for ev in pygame.event.get():
            if ev.type == QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
            elif ev.type == pygame.MOUSEWHEEL:
                cam_dist = max(0.5, min(10.0, cam_dist - ev.y * 0.3))
        # Mouse drag for camera orbit
        btns = pygame.mouse.get_pressed()
        if btns[2]:
            rel = pygame.mouse.get_rel()
            cam_angle -= rel[0] * 0.3
            cam_elev = max(-10, min(89, cam_elev + rel[1] * 0.3))
        else:
            pygame.mouse.get_rel()

        # Trajectory setpoint
        ref = traj_fn(t)

        # Position → attitude outer loop
        sim_pos = sim.state[POS]
        sim_vel = sim.state[VEL]
        pos_err_x = ref[0] - sim_pos[0]
        pos_err_y = ref[1] - sim_pos[1]
        desired_pitch = np.clip(-(KP_XY * pos_err_x - KD_XY * sim_vel[0]),
                                -MAX_TILT, MAX_TILT)
        desired_roll = np.clip((KP_XY * pos_err_y - KD_XY * sim_vel[1]),
                               -MAX_TILT, MAX_TILT)

        # PID controller
        sim_pwms = controller.compute(
            sim.state,
            desired_roll=desired_roll,
            desired_pitch=desired_pitch,
            desired_yaw_rate=0.0,
            desired_altitude=ref[2],
            dt=ctrl_dt,
        )

        # Step sim (real-time paced)
        steps = max(1, int(frame_dt / sim_dt))
        for _ in range(steps):
            sim.step(sim_pwms, command_type='pwm')
        t += frame_dt

        # Trail
        gl_pos = (sim_pos[0], sim_pos[1], -sim_pos[2])
        trail.append(gl_pos)
        if len(trail) > 800:
            trail.pop(0)

        # ── Render ────────────────────────────────────────────────
        glClearColor(*COL_SKY)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Camera orbiting the drone
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        elev_rad = math.radians(cam_elev)
        angle_rad = math.radians(cam_angle)
        cx = gl_pos[0] + cam_dist * math.cos(elev_rad) * math.cos(angle_rad)
        cy = gl_pos[1] + cam_dist * math.cos(elev_rad) * math.sin(angle_rad)
        cz = gl_pos[2] + cam_dist * math.sin(elev_rad)
        gluLookAt(cx, cy, cz, gl_pos[0], gl_pos[1], gl_pos[2], 0, 0, 1)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)

        draw_ground_grid()
        draw_altitude_line(gl_pos[0], gl_pos[1], gl_pos[2])
        draw_trail(trail)

        # Drone model
        glPushMatrix()
        glTranslatef(*gl_pos)
        R = quat_to_rotation_matrix(sim.state[QUAT])
        R_gl = R.copy()
        R_gl[2, :] = -R_gl[2, :]
        R_gl[:, 2] = -R_gl[:, 2]
        m4 = np.eye(4)
        m4[:3, :3] = R_gl
        glMultMatrixf(m4.T.flatten())
        draw_crazyflie(params.arm_length,
                       sim.motors.get_speeds(),
                       params.motor_max_speed * 2 * math.pi / 60)
        glPopMatrix()

        # Trajectory reference ring (thin grey circle on the ground)
        if manoeuvre_name == "circle":
            glColor4f(0.4, 0.4, 0.4, 0.5)
            glLineWidth(1.5)
            glBegin(GL_LINE_LOOP)
            for i in range(64):
                theta = 2 * math.pi * i / 64
                glVertex3f(0.4 * math.cos(theta),
                           0.4 * math.sin(theta), ref[2])
            glEnd()
            glLineWidth(1.0)

        pygame.display.flip()

    pygame.quit()
    print(f"  Done. ({t:.1f}s simulated)")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Live sim-to-real comparison for Crazyflie Brushless."
    )
    parser.add_argument(
        "--manoeuvre", "-m", default="hover",
        choices=list(cfg.MANOEUVRES.keys()),
        help="Manoeuvre to fly (default: hover)",
    )
    parser.add_argument(
        "--duration", "-d", type=float, default=None,
        help="Override manoeuvre duration (seconds)",
    )
    parser.add_argument(
        "--run", type=int, default=1,
        help="Run number label",
    )
    parser.add_argument(
        "--uri", type=str, default=None,
        help=f"Crazyflie URI (default: {cfg.URI})",
    )
    parser.add_argument(
        "--viz", action="store_true",
        help="Open a 3D OpenGL window showing both drones in real-time",
    )
    parser.add_argument(
        "--sim-only", action="store_true",
        help="Run simulation only with 3D window (no drone, for screen recording)",
    )
    args = parser.parse_args()

    if args.uri:
        cfg.URI = args.uri

    dur = args.duration or cfg.MANOEUVRES[args.manoeuvre]["duration"]

    if args.sim_only:
        run_sim_only(args.manoeuvre, duration=dur)
    else:
        run_live_comparison(args.manoeuvre, duration=dur, run_id=args.run,
                            viz=args.viz)


if __name__ == "__main__":
    main()
