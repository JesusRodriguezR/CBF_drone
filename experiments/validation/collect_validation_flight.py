"""
Validation Flight Data Collector
==================================

Fly the Crazyflie Brushless through predefined manoeuvres and log
extended telemetry for sim-to-real digital twin validation.

Uses the Flow deck v2 for state estimation (no OptiTrack required).
Logs: position, velocity, attitude, angular rates, motor PWMs,
      battery voltage, accelerometer, and control setpoints.

Usage
-----
    python collect_validation_flight.py                     # all manoeuvres
    python collect_validation_flight.py --manoeuvre hover    # single manoeuvre
    python collect_validation_flight.py --list               # show available
    python collect_validation_flight.py --dry-run            # print plan only

Platform: Crazyflie 2.1 Brushless + Flow deck v2
"""

import argparse
import csv
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

import validation_config as cfg


# ═══════════════════════════════════════════════════════════════════════
# Drone State (thread-safe accumulator for async log callbacks)
# ═══════════════════════════════════════════════════════════════════════

class DroneState:
    """Accumulates async log callbacks into a snapshot-able dictionary."""

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
# Logging Setup
# ═══════════════════════════════════════════════════════════════════════

_LOG_CALLBACKS = {
    "pose":     "update_pose",
    "attitude": "update_attitude",
    "motors":   "update_motors",
    "accel":    "update_accel",
}


def setup_logging(scf: SyncCrazyflie, state: DroneState):
    """Configure all onboard log groups and start streaming."""
    configs = []

    for group_name, group_info in cfg.LOG_GROUPS.items():
        lc = LogConfig(name=group_name,
                       period_in_ms=group_info["period_ms"])
        for var_name, var_type in group_info["variables"]:
            lc.add_variable(var_name, var_type)
        scf.cf.log.add_config(lc)

        cb_name = _LOG_CALLBACKS[group_name]
        lc.data_received_cb.add_callback(getattr(state, cb_name))
        lc.start()
        configs.append(lc)

    return configs


# ═══════════════════════════════════════════════════════════════════════
# EKF & Arming Helpers
# ═══════════════════════════════════════════════════════════════════════

def reset_estimator(scf: SyncCrazyflie):
    """Reset the Kalman filter and wait for Flow deck to stabilise."""
    cf = scf.cf
    cf.param.set_value("kalman.resetEstimation", "1")
    time.sleep(0.1)
    cf.param.set_value("kalman.resetEstimation", "0")

    print("  Waiting for estimator to stabilise …", end="", flush=True)
    time.sleep(3.0)

    # Sanity check: verify Flow deck is alive
    log_config = LogConfig(name="z_check", period_in_ms=100)
    log_config.add_variable("stateEstimate.z", "float")

    z_samples = []
    with SyncLogger(scf, log_config) as logger:
        for log_entry in logger:
            z_samples.append(log_entry[1]["stateEstimate.z"])
            if len(z_samples) >= 5:
                break

    z_mean = sum(z_samples) / len(z_samples)
    print(f" z={z_mean:.3f} m")

    if z_mean < 0.005:
        raise RuntimeError(
            "Flow deck sanity check failed (z ≈ 0 m).\n"
            "  → Is the Flow deck v2 attached and detected?\n"
            "  → Is the drone sitting on a surface?"
        )
    print(f"  ✓ Flow deck OK (ground height ≈ {z_mean*100:.1f} cm)")


def arm_motors(scf: SyncCrazyflie):
    """Arm brushless motors on the Crazyflie Brushless."""
    cf = scf.cf
    print("  Clearing crash flags …", end="", flush=True)
    cf.platform.send_crash_recovery_request()
    time.sleep(0.5)
    print(" OK")

    print("  Arming brushless motors …", end="", flush=True)
    cf.platform.send_arming_request(True)
    time.sleep(1.0)
    cf.commander.send_setpoint(0, 0, 0, 0)
    time.sleep(0.1)
    print(" ARMED")


# ═══════════════════════════════════════════════════════════════════════
# Flight Execution
# ═══════════════════════════════════════════════════════════════════════

def run_manoeuvre(manoeuvre_name: str, run_id: int = 1):
    """
    Execute a single validation manoeuvre on the physical drone.

    Stages: connect → reset EKF → arm → take-off → settle → track → land → save
    """
    if manoeuvre_name not in cfg.TRAJECTORY_FNS:
        raise ValueError(
            f"Unknown manoeuvre '{manoeuvre_name}'. "
            f"Available: {list(cfg.TRAJECTORY_FNS.keys())}"
        )

    traj_fn = cfg.TRAJECTORY_FNS[manoeuvre_name]
    duration = cfg.MANOEUVRES[manoeuvre_name]["duration"]
    description = cfg.MANOEUVRES[manoeuvre_name]["description"]

    # Prepare output path
    data_dir = Path(__file__).parent / cfg.DATA_DIR
    data_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = data_dir / f"val_{manoeuvre_name}_run{run_id}_{stamp}.csv"

    state = DroneState()
    rows = []
    dt = 1.0 / cfg.CONTROL_RATE_HZ

    print(f"\n{'='*60}")
    print(f"  VALIDATION FLIGHT: {manoeuvre_name}")
    print(f"  {description}")
    print(f"  Duration: {duration:.1f}s  |  Run: {run_id}")
    print(f"{'='*60}")

    # Initialise radio drivers
    cflib.crtp.init_drivers()

    with SyncCrazyflie(cfg.URI, cf=Crazyflie(rw_cache="./cache")) as scf:
        reset_estimator(scf)
        log_configs = setup_logging(scf, state)
        arm_motors(scf)
        commander = scf.cf.commander

        try:
            # ── Take-off ──────────────────────────────────────────
            print("  Taking off …")
            for _ in range(int(2.0 * cfg.CONTROL_RATE_HZ)):
                commander.send_position_setpoint(0, 0, cfg.FLIGHT_HEIGHT, 0)
                time.sleep(dt)

            # ── Settle at start ───────────────────────────────────
            start_ref = traj_fn(0.0)
            print(f"  Settling at ({start_ref[0]:.2f}, {start_ref[1]:.2f}, "
                  f"{start_ref[2]:.2f}) …")
            for _ in range(int(3.0 * cfg.CONTROL_RATE_HZ)):
                commander.send_position_setpoint(
                    start_ref[0], start_ref[1], start_ref[2], start_ref[3]
                )
                time.sleep(dt)

            # ── Trajectory tracking ───────────────────────────────
            print("  Tracking manoeuvre …")
            t0 = time.monotonic()

            while True:
                t_now = time.monotonic() - t0
                if t_now >= duration:
                    break

                ref = traj_fn(t_now)
                commander.send_position_setpoint(
                    ref[0], ref[1], ref[2], ref[3]
                )

                snap = state.snapshot()

                # Safety: battery check
                if snap["vbat"] > 0.5 and snap["vbat"] < cfg.MIN_BATTERY_V:
                    print("  ⚠ LOW BATTERY — aborting!")
                    break

                # Log row
                row = {
                    "t":       round(t_now, 4),
                    "x":       snap["x"],
                    "y":       snap["y"],
                    "z":       snap["z"],
                    "vx":      snap["vx"],
                    "vy":      snap["vy"],
                    "vz":      snap["vz"],
                    "roll":    snap["roll"],
                    "pitch":   snap["pitch"],
                    "yaw":     snap["yaw"],
                    "gyro_x":  snap["gyro_x"],
                    "gyro_y":  snap["gyro_y"],
                    "gyro_z":  snap["gyro_z"],
                    "m1":      snap["m1"],
                    "m2":      snap["m2"],
                    "m3":      snap["m3"],
                    "m4":      snap["m4"],
                    "vbat":    snap["vbat"],
                    "acc_x":   snap["acc_x"],
                    "acc_y":   snap["acc_y"],
                    "acc_z":   snap["acc_z"],
                    "x_ref":   ref[0],
                    "y_ref":   ref[1],
                    "z_ref":   ref[2],
                    "yaw_ref": ref[3],
                    "manoeuvre": manoeuvre_name,
                }
                rows.append(row)

                # Maintain loop rate
                elapsed = time.monotonic() - t0 - t_now
                sleep_t = dt - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

            # ── Land ──────────────────────────────────────────────
            print("  Landing …")
            snap = state.snapshot()
            for h in np.linspace(snap["z"], 0.05, 30):
                commander.send_position_setpoint(snap["x"], snap["y"], h, 0)
                time.sleep(0.1)
            commander.send_stop_setpoint()
            time.sleep(0.5)
            scf.cf.platform.send_arming_request(False)

        except KeyboardInterrupt:
            print("\n  ⚠ Interrupted — sending stop!")
            commander.send_stop_setpoint()
            scf.cf.platform.send_arming_request(False)

        finally:
            for lc in log_configs:
                lc.stop()

    # ── Save CSV ──────────────────────────────────────────────────────
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  ✓ Saved {len(rows)} samples → {csv_path}")
    else:
        print("\n  ✗ No data recorded.")

    return csv_path


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Collect validation flight data from Crazyflie Brushless."
    )
    parser.add_argument(
        "--manoeuvre", "-m", type=str, default=None,
        choices=list(cfg.MANOEUVRES.keys()),
        help="Run a single manoeuvre (default: run all sequentially)",
    )
    parser.add_argument(
        "--run", type=int, default=1,
        help="Run number label (default: 1)",
    )
    parser.add_argument(
        "--uri", type=str, default=None,
        help=f"Crazyflie URI (default: {cfg.URI})",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available manoeuvres and exit",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the flight plan without executing",
    )
    args = parser.parse_args()

    if args.uri:
        cfg.URI = args.uri

    if args.list:
        print("\nAvailable validation manoeuvres:")
        for name, info in cfg.MANOEUVRES.items():
            print(f"  {name:12s}  {info['duration']:5.1f}s  {info['description']}")
        return

    # Build list of manoeuvres to run
    if args.manoeuvre:
        plan = [args.manoeuvre]
    else:
        plan = list(cfg.MANOEUVRES.keys())

    total_time = sum(cfg.MANOEUVRES[m]["duration"] for m in plan)
    print(f"\n{'='*60}")
    print(f"  VALIDATION DATA COLLECTION")
    print(f"  {len(plan)} manoeuvre(s), ~{total_time:.0f}s total flight time")
    print(f"  Platform: Crazyflie Brushless + Flow deck v2")
    print(f"{'='*60}")

    for i, m in enumerate(plan):
        info = cfg.MANOEUVRES[m]
        print(f"\n  [{i+1}/{len(plan)}]  {m:12s}  {info['duration']:.0f}s  "
              f"{info['description']}")

    if args.dry_run:
        print(f"\n  [DRY RUN] — no flights executed.")
        return

    # Initialise radio once
    cflib.crtp.init_drivers()

    for i, m in enumerate(plan):
        print(f"\n{'─'*60}")
        print(f"  Manoeuvre {i+1}/{len(plan)}: {m}")
        print(f"{'─'*60}")

        run_manoeuvre(m, run_id=args.run)

        if i < len(plan) - 1:
            print("\n  Pausing 10s before next manoeuvre …")
            time.sleep(10)

    print(f"\n{'='*60}")
    print(f"  ✓ All manoeuvres complete!")
    print(f"  Data saved in experiments/validation/{cfg.DATA_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
