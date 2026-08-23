"""
Crazyflie Sensor Logger — Autonomous Mission
==============================================

Logs all onboard sensor data during a scripted mission for digital twin
validation. Saves timestamped CSV files to exp_results/.

PRE-FLIGHT CHECKLIST:
    [ ] Battery fully charged (≥3.7 V)
    [ ] CF firmware version confirmed and recorded
    [ ] Crazyradio PA connected
    [ ] Flight area clear, ground is level
    [ ] Radio channel / address set correctly below
    [ ] OptiTrack streaming if needed (Phase 2+)

Usage:
    python exp_sensor_logger_v1.py --mission static --duration 60  (Phase 1: IMU noise)
    python exp_sensor_logger_v1.py --mission hover --duration 10
    python exp_sensor_logger_v1.py --mission square --duration 30
    python exp_sensor_logger_v1.py --mission custom --waypoints waypoints.json
    python exp_sensor_logger_v1.py --mission hover --duration 5 --dry-run
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

# ── cflib imports ────────────────────────────────────────────────────────
try:
    import cflib.crtp
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.log import LogConfig
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
    from cflib.crazyflie.syncLogger import SyncLogger
    from cflib.positioning.motion_commander import MotionCommander
    CFLIB_AVAILABLE = True
except ImportError:
    CFLIB_AVAILABLE = False
    print("[WARN] cflib not installed. Install with: pip install cflib")
    print("       Running in dry-run mode only.\n")

# ── Configuration ────────────────────────────────────────────────────────
DEFAULT_URI = "radio://0/80/2M/E7E7E7E7E7"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_results")
LOG_RATE_HZ = 100  # Logging rate in Hz


# ── Column definitions ──────────────────────────────────────────────────
# Maps human-readable CSV column names to CF log variable paths.
# Grouped by log config (max 6 variables per LogConfig block).
LOG_GROUPS = {
    "imu": {
        "accel_x":  "acc.x",         # m/s²
        "accel_y":  "acc.y",         # m/s²
        "accel_z":  "acc.z",         # m/s²
        "gyro_x":   "gyro.x",       # °/s (raw from CF)
        "gyro_y":   "gyro.y",       # °/s
        "gyro_z":   "gyro.z",       # °/s
    },
    "baro_stab": {
        "baro_pressure":  "baro.pressure",    # mbar
        "baro_asl":       "baro.asl",         # m above sea level
        "stab_roll":      "stabilizer.roll",  # °
        "stab_pitch":     "stabilizer.pitch", # °
        "stab_yaw":       "stabilizer.yaw",   # °
    },
    "state_pos": {
        "stateEst_x":   "stateEstimate.x",   # m
        "stateEst_y":   "stateEstimate.y",   # m
        "stateEst_z":   "stateEstimate.z",   # m
        "stateEst_vx":  "stateEstimate.vx",  # m/s
        "stateEst_vy":  "stateEstimate.vy",  # m/s
        "stateEst_vz":  "stateEstimate.vz",  # m/s
    },
    "motors": {
        "motor_m1":  "motor.m1",   # PWM 0–65535
        "motor_m2":  "motor.m2",
        "motor_m3":  "motor.m3",
        "motor_m4":  "motor.m4",
    },
}

# Flat column order for CSV
CSV_COLUMNS = ["timestamp_ms"]
for group_cols in LOG_GROUPS.values():
    CSV_COLUMNS.extend(group_cols.keys())


def build_log_configs(log_groups: dict, rate_hz: int) -> list:
    """Create cflib LogConfig objects from our log group definitions."""
    configs = []
    for group_name, variables in log_groups.items():
        lc = LogConfig(name=group_name, period_in_ms=int(1000 / rate_hz))
        for csv_col, cf_var in variables.items():
            lc.add_variable(cf_var, "float")
        configs.append((group_name, lc))
    return configs


class SensorLogger:
    """Collects CF log data into rows for CSV export."""

    def __init__(self):
        self.rows = []
        self._current_row = {}
        self._start_time_ms = None
        self._groups_received = set()

    def on_log_data(self, timestamp, data, logconf):
        """Callback for each log config data packet."""
        if self._start_time_ms is None:
            self._start_time_ms = timestamp

        relative_ms = timestamp - self._start_time_ms
        group_name = logconf.name

        # Store timestamp
        self._current_row["timestamp_ms"] = relative_ms

        # Map CF variable names back to CSV column names
        group_vars = LOG_GROUPS[group_name]
        for csv_col, cf_var in group_vars.items():
            if cf_var in data:
                self._current_row[csv_col] = data[cf_var]

        self._groups_received.add(group_name)

        # When all groups have reported, flush the row
        if len(self._groups_received) == len(LOG_GROUPS):
            self.rows.append(dict(self._current_row))
            self._current_row = {}
            self._groups_received.clear()

    def save_csv(self, filepath: str):
        """Write collected rows to CSV."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({col: row.get(col, "") for col in CSV_COLUMNS})
        print(f"  ✓ Saved {len(self.rows)} samples to: {filepath}")


# ── Mission profiles ────────────────────────────────────────────────────

def mission_hover(mc, duration: float, altitude: float = 0.5):
    """Take off, hover at altitude for `duration` seconds, land."""
    print(f"  ▸ Taking off to {altitude}m...")
    mc.take_off(altitude, velocity=0.3)
    time.sleep(2.0)  # stabilise

    print(f"  ▸ Hovering for {duration}s...")
    time.sleep(duration)

    print("  ▸ Landing...")
    mc.land(velocity=0.2)
    time.sleep(2.0)


def mission_square(mc, duration: float, side: float = 1.0, altitude: float = 0.5):
    """Fly a square pattern at a given altitude."""
    velocity = 0.3  # m/s
    print(f"  ▸ Taking off to {altitude}m...")
    mc.take_off(altitude, velocity=0.3)
    time.sleep(2.0)

    laps = max(1, int(duration / (4 * side / velocity + 4 * 1.0)))
    print(f"  ▸ Flying {laps} square laps ({side}m sides)...")

    for lap in range(laps):
        print(f"    Lap {lap + 1}/{laps}")
        mc.forward(side, velocity=velocity)
        time.sleep(1.0)
        mc.right(side, velocity=velocity)
        time.sleep(1.0)
        mc.back(side, velocity=velocity)
        time.sleep(1.0)
        mc.left(side, velocity=velocity)
        time.sleep(1.0)

    print("  ▸ Landing...")
    mc.land(velocity=0.2)
    time.sleep(2.0)


def mission_custom(mc, waypoints_file: str, altitude: float = 0.5):
    """Fly through waypoints loaded from a JSON file.

    JSON format:
    [
        {"x": 0.5, "y": 0.0, "z": 0.5, "hold_s": 2.0},
        {"x": 0.5, "y": 0.5, "z": 0.5, "hold_s": 2.0},
        ...
    ]
    """
    with open(waypoints_file, "r") as f:
        waypoints = json.load(f)

    print(f"  ▸ Loaded {len(waypoints)} waypoints from {waypoints_file}")
    print(f"  ▸ Taking off to {altitude}m...")
    mc.take_off(altitude, velocity=0.3)
    time.sleep(2.0)

    for i, wp in enumerate(waypoints):
        dx = wp.get("x", 0.0)
        dy = wp.get("y", 0.0)
        dz = wp.get("z", altitude) - altitude
        hold = wp.get("hold_s", 2.0)
        print(f"    WP {i + 1}/{len(waypoints)}: "
              f"go_to({dx:.2f}, {dy:.2f}, {dz:.2f}), hold {hold:.1f}s")
        mc.move_distance(dx, dy, dz, velocity=0.3)
        time.sleep(hold)

    print("  ▸ Landing...")
    mc.land(velocity=0.2)
    time.sleep(2.0)


# ── Dry-run mode ─────────────────────────────────────────────────────────

def dry_run(mission: str, duration: float):
    """Simulate the logging pipeline without hardware."""
    import random

    print("╔══════════════════════════════════════════════════════╗")
    print("║           DRY RUN — No hardware required            ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    logger = SensorLogger()
    n_samples = int(duration * LOG_RATE_HZ)
    print(f"  Generating {n_samples} synthetic samples at {LOG_RATE_HZ} Hz...")

    for i in range(n_samples):
        row = {"timestamp_ms": i * (1000 // LOG_RATE_HZ)}
        for col in CSV_COLUMNS[1:]:
            if "motor" in col:
                row[col] = random.randint(10000, 40000)
            elif "accel" in col:
                row[col] = round(random.gauss(0, 0.5), 4)
            elif "gyro" in col:
                row[col] = round(random.gauss(0, 2.0), 4)
            elif "baro_pressure" in col:
                row[col] = round(1013.25 + random.gauss(0, 0.1), 2)
            elif "baro_asl" in col:
                row[col] = round(200.0 + random.gauss(0, 0.05), 3)
            elif "stab" in col:
                row[col] = round(random.gauss(0, 1.0), 3)
            elif "stateEst" in col:
                row[col] = round(random.gauss(0, 0.01), 4)
            else:
                row[col] = 0.0
        logger.rows.append(row)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = f"sensor_log_{mission}_{timestamp}_DRYRUN.csv"
    filepath = os.path.join(RESULTS_DIR, filename)
    logger.save_csv(filepath)

    # Print sample
    print(f"\n  CSV columns ({len(CSV_COLUMNS)}):")
    for col in CSV_COLUMNS:
        print(f"    • {col}")
    print(f"\n  First row: { {k: logger.rows[0][k] for k in list(CSV_COLUMNS)[:6]} }")
    print("\n  ✓ Dry run complete.")


# ── Main ─────────────────────────────────────────────────────────────────

def run_live(uri: str, mission: str, duration: float,
             waypoints_file: str = None, altitude: float = 0.5):
    """Run a live mission with sensor logging."""

    print("╔══════════════════════════════════════════════════════╗")
    print("║         CRAZYFLIE SENSOR LOGGER — LIVE              ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    cflib.crtp.init_drivers()
    logger = SensorLogger()

    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache="./cache")) as scf:
        cf = scf.cf

        # Print firmware info
        print(f"  Connected to: {uri}")

        # Set up logging
        log_configs = build_log_configs(LOG_GROUPS, LOG_RATE_HZ)
        for group_name, lc in log_configs:
            cf.log.add_config(lc)
            lc.data_received_cb.add_callback(logger.on_log_data)

        # Start logging
        for _, lc in log_configs:
            lc.start()
        print(f"  Logging started at {LOG_RATE_HZ} Hz")

        # Execute mission
        print(f"  Mission: {mission} | Duration: {duration}s")
        print()

        if mission == "static":
            # ── Static capture: no motors, just log sensors on the ground ──
            print("  ▸ STATIC MODE — motors OFF, CF should be on a flat surface.")
            print("    DO NOT MOVE the Crazyflie during capture.")
            print(f"    Logging for {duration}s...")
            print()
            for elapsed in range(int(duration)):
                time.sleep(1.0)
                remaining = int(duration) - elapsed - 1
                sys.stdout.write(f"\r    ⏱  {elapsed + 1:.0f}s / {duration:.0f}s  "
                                 f"({remaining}s remaining)  "
                                 f"[{len(logger.rows)} samples]   ")
                sys.stdout.flush()
            # Capture any fractional second
            frac = duration - int(duration)
            if frac > 0:
                time.sleep(frac)
            print()
        else:
            with MotionCommander(scf, default_height=altitude) as mc:
                if mission == "hover":
                    mission_hover(mc, duration, altitude)
                elif mission == "square":
                    mission_square(mc, duration, altitude=altitude)
                elif mission == "custom":
                    if not waypoints_file:
                        raise ValueError("--waypoints required for custom mission")
                    mission_custom(mc, waypoints_file, altitude)
                else:
                    raise ValueError(f"Unknown mission: {mission}")

        # Stop logging
        for _, lc in log_configs:
            lc.stop()

    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = f"sensor_log_{mission}_{timestamp}.csv"
    filepath = os.path.join(RESULTS_DIR, filename)
    logger.save_csv(filepath)

    # Summary
    print(f"\n  Total samples: {len(logger.rows)}")
    actual_duration = (logger.rows[-1]["timestamp_ms"] -
                       logger.rows[0]["timestamp_ms"]) / 1000.0 if logger.rows else 0
    print(f"  Actual duration: {actual_duration:.2f}s")
    print(f"  Effective rate: {len(logger.rows) / max(actual_duration, 0.001):.1f} Hz")


def main():
    parser = argparse.ArgumentParser(
        description="Crazyflie Sensor Logger — log all sensors during autonomous missions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python exp_sensor_logger_v1.py --mission static --duration 60  (Phase 1: IMU)
  python exp_sensor_logger_v1.py --mission hover --duration 10
  python exp_sensor_logger_v1.py --mission square --duration 30
  python exp_sensor_logger_v1.py --mission custom --waypoints waypoints.json
  python exp_sensor_logger_v1.py --mission hover --duration 5 --dry-run
        """,
    )
    parser.add_argument("--uri", default=DEFAULT_URI,
                        help=f"Crazyflie URI (default: {DEFAULT_URI})")
    parser.add_argument("--mission", required=True,
                        choices=["static", "hover", "square", "custom"],
                        help="Mission type to execute")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Mission duration in seconds (default: 10)")
    parser.add_argument("--altitude", type=float, default=0.5,
                        help="Flight altitude in meters (default: 0.5)")
    parser.add_argument("--waypoints", type=str, default=None,
                        help="Path to waypoints JSON file (for custom mission)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate synthetic data without hardware")

    args = parser.parse_args()

    print()
    if args.dry_run:
        dry_run(args.mission, args.duration)
    else:
        if not CFLIB_AVAILABLE:
            print("[ERROR] cflib not available. Use --dry-run or install cflib.")
            sys.exit(1)
        run_live(args.uri, args.mission, args.duration,
                 args.waypoints, args.altitude)


if __name__ == "__main__":
    main()
