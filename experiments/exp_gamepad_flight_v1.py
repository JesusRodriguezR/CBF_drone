"""
Crazyflie Gamepad Flight + Sensor Logger
==========================================

Fly the Crazyflie manually with a game controller (Xbox/PS) while logging
all sensor data. Identical logging format as exp_sensor_logger_v1.py with
additional controller-input columns.

PRE-FLIGHT CHECKLIST:
    [ ] Battery fully charged (≥3.7 V)
    [ ] CF firmware version confirmed
    [ ] Crazyradio PA connected
    [ ] Game controller plugged in and recognized by OS
    [ ] Flight area clear, props guard installed (recommended)
    [ ] OptiTrack streaming if needed (Phase 2+)

Usage:
    python exp_gamepad_flight_v1.py
    python exp_gamepad_flight_v1.py --uri radio://0/80/2M/E7E7E7E7E7
    python exp_gamepad_flight_v1.py --dry-run
"""

import argparse
import csv
import os
import sys
import time
import threading
from datetime import datetime

# ── pygame for controller input ──────────────────────────────────────────
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("[WARN] pygame not installed. Install with: pip install pygame")

# ── cflib imports ────────────────────────────────────────────────────────
try:
    import cflib.crtp
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.log import LogConfig
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
    CFLIB_AVAILABLE = True
except ImportError:
    CFLIB_AVAILABLE = False
    print("[WARN] cflib not installed. Install with: pip install cflib")

# ── Configuration ────────────────────────────────────────────────────────
DEFAULT_URI = "radio://0/80/2M/E7E7E7E7E7"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_results")
LOG_RATE_HZ = 100
CONTROL_RATE_HZ = 50  # Controller send rate

# ── Controller Mapping (Xbox layout) ────────────────────────────────────
# These axis/button indices work for most Xbox controllers on Windows.
# Adjust if using PS4/PS5 controller.
CONTROLLER_MAP = {
    "thrust_axis":    1,   # Left stick Y  (inverted: push up = negative)
    "yaw_axis":       0,   # Left stick X
    "pitch_axis":     3,   # Right stick Y (inverted)
    "roll_axis":      2,   # Right stick X
    "arm_button":     0,   # A button — arm/disarm toggle
    "estop_button":   1,   # B button — emergency stop
    "land_button":    3,   # Y button — auto-land
}

# ── Flight limits ────────────────────────────────────────────────────────
MAX_THRUST = 55000      # Max thrust PWM (out of 65535)
MIN_THRUST = 0          # Motors off
MAX_ROLL = 30.0         # Max roll angle [°]
MAX_PITCH = 30.0        # Max pitch angle [°]
MAX_YAW_RATE = 200.0    # Max yaw rate [°/s]
DEADZONE = 0.10         # Joystick deadzone (±10%)

# ── Log group definitions (same as sensor logger) ───────────────────────
LOG_GROUPS = {
    "imu": {
        "accel_x":  "acc.x",
        "accel_y":  "acc.y",
        "accel_z":  "acc.z",
        "gyro_x":   "gyro.x",
        "gyro_y":   "gyro.y",
        "gyro_z":   "gyro.z",
    },
    "baro_stab": {
        "baro_pressure":  "baro.pressure",
        "baro_asl":       "baro.asl",
        "stab_roll":      "stabilizer.roll",
        "stab_pitch":     "stabilizer.pitch",
        "stab_yaw":       "stabilizer.yaw",
    },
    "state_pos": {
        "stateEst_x":   "stateEstimate.x",
        "stateEst_y":   "stateEstimate.y",
        "stateEst_z":   "stateEstimate.z",
        "stateEst_vx":  "stateEstimate.vx",
        "stateEst_vy":  "stateEstimate.vy",
        "stateEst_vz":  "stateEstimate.vz",
    },
    "motors": {
        "motor_m1":  "motor.m1",
        "motor_m2":  "motor.m2",
        "motor_m3":  "motor.m3",
        "motor_m4":  "motor.m4",
    },
}

# CSV columns: sensor columns + controller input columns
SENSOR_COLUMNS = ["timestamp_ms"]
for group_cols in LOG_GROUPS.values():
    SENSOR_COLUMNS.extend(group_cols.keys())

CMD_COLUMNS = ["cmd_thrust", "cmd_roll", "cmd_pitch", "cmd_yaw"]
CSV_COLUMNS = SENSOR_COLUMNS + CMD_COLUMNS


def apply_deadzone(value: float, deadzone: float = DEADZONE) -> float:
    """Apply deadzone to joystick axis value."""
    if abs(value) < deadzone:
        return 0.0
    # Rescale to [0, 1] range outside deadzone
    sign = 1.0 if value > 0 else -1.0
    return sign * (abs(value) - deadzone) / (1.0 - deadzone)


class ControllerInput:
    """Reads game controller input via pygame."""

    def __init__(self):
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise RuntimeError(
                "No game controller found! Plug in an Xbox/PS controller."
            )

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print(f"  Controller: {self.joystick.get_name()}")
        print(f"  Axes: {self.joystick.get_numaxes()}")
        print(f"  Buttons: {self.joystick.get_numbuttons()}")

        self.armed = False
        self.estop = False
        self.land_requested = False
        self._prev_arm_btn = False

    def update(self):
        """Poll controller events and return current commands."""
        pygame.event.pump()

        # Read axes with deadzone
        raw_thrust = -self.joystick.get_axis(CONTROLLER_MAP["thrust_axis"])
        raw_yaw = self.joystick.get_axis(CONTROLLER_MAP["yaw_axis"])
        raw_pitch = -self.joystick.get_axis(CONTROLLER_MAP["pitch_axis"])
        raw_roll = self.joystick.get_axis(CONTROLLER_MAP["roll_axis"])

        thrust_norm = apply_deadzone(raw_thrust)
        yaw_norm = apply_deadzone(raw_yaw)
        pitch_norm = apply_deadzone(raw_pitch)
        roll_norm = apply_deadzone(raw_roll)

        # Map to CF command ranges
        # Thrust: 0 to MAX_THRUST (only positive)
        thrust = max(0.0, thrust_norm) * MAX_THRUST
        roll = roll_norm * MAX_ROLL
        pitch = pitch_norm * MAX_PITCH
        yaw_rate = yaw_norm * MAX_YAW_RATE

        # Buttons
        arm_btn = self.joystick.get_button(CONTROLLER_MAP["arm_button"])
        if arm_btn and not self._prev_arm_btn:
            self.armed = not self.armed
            state_str = "ARMED ⚠️" if self.armed else "DISARMED"
            print(f"\n  >>> {state_str} <<<\n")
        self._prev_arm_btn = arm_btn

        if self.joystick.get_button(CONTROLLER_MAP["estop_button"]):
            self.estop = True
            self.armed = False
            print("\n  >>> EMERGENCY STOP <<<\n")

        if self.joystick.get_button(CONTROLLER_MAP["land_button"]):
            self.land_requested = True
            print("\n  >>> AUTO-LAND REQUESTED <<<\n")

        # If not armed, zero all outputs
        if not self.armed or self.estop:
            thrust = 0.0
            roll = 0.0
            pitch = 0.0
            yaw_rate = 0.0

        return {
            "thrust": int(thrust),
            "roll": roll,
            "pitch": pitch,
            "yaw_rate": yaw_rate,
        }

    def cleanup(self):
        pygame.quit()


class SensorLogger:
    """Collects CF log data into rows for CSV export."""

    def __init__(self):
        self.rows = []
        self._current_row = {}
        self._start_time_ms = None
        self._groups_received = set()
        self._latest_sensors = {}
        self._lock = threading.Lock()

    def on_log_data(self, timestamp, data, logconf):
        """Callback for each log config data packet."""
        if self._start_time_ms is None:
            self._start_time_ms = timestamp

        relative_ms = timestamp - self._start_time_ms
        group_name = logconf.name

        with self._lock:
            self._current_row["timestamp_ms"] = relative_ms

            group_vars = LOG_GROUPS[group_name]
            for csv_col, cf_var in group_vars.items():
                if cf_var in data:
                    self._current_row[csv_col] = data[cf_var]

            self._groups_received.add(group_name)

            if len(self._groups_received) == len(LOG_GROUPS):
                self._latest_sensors = dict(self._current_row)
                self._current_row = {}
                self._groups_received.clear()

    def get_latest(self) -> dict:
        """Return latest sensor readings (thread-safe)."""
        with self._lock:
            return dict(self._latest_sensors)

    def add_row(self, sensor_data: dict, cmd_data: dict):
        """Append a combined sensor + command row."""
        row = {}
        row.update(sensor_data)
        row.update(cmd_data)
        self.rows.append(row)

    def save_csv(self, filepath: str):
        """Write collected rows to CSV."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({col: row.get(col, "") for col in CSV_COLUMNS})
        print(f"  ✓ Saved {len(self.rows)} samples to: {filepath}")


def print_hud(sensors: dict, cmd: dict, armed: bool, elapsed: float):
    """Print a compact real-time HUD to the terminal."""
    alt = sensors.get("stateEst_z", 0.0)
    roll = sensors.get("stab_roll", 0.0)
    pitch = sensors.get("stab_pitch", 0.0)
    yaw = sensors.get("stab_yaw", 0.0)
    baro = sensors.get("baro_asl", 0.0)

    state_str = "\033[91mARMED\033[0m" if armed else "\033[92mSAFE\033[0m"

    hud = (
        f"\r  [{state_str}] "
        f"T:{elapsed:5.1f}s │ "
        f"Alt:{alt:5.2f}m │ "
        f"R:{roll:+6.1f}° P:{pitch:+6.1f}° Y:{yaw:+6.1f}° │ "
        f"Thr:{cmd.get('thrust', 0):5d} │ "
        f"Baro:{baro:7.1f}m   "
    )
    sys.stdout.write(hud)
    sys.stdout.flush()


# ── Live flight mode ─────────────────────────────────────────────────────

def run_live(uri: str):
    """Main gamepad flight loop with live sensor logging."""
    print("╔══════════════════════════════════════════════════════╗")
    print("║       CRAZYFLIE GAMEPAD FLIGHT — LIVE               ║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  Controls (Xbox):                                   ║")
    print("║    Left stick Y  → Thrust       (push up)           ║")
    print("║    Left stick X  → Yaw rate                         ║")
    print("║    Right stick Y → Pitch        (push up = forward) ║")
    print("║    Right stick X → Roll                             ║")
    print("║    A button      → Arm / Disarm (toggle)            ║")
    print("║    B button      → EMERGENCY STOP                   ║")
    print("║    Y button      → Auto-land                        ║")
    print("║                                                     ║")
    print("║  Press Ctrl+C to end session and save log.          ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    cflib.crtp.init_drivers()
    controller = ControllerInput()
    logger = SensorLogger()

    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache="./cache")) as scf:
        cf = scf.cf
        print(f"  Connected to: {uri}")

        # Set up log configs
        log_configs = []
        for group_name, variables in LOG_GROUPS.items():
            lc = LogConfig(name=group_name, period_in_ms=int(1000 / LOG_RATE_HZ))
            for csv_col, cf_var in variables.items():
                lc.add_variable(cf_var, "float")
            cf.log.add_config(lc)
            lc.data_received_cb.add_callback(logger.on_log_data)
            log_configs.append(lc)

        # Start logging
        for lc in log_configs:
            lc.start()
        print(f"  Sensor logging at {LOG_RATE_HZ} Hz")
        print(f"  Control rate: {CONTROL_RATE_HZ} Hz")
        print()
        print("  Press A to arm, then apply thrust to fly.")
        print("  " + "─" * 50)

        start_time = time.time()
        dt_ctrl = 1.0 / CONTROL_RATE_HZ
        hud_counter = 0

        try:
            while not controller.estop:
                loop_start = time.time()
                elapsed = loop_start - start_time

                # Read controller
                cmd = controller.update()

                # Send to CF
                cf.commander.send_setpoint(
                    cmd["roll"],
                    cmd["pitch"],
                    cmd["yaw_rate"],
                    cmd["thrust"],
                )

                # Log combined row
                sensors = logger.get_latest()
                cmd_row = {
                    "cmd_thrust": cmd["thrust"],
                    "cmd_roll": round(cmd["roll"], 2),
                    "cmd_pitch": round(cmd["pitch"], 2),
                    "cmd_yaw": round(cmd["yaw_rate"], 2),
                }
                logger.add_row(sensors, cmd_row)

                # HUD update at ~10 Hz
                hud_counter += 1
                if hud_counter % (CONTROL_RATE_HZ // 10) == 0:
                    print_hud(sensors, cmd, controller.armed, elapsed)

                # Auto-land sequence
                if controller.land_requested:
                    print("\n  Auto-landing...")
                    land_thrust = cmd.get("thrust", 30000)
                    for t in range(50):  # ~2.5s descent
                        land_thrust = max(0, land_thrust - 600)
                        cf.commander.send_setpoint(0, 0, 0, int(land_thrust))
                        time.sleep(0.05)
                    cf.commander.send_setpoint(0, 0, 0, 0)
                    time.sleep(0.5)
                    break

                # Rate limit
                elapsed_loop = time.time() - loop_start
                sleep_time = dt_ctrl - elapsed_loop
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n\n  Ctrl+C — shutting down...")

        finally:
            # Kill motors
            cf.commander.send_setpoint(0, 0, 0, 0)
            time.sleep(0.1)
            cf.commander.send_stop_setpoint()

            # Stop logging
            for lc in log_configs:
                lc.stop()

            controller.cleanup()

    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = f"gamepad_flight_{timestamp}.csv"
    filepath = os.path.join(RESULTS_DIR, filename)
    logger.save_csv(filepath)

    total_time = time.time() - start_time
    print(f"  Flight time: {total_time:.1f}s")
    print(f"  Samples logged: {len(logger.rows)}")


# ── Dry-run mode ─────────────────────────────────────────────────────────

def dry_run():
    """Simulate the gamepad flight pipeline without hardware."""
    import random

    print("╔══════════════════════════════════════════════════════╗")
    print("║       GAMEPAD FLIGHT — DRY RUN                      ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # Test controller if available
    if PYGAME_AVAILABLE:
        try:
            controller = ControllerInput()
            print("  Controller detected! Testing for 5 seconds...")
            print("  Move sticks and press buttons to verify mapping.\n")

            start = time.time()
            while time.time() - start < 5.0:
                cmd = controller.update()
                sys.stdout.write(
                    f"\r  Thr:{cmd['thrust']:5d} "
                    f"R:{cmd['roll']:+6.1f}° "
                    f"P:{cmd['pitch']:+6.1f}° "
                    f"Y:{cmd['yaw_rate']:+6.1f}°/s "
                    f"{'ARMED' if controller.armed else 'safe'}   "
                )
                sys.stdout.flush()
                time.sleep(0.02)

            controller.cleanup()
            print("\n  ✓ Controller test complete.\n")
        except RuntimeError as e:
            print(f"  Controller test skipped: {e}\n")
    else:
        print("  pygame not available — skipping controller test.\n")

    # Generate synthetic CSV
    logger = SensorLogger()
    n_samples = 500  # ~10s at 50 Hz
    print(f"  Generating {n_samples} synthetic samples...")

    for i in range(n_samples):
        row = {"timestamp_ms": i * 20}  # 50 Hz
        for col in SENSOR_COLUMNS[1:]:
            if "motor" in col:
                row[col] = random.randint(10000, 40000)
            elif "accel" in col:
                row[col] = round(random.gauss(0, 0.5), 4)
            elif "gyro" in col:
                row[col] = round(random.gauss(0, 2.0), 4)
            elif "stab" in col:
                row[col] = round(random.gauss(0, 1.0), 3)
            else:
                row[col] = round(random.gauss(0, 0.01), 4)
        row["cmd_thrust"] = random.randint(0, 55000)
        row["cmd_roll"] = round(random.gauss(0, 5.0), 2)
        row["cmd_pitch"] = round(random.gauss(0, 5.0), 2)
        row["cmd_yaw"] = round(random.gauss(0, 20.0), 2)
        logger.rows.append(row)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = f"gamepad_flight_{timestamp}_DRYRUN.csv"
    filepath = os.path.join(RESULTS_DIR, filename)
    logger.save_csv(filepath)

    print(f"\n  CSV columns ({len(CSV_COLUMNS)}):")
    for col in CSV_COLUMNS:
        print(f"    • {col}")
    print("\n  ✓ Dry run complete.")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Crazyflie Gamepad Flight — fly with controller while logging all sensors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls (Xbox):
  Left stick Y   Thrust (push up to ascend)
  Left stick X   Yaw rate
  Right stick Y  Pitch (push up = forward)
  Right stick X  Roll
  A button       Arm / Disarm (toggle)
  B button       EMERGENCY STOP
  Y button       Auto-land
        """,
    )
    parser.add_argument("--uri", default=DEFAULT_URI,
                        help=f"Crazyflie URI (default: {DEFAULT_URI})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test controller mapping and generate synthetic CSV")

    args = parser.parse_args()

    print()
    if args.dry_run:
        dry_run()
    else:
        if not CFLIB_AVAILABLE:
            print("[ERROR] cflib not available. Use --dry-run or install cflib.")
            sys.exit(1)
        if not PYGAME_AVAILABLE:
            print("[ERROR] pygame not available. Install with: pip install pygame")
            sys.exit(1)
        run_live(args.uri)


if __name__ == "__main__":
    main()
