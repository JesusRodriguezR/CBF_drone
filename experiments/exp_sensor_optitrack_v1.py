"""
Crazyflie Sensor Logger + OptiTrack Ground Truth — Direct Control
==================================================================

Logs all onboard CF sensor data AND OptiTrack rigid body position/rotation
simultaneously. Uses DIRECT thrust/attitude commands with OptiTrack-based
PID control (no MotionCommander — the bare CF2.1 has no onboard position
estimator without a Flow deck).

Two CSV files are saved:
    1. sensor_optitrack_{mission}_{timestamp}.csv — merged CF + OptiTrack data
    2. optitrack_raw_{timestamp}.csv             — full-rate OptiTrack data (120 Hz)

PRE-FLIGHT CHECKLIST:
    [ ] Battery fully charged (≥3.7 V)
    [ ] Crazyradio PA connected
    [ ] OptiTrack Motive running, Broadcast Frame Data ON
    [ ] Rigid body defined in Motive (check --rigid-body-id)
    [ ] Flight area clear, ground is level

Usage:
    python exp_sensor_optitrack_v1.py --mission hover --duration 10
    python exp_sensor_optitrack_v1.py --mission hover --duration 10 --altitude 0.4
    python exp_sensor_optitrack_v1.py --mission square --duration 30
    python exp_sensor_optitrack_v1.py --mission static --duration 30
    python exp_sensor_optitrack_v1.py --mission hover --duration 5 --dry-run
"""

import argparse
import csv
import math
import os
import signal
import socket
import struct
import sys
import threading
import time
from datetime import datetime
from io import BytesIO

# ── cflib imports ────────────────────────────────────────────────────────
try:
    import cflib.crtp
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.log import LogConfig
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
    CFLIB_AVAILABLE = True
except ImportError:
    CFLIB_AVAILABLE = False
    print("[WARN] cflib not installed. Running in dry-run mode only.\n")

# ── Configuration ────────────────────────────────────────────────────────
DEFAULT_URI = "radio://0/80/2M/E7E7E7E7E7"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_results")
LOG_RATE_HZ = 100
OPTITRACK_PORT = 1511
DEFAULT_RB_ID = 16

# ── PID controller (from proven square_working_simple.py) ────────────────
# Gains calibrated on the real CF2.1 hardware
BASE_THRUST = 52000
ALT_KP, ALT_KI, ALT_KD = 4000, 300, 1000
ALT_I_LIMIT, ALT_OUT_LIMIT = 0.1, 5000
XY_P = 8.0       # deg per meter error
XY_D = 8.0       # deg per (m/s) damping
MAX_ANGLE = 12.0  # deg
MIN_THRUST, MAX_THRUST = 15000, 65000


# ── CF Log column definitions ───────────────────────────────────────────
LOG_GROUPS = {
    "imu": {
        "accel_x":  "acc.x", "accel_y":  "acc.y", "accel_z":  "acc.z",
        "gyro_x":   "gyro.x", "gyro_y":   "gyro.y", "gyro_z":   "gyro.z",
    },
    "baro_stab": {
        "baro_pressure": "baro.pressure", "baro_asl": "baro.asl",
        "stab_roll": "stabilizer.roll", "stab_pitch": "stabilizer.pitch",
        "stab_yaw": "stabilizer.yaw",
    },
    "state_pos": {
        "stateEst_x": "stateEstimate.x", "stateEst_y": "stateEstimate.y",
        "stateEst_z": "stateEstimate.z", "stateEst_vx": "stateEstimate.vx",
        "stateEst_vy": "stateEstimate.vy", "stateEst_vz": "stateEstimate.vz",
    },
    "motors": {
        "motor_m1": "motor.m1", "motor_m2": "motor.m2",
        "motor_m3": "motor.m3", "motor_m4": "motor.m4",
    },
}

CF_COLUMNS = ["timestamp_ms"]
for gc in LOG_GROUPS.values():
    CF_COLUMNS.extend(gc.keys())

OT_COLUMNS = [
    "ot_x", "ot_y", "ot_z",
    "ot_qx", "ot_qy", "ot_qz", "ot_qw",
    "ot_roll", "ot_pitch", "ot_yaw", "ot_valid",
]

ALL_COLUMNS = CF_COLUMNS + OT_COLUMNS


# ═════════════════════════════════════════════════════════════════════════
# NatNet broadcast parser
# ═════════════════════════════════════════════════════════════════════════

def _read_cstring(stream: BytesIO) -> str:
    chars = []
    while True:
        c = stream.read(1)
        if c == b'\x00' or c == b'':
            break
        chars.append(c)
    return b''.join(chars).decode('utf-8', errors='replace')


def parse_frame_of_data(data: bytes) -> list:
    stream = BytesIO(data)
    msg_id, _ = struct.unpack('<HH', stream.read(4))
    if msg_id != 7:
        return []
    frame_number = struct.unpack('<I', stream.read(4))[0]
    for _ in range(struct.unpack('<I', stream.read(4))[0]):
        _read_cstring(stream)
        n = struct.unpack('<I', stream.read(4))[0]
        stream.read(n * 12)
    n = struct.unpack('<I', stream.read(4))[0]
    stream.read(n * 12)
    bodies = []
    for _ in range(struct.unpack('<I', stream.read(4))[0]):
        rb = stream.read(32)
        if len(rb) < 32:
            break
        body_id = struct.unpack('<i', rb[0:4])[0]
        x, y, z = struct.unpack('<3f', rb[4:16])
        qx, qy, qz, qw = struct.unpack('<4f', rb[16:32])
        struct.unpack('<f', stream.read(4))  # mean_error
        try:
            params = struct.unpack('<H', stream.read(2))[0]
            valid = bool(params & 0x01)
        except struct.error:
            valid = True
        bodies.append({
            'id': body_id, 'pos': (x, y, z),
            'rot': (qx, qy, qz, qw), 'valid': valid,
            'frame': frame_number,
        })
    return bodies


def quat_to_euler(qx, qy, qz, qw):
    sinr = 2.0 * (qw * qx + qy * qz)
    cosr = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr, cosr)
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny, cosy)
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


# ═════════════════════════════════════════════════════════════════════════
# OptiTrack receiver thread
# ═════════════════════════════════════════════════════════════════════════

class OptiTrackReceiver:
    def __init__(self, rigid_body_id=DEFAULT_RB_ID, port=OPTITRACK_PORT):
        self.rb_id = rigid_body_id
        self.port = port
        self.latest = None
        self.raw_log = []
        self._running = False
        self._thread = None
        self._start_time = None

    def start(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self._sock.bind(("", self.port))
        self._sock.settimeout(0.5)
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        try:
            self._sock.close()
        except:
            pass

    def _recv_loop(self):
        while self._running:
            try:
                data, _ = self._sock.recvfrom(65535)
                if struct.unpack('<H', data[0:2])[0] != 7:
                    continue
                for b in parse_frame_of_data(data):
                    if b['id'] == self.rb_id:
                        x, y, z = b['pos']
                        qx, qy, qz, qw = b['rot']
                        roll, pitch, yaw = quat_to_euler(qx, qy, qz, qw)
                        entry = {
                            'time_s': time.time() - self._start_time,
                            'frame': b['frame'],
                            'x': x, 'y': y, 'z': z,
                            'qx': qx, 'qy': qy, 'qz': qz, 'qw': qw,
                            'roll': roll, 'pitch': pitch, 'yaw': yaw,
                            'valid': b['valid'],
                        }
                        self.latest = entry
                        self.raw_log.append(entry)
                        break
            except socket.timeout:
                pass
            except:
                pass

    def get_position(self):
        """Returns (x, y, z) or None."""
        e = self.latest
        return (e['x'], e['y'], e['z']) if e else None

    def get_snapshot(self):
        e = self.latest
        if e is None:
            return {}
        return {
            "ot_x": e['x'], "ot_y": e['y'], "ot_z": e['z'],
            "ot_qx": e['qx'], "ot_qy": e['qy'],
            "ot_qz": e['qz'], "ot_qw": e['qw'],
            "ot_roll": e['roll'], "ot_pitch": e['pitch'],
            "ot_yaw": e['yaw'],
            "ot_valid": 1 if e['valid'] else 0,
        }

    def save_raw_csv(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        cols = ['time_s', 'frame', 'x', 'y', 'z',
                'qx', 'qy', 'qz', 'qw', 'roll', 'pitch', 'yaw', 'valid']
        with open(filepath, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in self.raw_log:
                w.writerow(r)
        print(f"  ✓ Saved {len(self.raw_log)} OptiTrack samples to: {filepath}")


# ═════════════════════════════════════════════════════════════════════════
# PID controller
# ═════════════════════════════════════════════════════════════════════════

class PID:
    def __init__(self, kp, ki=0.0, kd=0.0, i_limit=None, out_limit=None):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i_limit, self.out_limit = i_limit, out_limit
        self.integral, self.prev_err = 0.0, None

    def reset(self):
        self.integral, self.prev_err = 0.0, None

    def update(self, err, dt):
        if dt <= 0:
            return 0.0
        self.integral += err * dt
        if self.i_limit is not None:
            self.integral = max(-self.i_limit, min(self.i_limit, self.integral))
        d = (err - self.prev_err) / dt if self.prev_err is not None else 0.0
        self.prev_err = err
        out = self.kp * err + self.ki * self.integral + self.kd * d
        if self.out_limit is not None:
            out = max(-self.out_limit, min(self.out_limit, out))
        return out


# ═════════════════════════════════════════════════════════════════════════
# CF Sensor Logger (with OptiTrack merge)
# ═════════════════════════════════════════════════════════════════════════

def build_log_configs(log_groups, rate_hz):
    configs = []
    for name, variables in log_groups.items():
        lc = LogConfig(name=name, period_in_ms=int(1000 / rate_hz))
        for csv_col, cf_var in variables.items():
            lc.add_variable(cf_var, "float")
        configs.append((name, lc))
    return configs


class SensorOptiTrackLogger:
    def __init__(self, optitrack=None):
        self.rows = []
        self._current_row = {}
        self._start_time_ms = None
        self._groups_received = set()
        self._optitrack = optitrack

    def on_log_data(self, timestamp, data, logconf):
        if self._start_time_ms is None:
            self._start_time_ms = timestamp
        self._current_row["timestamp_ms"] = timestamp - self._start_time_ms

        for csv_col, cf_var in LOG_GROUPS[logconf.name].items():
            if cf_var in data:
                self._current_row[csv_col] = data[cf_var]
        self._groups_received.add(logconf.name)

        if len(self._groups_received) == len(LOG_GROUPS):
            if self._optitrack:
                self._current_row.update(self._optitrack.get_snapshot())
            self.rows.append(dict(self._current_row))
            self._current_row = {}
            self._groups_received.clear()

    def save_csv(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
            w.writeheader()
            for r in self.rows:
                w.writerow({c: r.get(c, "") for c in ALL_COLUMNS})
        print(f"  ✓ Saved {len(self.rows)} merged samples to: {filepath}")


# ═════════════════════════════════════════════════════════════════════════
# Live flight with direct PID control
# ═════════════════════════════════════════════════════════════════════════

def run_live(uri, mission, duration, altitude=0.5, rb_id=DEFAULT_RB_ID):

    print("╔══════════════════════════════════════════════════════════╗")
    print("║    CRAZYFLIE SENSOR + OPTITRACK — DIRECT CONTROL       ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # Graceful shutdown
    stop_flag = threading.Event()
    def on_signal(sig, frame):
        print("\n  ⚠️  Ctrl+C — landing...")
        stop_flag.set()
    signal.signal(signal.SIGINT, on_signal)

    # ── Start OptiTrack ──────────────────────────────────────────────
    ot = OptiTrackReceiver(rigid_body_id=rb_id)
    ot.start()
    print(f"  OptiTrack receiver started (RB ID: {rb_id})")

    print("  Waiting for OptiTrack data...", end="", flush=True)
    t0 = time.time()
    while ot.latest is None and time.time() - t0 < 10.0:
        time.sleep(0.1)
        print(".", end="", flush=True)

    if ot.latest is None:
        print("\n  [ERROR] No OptiTrack data! Check Motive.")
        ot.stop()
        return

    p = ot.latest
    print(f"\n  ✅ OptiTrack: X={p['x']:+.3f}  Y={p['y']:+.3f}  Z={p['z']:+.3f}")
    print()

    # ── Connect to CF ────────────────────────────────────────────────
    cflib.crtp.init_drivers()
    logger = SensorOptiTrackLogger(optitrack=ot)

    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache="./cache")) as scf:
        cf = scf.cf
        print(f"  Connected to CF: {uri}")

        # Set up sensor logging
        log_configs = build_log_configs(LOG_GROUPS, LOG_RATE_HZ)
        for name, lc in log_configs:
            cf.log.add_config(lc)
            lc.data_received_cb.add_callback(logger.on_log_data)
        for _, lc in log_configs:
            lc.start()
        print(f"  CF logging at {LOG_RATE_HZ} Hz")
        print(f"  Mission: {mission} | Duration: {duration}s | Altitude: {altitude}m")
        print()

        if mission == "static":
            # ── Static: no motors ────────────────────────────────────
            print("  ▸ STATIC — motors OFF, logging CF + OptiTrack")
            for elapsed in range(int(duration)):
                if stop_flag.is_set():
                    break
                time.sleep(1.0)
                snap = ot.latest or {}
                sys.stdout.write(
                    f"\r    ⏱ {elapsed+1}s/{duration:.0f}s  "
                    f"[CF:{len(logger.rows)} OT:{len(ot.raw_log)}]  "
                    f"({snap.get('x',0):+.2f},{snap.get('y',0):+.2f},{snap.get('z',0):+.2f})   "
                )
                sys.stdout.flush()
            print()

        else:
            # ── Flight with PID control ──────────────────────────────
            alt_pid = PID(ALT_KP, ALT_KI, ALT_KD,
                          i_limit=ALT_I_LIMIT, out_limit=ALT_OUT_LIMIT)

            # Capture initial position
            pos = ot.get_position()
            target_x, target_z = pos[0], pos[2]
            prev_x, prev_y, prev_z = pos
            prev_time = time.time()

            print(f"  ▸ Initial pos: X={pos[0]:+.3f} Y={pos[1]:+.3f} Z={pos[2]:+.3f}")
            print(f"  ▸ Target: altitude={altitude}m, hold X={target_x:.3f} Z={target_z:.3f}")
            print()

            # Safe motor init
            cf.commander.send_setpoint(0, 0, 0, 0)
            time.sleep(0.3)

            # ── Thrust ramp takeoff (from proven controller) ─────────
            print("  ▸ TAKEOFF — thrust ramp...")
            ramp_steps = 6
            for step in range(ramp_steps):
                ramp_thrust = int(25000 + (BASE_THRUST - 25000) * (step + 1) / ramp_steps)
                cf.commander.send_setpoint(0, 0, 0, ramp_thrust)
                time.sleep(0.15)

            # ── Main control loop ────────────────────────────────────
            print("  ▸ HOVERING — PID active")
            flight_start = time.time()
            log_interval = 0.2
            last_log = 0.0

            while not stop_flag.is_set():
                elapsed = time.time() - flight_start
                if elapsed >= duration:
                    break

                pos = ot.get_position()
                if pos is None:
                    cf.commander.send_setpoint(0, 0, 0, BASE_THRUST)
                    time.sleep(0.01)
                    continue

                x, y, z = pos
                now = time.time()
                dt = now - prev_time
                dt = max(0.001, min(0.1, dt))
                prev_time = now

                # Velocities
                vx = (x - prev_x) / dt
                vy = (y - prev_y) / dt
                vz = (z - prev_z) / dt
                prev_x, prev_y, prev_z = x, y, z

                # Lag compensation
                lag = 0.1
                pred_x = x + vx * lag
                pred_y = y + vy * lag
                pred_z = z + vz * lag

                # Errors
                h_err = altitude - pred_y      # OptiTrack: +Y = UP
                x_err = target_x - pred_x
                z_err = target_z - pred_z

                # Deadband
                if abs(x_err) < 0.02:
                    x_err = 0
                if abs(z_err) < 0.02:
                    z_err = 0

                # Altitude PID
                alt_delta = alt_pid.update(h_err, dt)
                thrust = int(BASE_THRUST + alt_delta)
                thrust = max(MIN_THRUST, min(MAX_THRUST, thrust))

                # Lateral P+D (from proven controller)
                roll_cmd = -(XY_P * x_err - XY_D * vx)   # inverted for OptiTrack coords
                pitch_cmd = XY_P * z_err - XY_D * vz
                roll_cmd = max(-MAX_ANGLE, min(MAX_ANGLE, roll_cmd))
                pitch_cmd = max(-MAX_ANGLE, min(MAX_ANGLE, pitch_cmd))

                cf.commander.send_setpoint(roll_cmd, pitch_cmd, 0, thrust)

                # Status
                if now - last_log >= log_interval:
                    last_log = now
                    remaining = duration - elapsed
                    sys.stdout.write(
                        f"\r    H:{y:.2f}→{altitude}  X:{x:+.2f}  Z:{z:+.2f}  "
                        f"T:{thrust}  R:{roll_cmd:+.1f}° P:{pitch_cmd:+.1f}°  "
                        f"[{remaining:.0f}s]  CF:{len(logger.rows)}  "
                    )
                    sys.stdout.flush()

                time.sleep(0.02)  # ~50 Hz control loop

            # ── Landing (gentle thrust ramp-down) ──────────────────
            print("\n  ▸ LANDING...")
            for step in range(20):
                landing_thrust = max(10000, int(BASE_THRUST * (1 - (step + 1) / 20)))
                cf.commander.send_setpoint(0, 0, 0, landing_thrust)
                time.sleep(0.15)

            # Extra gentle final
            for step in range(5):
                cf.commander.send_setpoint(0, 0, 0, max(5000, 8000 - step * 600))
                time.sleep(0.2)

            cf.commander.send_stop_setpoint()
            print("  ✅ Motors stopped")
            time.sleep(0.5)

        # Stop CF logging
        for _, lc in log_configs:
            lc.stop()

    # ── Stop OptiTrack and save ──────────────────────────────────────
    ot.stop()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    merged_path = os.path.join(RESULTS_DIR,
                               f"sensor_optitrack_{mission}_{timestamp}.csv")
    logger.save_csv(merged_path)

    ot_path = os.path.join(RESULTS_DIR, f"optitrack_raw_{timestamp}.csv")
    ot.save_raw_csv(ot_path)

    print()
    if logger.rows:
        dur = (logger.rows[-1]["timestamp_ms"] -
               logger.rows[0]["timestamp_ms"]) / 1000.0
        print(f"  CF samples:  {len(logger.rows)}  ({len(logger.rows)/max(dur,0.001):.1f} Hz)")
        print(f"  OT samples:  {len(ot.raw_log)}  ({len(ot.raw_log)/max(dur,0.001):.1f} Hz)")
        print(f"  Duration:    {dur:.2f}s")


# ═════════════════════════════════════════════════════════════════════════
# Dry run
# ═════════════════════════════════════════════════════════════════════════

def dry_run(mission, duration):
    import random
    print("╔══════════════════════════════════════════════════════════╗")
    print("║    SENSOR + OPTITRACK — DRY RUN                        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    n = int(duration * LOG_RATE_HZ)
    rows = []
    for i in range(n):
        t_ms = i * (1000 / LOG_RATE_HZ)
        rows.append({
            "timestamp_ms": t_ms,
            "accel_x": random.gauss(0, 0.01), "accel_y": random.gauss(0, 0.01),
            "accel_z": random.gauss(1.0, 0.01),
            "gyro_x": random.gauss(0, 0.5), "gyro_y": random.gauss(0, 0.5),
            "gyro_z": random.gauss(0, 0.3),
            "baro_pressure": 791.5, "baro_asl": 2229.0,
            "stab_roll": random.gauss(0, 1), "stab_pitch": random.gauss(0, 1),
            "stab_yaw": 0,
            "stateEst_x": 0, "stateEst_y": 0, "stateEst_z": 0.5,
            "stateEst_vx": 0, "stateEst_vy": 0, "stateEst_vz": 0,
            "motor_m1": 32000, "motor_m2": 32000,
            "motor_m3": 32000, "motor_m4": 32000,
            "ot_x": 2.5, "ot_y": 0.5, "ot_z": -0.2,
            "ot_qx": 0, "ot_qy": 0, "ot_qz": 0, "ot_qw": -1,
            "ot_roll": 0, "ot_pitch": 0, "ot_yaw": 0, "ot_valid": 1,
        })

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fp = os.path.join(RESULTS_DIR, f"sensor_optitrack_{mission}_{timestamp}_DRYRUN.csv")
    with open(fp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in ALL_COLUMNS})
    print(f"  Generated {n} samples → {fp}")
    print("  ✓ Dry run complete.")


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CF Sensor + OptiTrack Logger — direct PID control",
    )
    parser.add_argument("--uri", default=DEFAULT_URI)
    parser.add_argument("--mission", required=True,
                        choices=["static", "hover", "square"])
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--altitude", type=float, default=0.5)
    parser.add_argument("--rigid-body-id", type=int, default=DEFAULT_RB_ID)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    print()

    if args.dry_run:
        dry_run(args.mission, args.duration)
    else:
        if not CFLIB_AVAILABLE:
            print("[ERROR] cflib not available.")
            sys.exit(1)
        run_live(args.uri, args.mission, args.duration,
                 args.altitude, args.rigid_body_id)


if __name__ == "__main__":
    main()
