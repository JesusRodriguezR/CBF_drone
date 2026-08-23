"""
Digital Twin Validation Engine
===============================

Replays physical flight data through the simulation and computes
fidelity metrics comparing sim vs real behaviour.

Two validation modes:
  1. Open-loop:  replay recorded motor PWMs through sim physics
  2. Closed-loop: run sim PID tracking same position setpoints

Usage
-----
    python validate_twin.py path/to/flight.csv
    python validate_twin.py path/to/flight.csv --mode closed-loop
    python validate_twin.py --sim-only --manoeuvre hover   # no hardware needed
"""

import argparse
import csv
import sys
import os
from pathlib import Path

import numpy as np

# Add simulation modules to path
SCRIPT_DIR = Path(__file__).resolve().parent
SIM_DIR = SCRIPT_DIR.parent.parent / "simulations"
CTRL_DIR = SCRIPT_DIR.parent.parent / "control_arq_scripts"
sys.path.insert(0, str(SIM_DIR))
sys.path.insert(0, str(CTRL_DIR))

from crazyflie_params import CrazyflieParams
from crazyflie_sim import CrazyflieSim, SimConfig
from rigid_body_dynamics import quat_to_euler, POS, VEL, QUAT, OMEGA
from pid_controller import CrazyflieController

import validation_config as cfg


# ═══════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════

def load_flight_csv(csv_path: str) -> dict:
    """Load a validation flight CSV into numpy arrays."""
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"Empty CSV: {csv_path}")

    n = len(rows)
    data = {
        "t":     np.array([float(r["t"]) for r in rows]),
        "x":     np.array([float(r["x"]) for r in rows]),
        "y":     np.array([float(r["y"]) for r in rows]),
        "z":     np.array([float(r["z"]) for r in rows]),
        "vx":    np.array([float(r["vx"]) for r in rows]),
        "vy":    np.array([float(r["vy"]) for r in rows]),
        "vz":    np.array([float(r["vz"]) for r in rows]),
        "m1":    np.array([float(r["m1"]) for r in rows]),
        "m2":    np.array([float(r["m2"]) for r in rows]),
        "m3":    np.array([float(r["m3"]) for r in rows]),
        "m4":    np.array([float(r["m4"]) for r in rows]),
        "vbat":  np.array([float(r["vbat"]) for r in rows]),
    }

    # Optional columns (may not exist in older IMAV data)
    optional = {
        "roll": "roll", "pitch": "pitch", "yaw": "yaw",
        "gyro_x": "gyro_x", "gyro_y": "gyro_y", "gyro_z": "gyro_z",
        "acc_x": "acc_x", "acc_y": "acc_y", "acc_z": "acc_z",
        "x_ref": "x_ref", "y_ref": "y_ref", "z_ref": "z_ref",
        "yaw_ref": "yaw_ref",
    }
    for csv_col, key in optional.items():
        if csv_col in rows[0]:
            data[key] = np.array([float(r[csv_col]) for r in rows])

    print(f"  Loaded {n} samples, {data['t'][-1]:.1f}s from {csv_path}")
    return data


def load_imav_csv(csv_path: str) -> dict:
    """Load an IMAV-format CSV (x_ref/y_ref/z_ref, no attitude)."""
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    n = len(rows)
    data = {
        "t":  np.array([float(r["t"]) for r in rows]),
        "x":  np.array([float(r["x"]) for r in rows]),
        "y":  np.array([float(r["y"]) for r in rows]),
        "z":  np.array([float(r["z"]) for r in rows]),
        "vx": np.array([float(r["vx"]) for r in rows]),
        "vy": np.array([float(r["vy"]) for r in rows]),
        "vz": np.array([float(r["vz"]) for r in rows]),
        "m1": np.array([float(r["m1"]) for r in rows]),
        "m2": np.array([float(r["m2"]) for r in rows]),
        "m3": np.array([float(r["m3"]) for r in rows]),
        "m4": np.array([float(r["m4"]) for r in rows]),
        "vbat": np.array([float(r["vbat"]) for r in rows]),
        "x_ref":   np.array([float(r["x_ref"]) for r in rows]),
        "y_ref":   np.array([float(r["y_ref"]) for r in rows]),
        "z_ref":   np.array([float(r["z_ref"]) for r in rows]),
        "yaw_ref": np.array([float(r["yaw_ref"]) for r in rows]),
    }
    print(f"  Loaded IMAV data: {n} samples, {data['t'][-1]:.1f}s")
    return data


# ═══════════════════════════════════════════════════════════════════════
# Fidelity Metrics
# ═══════════════════════════════════════════════════════════════════════

def compute_metrics(real: np.ndarray, sim: np.ndarray, label: str = "") -> dict:
    """Compute comparison metrics between two aligned time series."""
    error = sim - real
    n = len(error)

    rmse = np.sqrt(np.mean(error**2))
    mae = np.mean(np.abs(error))
    max_err = np.max(np.abs(error))

    # Normalised RMSE (as % of signal range)
    signal_range = np.ptp(real)
    nrmse = (rmse / signal_range * 100) if signal_range > 1e-9 else float('inf')

    # Cross-correlation (normalised)
    if np.std(real) > 1e-9 and np.std(sim) > 1e-9:
        corr = np.corrcoef(real, sim)[0, 1]
    else:
        corr = float('nan')

    # Theil's U-statistic
    num = np.sqrt(np.mean(error**2))
    den = np.sqrt(np.mean(real**2)) + np.sqrt(np.mean(sim**2))
    theil_u = num / den if den > 1e-12 else float('inf')

    # Theil decomposition (bias, variance, covariance proportions)
    mean_err = np.mean(error)
    bias_prop = mean_err**2 / (rmse**2) if rmse > 1e-12 else 0.0
    var_prop = (np.std(sim) - np.std(real))**2 / (rmse**2) if rmse > 1e-12 else 0.0
    cov_prop = max(0.0, 1.0 - bias_prop - var_prop)

    return {
        "label":    label,
        "rmse":     rmse,
        "mae":      mae,
        "max_err":  max_err,
        "nrmse":    nrmse,
        "corr":     corr,
        "theil_u":  theil_u,
        "bias_pct": bias_prop * 100,
        "var_pct":  var_prop * 100,
        "cov_pct":  cov_prop * 100,
    }


def print_metrics_table(all_metrics: list):
    """Print a formatted comparison table."""
    print(f"\n  {'Channel':<12} {'RMSE':>10} {'MAE':>10} {'MaxErr':>10} "
          f"{'NRMSE%':>8} {'Corr':>7} {'Theil-U':>8}")
    print(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*10} {'─'*8} {'─'*7} {'─'*8}")
    for m in all_metrics:
        corr_str = f"{m['corr']:.4f}" if not np.isnan(m['corr']) else "  N/A"
        print(f"  {m['label']:<12} {m['rmse']:10.5f} {m['mae']:10.5f} "
              f"{m['max_err']:10.5f} {m['nrmse']:7.1f}% {corr_str:>7} "
              f"{m['theil_u']:8.4f}")


# ═══════════════════════════════════════════════════════════════════════
# Open-Loop Replay
# ═══════════════════════════════════════════════════════════════════════

def replay_open_loop(data: dict, params: CrazyflieParams = None) -> dict:
    """
    Feed recorded motor PWMs into the sim and record resulting state.

    This is the strongest test: identical motor inputs should produce
    identical state evolution if the physics model is accurate.
    """
    params = params or CrazyflieParams.brushless()
    sim_config = SimConfig(
        dt=0.001,
        enable_sensors=False,
        enable_ground_collision=True,
        enable_ground_effect=True,
        enable_aerodynamics=True,
        log_data=False,
    )
    sim = CrazyflieSim(params, sim_config)

    # Initial state from physical data (NED: z positive = down)
    init_pos = np.array([data["x"][0], data["y"][0], -data["z"][0]])
    sim.reset(position=init_pos)

    # Pre-spin motors to approximate hover state
    hover_pwm = (-params.pwm_thrust_b + np.sqrt(
        params.pwm_thrust_b**2 + 4 * params.pwm_thrust_a * (params.weight / 4.0)
    )) / (2 * params.pwm_thrust_a)
    for _ in range(200):
        sim.step(np.array([hover_pwm]*4), command_type='pwm')
    sim.state[VEL] = np.zeros(3)
    sim.state[OMEGA] = np.zeros(3)
    sim.time = 0.0

    t_phys = data["t"]
    n = len(t_phys)
    sim_dt = sim_config.dt

    # Allocate output arrays
    sim_out = {
        "t": np.zeros(n),
        "x": np.zeros(n), "y": np.zeros(n), "z": np.zeros(n),
        "vx": np.zeros(n), "vy": np.zeros(n), "vz": np.zeros(n),
        "roll": np.zeros(n), "pitch": np.zeros(n), "yaw": np.zeros(n),
    }

    # Normalise motor PWMs: uint16 (0–65535) → float (0–1)
    pwm_scale = 1.0 / 65535.0
    m1 = data["m1"] * pwm_scale
    m2 = data["m2"] * pwm_scale
    m3 = data["m3"] * pwm_scale
    m4 = data["m4"] * pwm_scale

    print(f"  Running open-loop replay ({n} samples) …")
    sim_t = 0.0
    sample_idx = 0

    for i in range(n):
        target_t = t_phys[i]
        pwm_cmd = np.array([m1[i], m2[i], m3[i], m4[i]])

        # Advance sim to this timestamp
        while sim_t < target_t - sim_dt * 0.5:
            sim.step(pwm_cmd, command_type='pwm')
            sim_t += sim_dt

        # Record sim state
        pos = sim.state[POS]
        vel = sim.state[VEL]
        euler = quat_to_euler(sim.state[QUAT])

        sim_out["t"][i] = sim_t
        sim_out["x"][i] = pos[0]
        sim_out["y"][i] = pos[1]
        sim_out["z"][i] = -pos[2]  # NED → altitude
        sim_out["vx"][i] = vel[0]
        sim_out["vy"][i] = vel[1]
        sim_out["vz"][i] = vel[2]
        sim_out["roll"][i] = np.degrees(euler[0])
        sim_out["pitch"][i] = np.degrees(euler[1])
        sim_out["yaw"][i] = np.degrees(euler[2])

    print(f"  ✓ Open-loop replay complete")
    return sim_out


# ═══════════════════════════════════════════════════════════════════════
# Closed-Loop Replay
# ═══════════════════════════════════════════════════════════════════════

def replay_closed_loop(data: dict, params: CrazyflieParams = None) -> dict:
    """
    Run sim PID tracking the same position setpoints as the physical flight.
    Compares closed-loop behaviour (different PIDs, same references).
    """
    if "x_ref" not in data:
        raise ValueError("CSV has no x_ref column — cannot run closed-loop mode")

    params = params or CrazyflieParams.brushless()
    sim_config = SimConfig(
        dt=0.001,
        enable_sensors=False,
        enable_ground_collision=True,
        enable_ground_effect=True,
        enable_aerodynamics=True,
        log_data=False,
    )
    sim = CrazyflieSim(params, sim_config)
    controller = CrazyflieController(params)

    init_pos = np.array([data["x"][0], data["y"][0], -data["z"][0]])
    sim.reset(position=init_pos)

    # Pre-spin
    hover_pwm = (-params.pwm_thrust_b + np.sqrt(
        params.pwm_thrust_b**2 + 4 * params.pwm_thrust_a * (params.weight / 4.0)
    )) / (2 * params.pwm_thrust_a)
    for _ in range(200):
        sim.step(np.array([hover_pwm]*4), command_type='pwm')
    sim.state[VEL] = np.zeros(3)
    sim.state[OMEGA] = np.zeros(3)
    controller.reset()
    sim.time = 0.0

    t_phys = data["t"]
    n = len(t_phys)
    sim_dt = sim_config.dt
    ctrl_dt = 1.0 / cfg.CONTROL_RATE_HZ

    sim_out = {
        "t": np.zeros(n),
        "x": np.zeros(n), "y": np.zeros(n), "z": np.zeros(n),
        "vx": np.zeros(n), "vy": np.zeros(n), "vz": np.zeros(n),
        "roll": np.zeros(n), "pitch": np.zeros(n), "yaw": np.zeros(n),
    }

    print(f"  Running closed-loop replay ({n} samples) …")
    sim_t = 0.0
    motor_pwms = np.array([hover_pwm]*4)
    steps_per_ctrl = max(1, int(ctrl_dt / sim_dt))

    for i in range(n):
        target_t = t_phys[i]
        desired_alt = data["z_ref"][i]

        while sim_t < target_t - sim_dt * 0.5:
            # Recompute control at ~100 Hz
            motor_pwms = controller.compute(
                sim.state,
                desired_roll=0.0,
                desired_pitch=0.0,
                desired_yaw_rate=0.0,
                desired_altitude=desired_alt,
                dt=ctrl_dt,
            )
            for _ in range(steps_per_ctrl):
                sim.step(motor_pwms, command_type='pwm')
                sim_t += sim_dt

        pos = sim.state[POS]
        vel = sim.state[VEL]
        euler = quat_to_euler(sim.state[QUAT])

        sim_out["t"][i] = sim_t
        sim_out["x"][i] = pos[0]
        sim_out["y"][i] = pos[1]
        sim_out["z"][i] = -pos[2]
        sim_out["vx"][i] = vel[0]
        sim_out["vy"][i] = vel[1]
        sim_out["vz"][i] = vel[2]
        sim_out["roll"][i] = np.degrees(euler[0])
        sim_out["pitch"][i] = np.degrees(euler[1])
        sim_out["yaw"][i] = np.degrees(euler[2])

    print(f"  ✓ Closed-loop replay complete")
    return sim_out


# ═══════════════════════════════════════════════════════════════════════
# Sim-Only Mode (synthetic reference for dry-run testing)
# ═══════════════════════════════════════════════════════════════════════

def generate_synthetic_flight(manoeuvre: str = "hover",
                              duration: float = 10.0) -> dict:
    """Run the sim with its own PID and produce a 'synthetic physical' CSV."""
    params = CrazyflieParams.brushless()
    sim_config = SimConfig(
        dt=0.001, enable_sensors=True, enable_ground_collision=True,
        enable_ground_effect=True, enable_aerodynamics=True, log_data=False,
    )
    sim = CrazyflieSim(params, sim_config)
    controller = CrazyflieController(params)

    traj_fn = cfg.TRAJECTORY_FNS.get(manoeuvre, cfg.TRAJECTORY_FNS["hover"])

    sim.reset(position=np.array([0.0, 0.0, -0.5]))

    # Pre-spin
    hover_pwm = (-params.pwm_thrust_b + np.sqrt(
        params.pwm_thrust_b**2 + 4 * params.pwm_thrust_a * (params.weight / 4.0)
    )) / (2 * params.pwm_thrust_a)
    for _ in range(200):
        sim.step(np.array([hover_pwm]*4), command_type='pwm')
    sim.state[VEL] = np.zeros(3)
    sim.state[OMEGA] = np.zeros(3)
    controller.reset()

    dt_ctrl = 1.0 / cfg.CONTROL_RATE_HZ
    dt_sim = sim_config.dt
    steps_per_ctrl = int(dt_ctrl / dt_sim)
    n_samples = int(duration * cfg.CONTROL_RATE_HZ)

    data = {k: np.zeros(n_samples) for k in [
        "t", "x", "y", "z", "vx", "vy", "vz",
        "roll", "pitch", "yaw",
        "m1", "m2", "m3", "m4", "vbat",
        "x_ref", "y_ref", "z_ref", "yaw_ref",
    ]}

    print(f"  Generating synthetic flight: {manoeuvre} ({duration:.0f}s) …")
    sim_t = 0.0

    for i in range(n_samples):
        ref = traj_fn(sim_t)
        motor_pwms = controller.compute(
            sim.state,
            desired_roll=0.0, desired_pitch=0.0,
            desired_yaw_rate=0.0, desired_altitude=ref[2],
            dt=dt_ctrl,
        )
        for _ in range(steps_per_ctrl):
            sim.step(motor_pwms, command_type='pwm')
            sim_t += dt_sim

        pos = sim.state[POS]
        vel = sim.state[VEL]
        euler = quat_to_euler(sim.state[QUAT])

        data["t"][i] = sim_t
        data["x"][i] = pos[0]
        data["y"][i] = pos[1]
        data["z"][i] = -pos[2]
        data["vx"][i] = vel[0]
        data["vy"][i] = vel[1]
        data["vz"][i] = vel[2]
        data["roll"][i] = np.degrees(euler[0])
        data["pitch"][i] = np.degrees(euler[1])
        data["yaw"][i] = np.degrees(euler[2])
        # Store PWMs as uint16 scale for consistency
        data["m1"][i] = motor_pwms[0] * 65535
        data["m2"][i] = motor_pwms[1] * 65535
        data["m3"][i] = motor_pwms[2] * 65535
        data["m4"][i] = motor_pwms[3] * 65535
        data["vbat"][i] = 3.7
        data["x_ref"][i] = ref[0]
        data["y_ref"][i] = ref[1]
        data["z_ref"][i] = ref[2]
        data["yaw_ref"][i] = ref[3]

    print(f"  ✓ Synthetic flight: {n_samples} samples")
    return data


# ═══════════════════════════════════════════════════════════════════════
# Main Validation Pipeline
# ═══════════════════════════════════════════════════════════════════════

def run_validation(data: dict, mode: str = "open-loop",
                   params: CrazyflieParams = None) -> dict:
    """Run validation and return results dict."""
    if mode == "open-loop":
        sim_out = replay_open_loop(data, params)
    elif mode == "closed-loop":
        sim_out = replay_closed_loop(data, params)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Compute metrics for available channels
    channels = [
        ("x", "x"), ("y", "y"), ("z", "z"),
        ("vx", "vx"), ("vy", "vy"), ("vz", "vz"),
    ]
    if "roll" in data:
        channels += [("roll", "roll"), ("pitch", "pitch"), ("yaw", "yaw")]

    all_metrics = []
    for real_key, sim_key in channels:
        if real_key in data and sim_key in sim_out:
            m = compute_metrics(data[real_key], sim_out[sim_key], label=real_key)
            all_metrics.append(m)

    print_metrics_table(all_metrics)

    return {
        "real": data,
        "sim": sim_out,
        "metrics": all_metrics,
        "mode": mode,
    }


def save_results(results: dict, output_dir: str):
    """Save comparison results to CSV."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Metrics summary
    metrics_path = out_dir / "metrics_summary.csv"
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results["metrics"][0].keys()))
        writer.writeheader()
        writer.writerows(results["metrics"])
    print(f"  ✓ Metrics saved → {metrics_path}")

    # Sim output for plotting
    sim = results["sim"]
    sim_path = out_dir / "sim_replay.csv"
    n = len(sim["t"])
    with open(sim_path, "w", newline="") as f:
        cols = list(sim.keys())
        writer = csv.writer(f)
        writer.writerow(cols)
        for i in range(n):
            writer.writerow([sim[c][i] for c in cols])
    print(f"  ✓ Sim replay saved → {sim_path}")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Validate digital twin against physical flight data."
    )
    parser.add_argument(
        "csv_path", nargs="?", default=None,
        help="Path to physical flight CSV",
    )
    parser.add_argument(
        "--mode", choices=["open-loop", "closed-loop"], default="open-loop",
        help="Validation mode (default: open-loop)",
    )
    parser.add_argument(
        "--sim-only", action="store_true",
        help="Generate synthetic data and validate (no hardware)",
    )
    parser.add_argument(
        "--manoeuvre", default="hover",
        choices=list(cfg.MANOEUVRES.keys()),
        help="Manoeuvre for sim-only mode",
    )
    parser.add_argument(
        "--imav", action="store_true",
        help="Load IMAV-format CSV (from imavpaper2 experiments)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate comparison plots after validation",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  DIGITAL TWIN VALIDATION")
    print(f"  Mode: {args.mode}")
    print(f"{'='*60}")

    # Load or generate data
    if args.sim_only:
        data = generate_synthetic_flight(args.manoeuvre, duration=10.0)
        source = f"synthetic_{args.manoeuvre}"
    elif args.csv_path:
        if args.imav:
            data = load_imav_csv(args.csv_path)
        else:
            data = load_flight_csv(args.csv_path)
        source = Path(args.csv_path).stem
    else:
        parser.error("Provide a CSV path or use --sim-only")
        return

    # Run validation
    results = run_validation(data, mode=args.mode)

    # Save results
    out_dir = args.output or str(SCRIPT_DIR / "results" / source)
    save_results(results, out_dir)

    # Optional plotting
    if args.plot:
        try:
            from plot_validation import plot_comparison
            plot_comparison(results, output_dir=out_dir)
        except ImportError:
            print("  ⚠ plot_validation.py not found — skipping plots")

    print(f"\n  ✓ Validation complete!")


if __name__ == "__main__":
    main()
