"""
Sim vs Real — Comparison & Overlay Plots
==========================================

Loads a simulation log and a real-flight CSV, time-aligns them, and generates
overlay plots with RMSE metrics. Used for digital twin validation.

Usage:
    python compare_sim_vs_real.py --real ../exp_results/sensor_log_hover_2026-04-14_120000.csv \
                                  --sim ../../simulations/simulation_results/hover_log.npz

    python compare_sim_vs_real.py --real ../exp_results/sensor_log_hover_2026-04-14_120000.csv \
                                  --sim ../../simulations/simulation_results/hover_log.npz \
                                  --optitrack ../exp_results/optitrack_take001.csv
"""

import argparse
import csv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────

def load_real_csv(filepath: str) -> dict:
    """Load a sensor log CSV from exp_results/.

    Returns dict of numpy arrays keyed by column name.
    """
    data = {}
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        for col in columns:
            data[col] = []
        for row in reader:
            for col in columns:
                try:
                    data[col].append(float(row[col]) if row[col] != "" else np.nan)
                except ValueError:
                    data[col].append(np.nan)
    return {k: np.array(v) for k, v in data.items()}


def load_sim_log(filepath: str) -> dict:
    """Load simulation log.

    Supports .npz (from CrazyflieSim.get_log_arrays()) or .csv format.
    """
    if filepath.endswith(".npz"):
        raw = np.load(filepath, allow_pickle=True)
        return {k: raw[k] for k in raw.files}
    elif filepath.endswith(".csv"):
        return load_real_csv(filepath)
    else:
        raise ValueError(f"Unsupported sim log format: {filepath}")


def load_optitrack_csv(filepath: str) -> dict:
    """Load OptiTrack Motive CSV export.

    Expects columns: Frame, Time, X, Y, Z, Roll, Pitch, Yaw (at minimum).
    Motive CSVs can have varying headers — this handles the common export format.
    """
    data = {"time": [], "x": [], "y": [], "z": [],
            "roll": [], "pitch": [], "yaw": []}

    with open(filepath, "r") as f:
        # Skip Motive header lines (usually 6–7 lines before data)
        lines = f.readlines()

    # Find the actual data start
    header_idx = None
    for i, line in enumerate(lines):
        if "Frame" in line and "Time" in line:
            header_idx = i
            break

    if header_idx is None:
        # Try simpler CSV format
        reader = csv.DictReader(lines)
        for row in reader:
            for key in data:
                if key in row:
                    data[key].append(float(row[key]))
    else:
        headers = lines[header_idx].strip().split(",")
        for line in lines[header_idx + 1:]:
            parts = line.strip().split(",")
            if len(parts) < len(headers):
                continue
            row = dict(zip(headers, parts))
            try:
                data["time"].append(float(row.get("Time", 0)))
                data["x"].append(float(row.get("X", 0)))
                data["y"].append(float(row.get("Y", 0)))
                data["z"].append(float(row.get("Z", 0)))
                data["roll"].append(float(row.get("Roll", 0)))
                data["pitch"].append(float(row.get("Pitch", 0)))
                data["yaw"].append(float(row.get("Yaw", 0)))
            except ValueError:
                continue

    return {k: np.array(v) for k, v in data.items()}


# ─────────────────────────────────────────────────────────────────────────
# Time alignment & interpolation
# ─────────────────────────────────────────────────────────────────────────

def time_align(t_ref: np.ndarray, t_data: np.ndarray,
               signals: dict) -> dict:
    """Interpolate signals onto a reference time grid."""
    aligned = {}
    for name, values in signals.items():
        if len(values) == len(t_data):
            aligned[name] = np.interp(t_ref, t_data, values)
        else:
            aligned[name] = values  # pass through if sizes don't match
    return aligned


def compute_rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Root mean square error between two signals."""
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() == 0:
        return np.nan
    return np.sqrt(np.mean((a[mask] - b[mask]) ** 2))


def compute_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 2:
        return np.nan
    return np.corrcoef(a[mask], b[mask])[0, 1]


# ─────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────

def plot_comparison(real: dict, sim: dict, optitrack: dict = None,
                    save_dir: str = "."):
    """Generate overlay comparison plots."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    os.makedirs(save_dir, exist_ok=True)

    # Extract time vectors
    t_real = real.get("timestamp_ms", np.array([])) / 1000.0  # → seconds
    t_sim = sim.get("time", np.array([]))

    # Common time grid (use shorter duration)
    t_max = min(t_real[-1] if len(t_real) else 0,
                t_sim[-1] if len(t_sim) else 0)
    if t_max <= 0:
        print("[ERROR] No overlapping time range found.")
        return
    t_common = np.linspace(0, t_max, int(t_max * 100))  # 100 Hz grid

    print(f"  Time range: 0 – {t_max:.2f}s ({len(t_common)} points)")

    # ── Figure 1: Attitude comparison ────────────────────────────────
    fig1 = plt.figure(figsize=(14, 10))
    fig1.suptitle("Digital Twin Validation — Attitude Comparison",
                  fontsize=14, fontweight="bold")
    gs = GridSpec(3, 2, figure=fig1, hspace=0.4, wspace=0.3)

    labels = ["Roll", "Pitch", "Yaw"]
    real_keys = ["stab_roll", "stab_pitch", "stab_yaw"]

    for i, (label, rkey) in enumerate(zip(labels, real_keys)):
        ax = fig1.add_subplot(gs[i, 0])

        # Real data
        if rkey in real:
            r_interp = np.interp(t_common, t_real, real[rkey])
            ax.plot(t_common, r_interp, "b-", linewidth=1, label="Real", alpha=0.8)

        # Sim data (euler_angles is Nx3 array: [roll, pitch, yaw] in rad)
        if "euler_angles" in sim:
            euler = sim["euler_angles"]
            if euler.ndim == 2 and euler.shape[1] >= 3:
                s_deg = np.degrees(euler[:, i])
                s_interp = np.interp(t_common, t_sim, s_deg)
                ax.plot(t_common, s_interp, "r--", linewidth=1,
                        label="Sim", alpha=0.8)
                rmse = compute_rmse(r_interp, s_interp) if rkey in real else np.nan
                ax.set_title(f"{label}  (RMSE: {rmse:.2f}°)")
            else:
                ax.set_title(label)
        else:
            ax.set_title(label)

        ax.set_ylabel("Angle [°]")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if i == 2:
            ax.set_xlabel("Time [s]")

    # ── Position comparison (right column) ───────────────────────────
    pos_labels = ["X", "Y", "Z (altitude)"]
    real_pos_keys = ["stateEst_x", "stateEst_y", "stateEst_z"]

    for i, (label, rkey) in enumerate(zip(pos_labels, real_pos_keys)):
        ax = fig1.add_subplot(gs[i, 1])

        if rkey in real:
            r_interp = np.interp(t_common, t_real, real[rkey])
            ax.plot(t_common, r_interp, "b-", linewidth=1, label="Real", alpha=0.8)

        if "position" in sim:
            pos = sim["position"]
            if pos.ndim == 2 and pos.shape[1] >= 3:
                s_data = pos[:, i]
                if i == 2:  # NED z → altitude
                    s_data = -s_data
                s_interp = np.interp(t_common, t_sim, s_data)
                ax.plot(t_common, s_interp, "r--", linewidth=1,
                        label="Sim", alpha=0.8)
                rmse = compute_rmse(r_interp, s_interp) if rkey in real else np.nan
                ax.set_title(f"{label}  (RMSE: {rmse:.4f} m)")

        # OptiTrack ground truth overlay
        if optitrack is not None:
            ot_keys = ["x", "y", "z"]
            if ot_keys[i] in optitrack and "time" in optitrack:
                ot_interp = np.interp(t_common, optitrack["time"],
                                      optitrack[ot_keys[i]])
                ax.plot(t_common, ot_interp, "g:", linewidth=1.5,
                        label="OptiTrack", alpha=0.9)

        ax.set_ylabel("Position [m]")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if i == 2:
            ax.set_xlabel("Time [s]")

    path1 = os.path.join(save_dir, f"validation_attitude_pos_{timestamp}.png")
    fig1.savefig(path1, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: {path1}")

    # ── Figure 2: IMU comparison ─────────────────────────────────────
    fig2 = plt.figure(figsize=(14, 8))
    fig2.suptitle("Digital Twin Validation — IMU Comparison",
                  fontsize=14, fontweight="bold")
    gs2 = GridSpec(2, 3, figure=fig2, hspace=0.4, wspace=0.35)

    # Accelerometer
    accel_keys = ["accel_x", "accel_y", "accel_z"]
    for i, key in enumerate(accel_keys):
        ax = fig2.add_subplot(gs2[0, i])
        if key in real:
            r_interp = np.interp(t_common, t_real, real[key])
            ax.plot(t_common, r_interp, "b-", linewidth=0.5, alpha=0.6, label="Real")
        ax.set_title(f"Accel {['X','Y','Z'][i]}")
        ax.set_ylabel("m/s²")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Gyroscope
    gyro_keys = ["gyro_x", "gyro_y", "gyro_z"]
    for i, key in enumerate(gyro_keys):
        ax = fig2.add_subplot(gs2[1, i])
        if key in real:
            r_interp = np.interp(t_common, t_real, real[key])
            ax.plot(t_common, r_interp, "b-", linewidth=0.5, alpha=0.6, label="Real")
        ax.set_title(f"Gyro {['X','Y','Z'][i]}")
        ax.set_ylabel("°/s")
        ax.set_xlabel("Time [s]")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    path2 = os.path.join(save_dir, f"validation_imu_{timestamp}.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: {path2}")

    # ── Figure 3: Motor PWM comparison ───────────────────────────────
    fig3, axes3 = plt.subplots(2, 2, figsize=(12, 8))
    fig3.suptitle("Digital Twin Validation — Motor PWM Comparison",
                  fontsize=14, fontweight="bold")

    for i in range(4):
        ax = axes3[i // 2][i % 2]
        rkey = f"motor_m{i+1}"

        if rkey in real:
            r_interp = np.interp(t_common, t_real, real[rkey])
            ax.plot(t_common, r_interp, "b-", linewidth=0.8, label="Real", alpha=0.7)

        ax.set_title(f"Motor {i+1}")
        ax.set_ylabel("PWM")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if i >= 2:
            ax.set_xlabel("Time [s]")

    fig3.tight_layout()
    path3 = os.path.join(save_dir, f"validation_motors_{timestamp}.png")
    fig3.savefig(path3, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: {path3}")

    plt.close("all")

    # ── Print summary metrics ────────────────────────────────────────
    print("\n  ┌─────────────────────────────────────────────┐")
    print("  │         VALIDATION SUMMARY METRICS           │")
    print("  ├────────────────────┬───────────┬─────────────┤")
    print("  │ Signal             │ RMSE      │ Correlation │")
    print("  ├────────────────────┼───────────┼─────────────┤")

    metrics_to_check = [
        ("Roll [°]", "stab_roll", "euler_angles", 0),
        ("Pitch [°]", "stab_pitch", "euler_angles", 1),
        ("Yaw [°]", "stab_yaw", "euler_angles", 2),
    ]

    for label, rkey, skey, idx in metrics_to_check:
        if rkey in real and skey in sim:
            r_interp = np.interp(t_common, t_real, real[rkey])
            s_data = np.degrees(sim[skey][:, idx]) if sim[skey].ndim == 2 else sim[skey]
            s_interp = np.interp(t_common, t_sim, s_data)
            rmse = compute_rmse(r_interp, s_interp)
            corr = compute_correlation(r_interp, s_interp)
            print(f"  │ {label:<18} │ {rmse:9.4f} │ {corr:11.4f} │")
        else:
            print(f"  │ {label:<18} │       N/A │         N/A │")

    print("  └────────────────────┴───────────┴─────────────┘")


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare simulation and real-flight data for digital twin validation",
    )
    parser.add_argument("--real", required=True,
                        help="Path to real-flight CSV (from exp_results/)")
    parser.add_argument("--sim", required=True,
                        help="Path to simulation log (.npz or .csv)")
    parser.add_argument("--optitrack", default=None,
                        help="Optional: Path to OptiTrack Motive CSV export")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for plots (default: same as --real)")

    args = parser.parse_args()

    save_dir = args.output_dir or os.path.dirname(args.real)

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   DIGITAL TWIN VALIDATION — SIM vs REAL COMPARISON  ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()
    print(f"  Real data:  {args.real}")
    print(f"  Sim data:   {args.sim}")
    if args.optitrack:
        print(f"  OptiTrack:  {args.optitrack}")
    print(f"  Output dir: {save_dir}")
    print()

    # Load data
    print("  Loading data...")
    real = load_real_csv(args.real)
    sim = load_sim_log(args.sim)
    optitrack = load_optitrack_csv(args.optitrack) if args.optitrack else None

    print(f"    Real: {len(next(iter(real.values())))} samples, "
          f"{len(real)} columns")
    print(f"    Sim:  {len(next(iter(sim.values())))} samples, "
          f"{len(sim)} columns")
    if optitrack:
        print(f"    OptiTrack: {len(next(iter(optitrack.values())))} samples")
    print()

    # Plot
    print("  Generating comparison plots...")
    plot_comparison(real, sim, optitrack, save_dir)

    print("\n  ✓ Validation analysis complete.")


if __name__ == "__main__":
    main()
