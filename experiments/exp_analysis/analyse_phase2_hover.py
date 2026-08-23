"""
Phase 2 — Hover Dynamics Comparison: Real vs Digital Twin
===========================================================

Compares real CF hover flight (with OptiTrack ground truth) against the
digital twin simulation across:

  1. Altitude stability — position hold RMSE and distribution
  2. IMU noise in-flight — accel/gyro noise density with motors running
  3. Attitude angles — roll/pitch distributions during hover
  4. Motor PWM — real vs simulated hover thrust
  5. OptiTrack vs onboard state estimate — position agreement

Inputs:
    - Real flight CSV:  sensor_optitrack_hover_*.csv
    - Sim params:       simulations/crazyflie_params.py

Usage:
    python analyse_phase2_hover.py
    python analyse_phase2_hover.py --real-csv path/to/real_data.csv
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add simulations dir for sim params
SIM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "simulations")
sys.path.insert(0, SIM_DIR)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exp_results")
PLOT_DIR = os.path.join(RESULTS_DIR, "phase2_plots")

# ═════════════════════════════════════════════════════════════════════════
# Data loading
# ═════════════════════════════════════════════════════════════════════════

def load_real_hover(csv_path: str) -> dict:
    """Load merged CF + OptiTrack CSV and return numpy arrays."""
    rows = list(csv.DictReader(open(csv_path)))
    n = len(rows)
    print(f"  Loaded {n} samples from {os.path.basename(csv_path)}")

    def col(name):
        vals = []
        for r in rows:
            v = r.get(name, "")
            vals.append(float(v) if v else np.nan)
        return np.array(vals)

    d = {
        "t_ms": col("timestamp_ms"),
        "ax": col("accel_x"), "ay": col("accel_y"), "az": col("accel_z"),
        "gx": col("gyro_x"), "gy": col("gyro_y"), "gz": col("gyro_z"),
        "roll": col("stab_roll"), "pitch": col("stab_pitch"), "yaw": col("stab_yaw"),
        "m1": col("motor_m1"), "m2": col("motor_m2"),
        "m3": col("motor_m3"), "m4": col("motor_m4"),
        "est_x": col("stateEst_x"), "est_y": col("stateEst_y"), "est_z": col("stateEst_z"),
        "ot_x": col("ot_x"), "ot_y": col("ot_y"), "ot_z": col("ot_z"),
        "ot_roll": col("ot_roll"), "ot_pitch": col("ot_pitch"), "ot_yaw": col("ot_yaw"),
        "ot_valid": col("ot_valid"),
    }

    # Convert time to seconds
    d["t_s"] = (d["t_ms"] - d["t_ms"][0]) / 1000.0

    # Compute dt
    dt = np.median(np.diff(d["t_s"]))
    d["dt"] = dt
    d["fs"] = 1.0 / dt

    return d


def generate_sim_hover(duration: float = 15.0, altitude: float = 0.5) -> dict:
    """Run a simulated hover and return data in the same format as real data."""
    try:
        from crazyflie_params import CrazyflieParams
        from crazyflie_sim import CrazyflieSim, SimConfig
    except ImportError:
        print("  [WARN] Simulation modules not found, generating synthetic sim data")
        return generate_synthetic_sim(duration, altitude)

    params = CrazyflieParams()
    config = SimConfig(dt=0.001, enable_sensors=True, log_data=True)
    sim = CrazyflieSim(params, config)

    # Start at desired altitude (NED: z = -altitude)
    sim.reset(position=np.array([0.0, 0.0, -altitude]))

    # Compute hover PWM
    a = params.pwm_thrust_a
    b = params.pwm_thrust_b
    disc = b**2 + 4 * a * (params.weight / 4.0)
    hover_pwm = (-b + np.sqrt(disc)) / (2 * a)

    n_steps = int(duration / config.dt)
    command = np.array([hover_pwm] * 4)

    print(f"  Running simulation hover: {duration}s at {altitude}m, PWM={hover_pwm:.6f}")
    for _ in range(n_steps):
        sim.step(command, command_type='pwm')

    log = sim.get_log_arrays()

    # Downsample to ~100 Hz to match real data rate
    ds = int(1.0 / config.dt / 100)
    idx = np.arange(0, len(log['time']), ds)

    d = {
        "t_s": log['time'][idx],
        "ax": log['sensors_accel'][idx, 0] / 9.81 if 'sensors_accel' in log else np.zeros(len(idx)),
        "ay": log['sensors_accel'][idx, 1] / 9.81 if 'sensors_accel' in log else np.zeros(len(idx)),
        "az": log['sensors_accel'][idx, 2] / 9.81 if 'sensors_accel' in log else np.zeros(len(idx)),
        "gx": np.degrees(log['sensors_gyro'][idx, 0]) if 'sensors_gyro' in log else np.zeros(len(idx)),
        "gy": np.degrees(log['sensors_gyro'][idx, 1]) if 'sensors_gyro' in log else np.zeros(len(idx)),
        "gz": np.degrees(log['sensors_gyro'][idx, 2]) if 'sensors_gyro' in log else np.zeros(len(idx)),
        "roll": np.degrees(log['euler_angles'][idx, 0]),
        "pitch": np.degrees(log['euler_angles'][idx, 1]),
        "yaw": np.degrees(log['euler_angles'][idx, 2]),
        "altitude": -log['position'][idx, 2],  # NED z → altitude
        "pos_x": log['position'][idx, 0],
        "pos_y": log['position'][idx, 1],
    }

    # Compute approximate motor PWM (PWM fractions × 65535)
    d["motor_avg_pwm"] = hover_pwm * 65535 * np.ones(len(idx))

    dt = np.median(np.diff(d["t_s"]))
    d["dt"] = dt
    d["fs"] = 1.0 / dt

    print(f"  Sim: {len(idx)} samples at {d['fs']:.0f} Hz")
    return d


def generate_synthetic_sim(duration, altitude):
    """Fallback synthetic sim data if sim modules unavailable."""
    fs = 100.0
    n = int(duration * fs)
    t = np.arange(n) / fs

    return {
        "t_s": t,
        "ax": np.random.normal(0, 0.005, n),
        "ay": np.random.normal(0, 0.005, n),
        "az": np.random.normal(1.0, 0.005, n),
        "gx": np.random.normal(0, 0.3, n),
        "gy": np.random.normal(0, 0.3, n),
        "gz": np.random.normal(0, 0.2, n),
        "roll": np.random.normal(0, 0.3, n),
        "pitch": np.random.normal(0, 0.3, n),
        "yaw": np.random.normal(0, 0.1, n),
        "altitude": altitude + np.random.normal(0, 0.002, n),
        "motor_avg_pwm": 34000 * np.ones(n),
        "dt": 1.0 / fs, "fs": fs,
    }


# ═════════════════════════════════════════════════════════════════════════
# Analysis functions
# ═════════════════════════════════════════════════════════════════════════

def compute_noise_density(signal, fs, axis_name=""):
    """Compute noise density from signal std and sample rate."""
    mean = np.nanmean(signal)
    std = np.nanstd(signal)
    noise_density = std / np.sqrt(fs)
    return mean, std, noise_density


def compute_psd(signal, fs):
    """Compute one-sided power spectral density."""
    signal = signal - np.nanmean(signal)
    n = len(signal)
    fft = np.fft.rfft(signal)
    psd = (2.0 / (n * fs)) * np.abs(fft)**2
    freqs = np.fft.rfftfreq(n, 1.0/fs)
    return freqs, psd


# ═════════════════════════════════════════════════════════════════════════
# Plotting
# ═════════════════════════════════════════════════════════════════════════

REAL_COLOR = '#2196F3'   # Blue
SIM_COLOR  = '#FF5722'   # Orange
OT_COLOR   = '#4CAF50'   # Green


def plot_altitude_stability(real, sim, save_dir):
    """Plot 1: Altitude hold comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Phase 2 — Altitude Stability Comparison", fontsize=14, fontweight='bold')

    # Real: OptiTrack altitude (Y is UP in OptiTrack)
    ot_y = real["ot_y"]
    valid = real["ot_valid"] == 1
    ot_y_valid = np.where(valid, ot_y, np.nan)

    t_real = real["t_s"]

    # 1a: Altitude vs time
    ax = axes[0, 0]
    ax.plot(t_real, ot_y_valid, color=REAL_COLOR, lw=0.5, alpha=0.7, label="Real (OptiTrack Y)")
    if "altitude" in sim:
        ax.plot(sim["t_s"], sim["altitude"], color=SIM_COLOR, lw=0.5, alpha=0.7, label="Sim altitude")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Altitude (m)")
    ax.set_title("Altitude vs Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 1b: Altitude deviation histogram
    ax = axes[0, 1]
    ot_mean = np.nanmean(ot_y_valid)
    ot_dev = ot_y_valid[~np.isnan(ot_y_valid)] - ot_mean
    ax.hist(ot_dev * 100, bins=50, color=REAL_COLOR, alpha=0.7, label=f"Real σ={np.std(ot_dev)*100:.2f} cm", density=True)
    if "altitude" in sim:
        sim_mean = np.mean(sim["altitude"])
        sim_dev = sim["altitude"] - sim_mean
        ax.hist(sim_dev * 100, bins=50, color=SIM_COLOR, alpha=0.5, label=f"Sim σ={np.std(sim_dev)*100:.2f} cm", density=True)
    ax.set_xlabel("Altitude deviation (cm)")
    ax.set_ylabel("Density")
    ax.set_title("Altitude Hold Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 1c: XZ lateral drift (OptiTrack)
    ax = axes[1, 0]
    ot_x = real["ot_x"]
    ot_z = real["ot_z"]
    ot_x0, ot_z0 = np.nanmean(ot_x[:20]), np.nanmean(ot_z[:20])
    ax.plot((ot_x - ot_x0) * 100, (ot_z - ot_z0) * 100, color=REAL_COLOR, lw=0.3, alpha=0.7)
    ax.scatter(0, 0, color='red', s=50, zorder=5, marker='x', label="Start")
    ax.set_xlabel("X drift (cm)")
    ax.set_ylabel("Z drift (cm)")
    ax.set_title("Lateral Position Drift (OptiTrack)")
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 1d: Statistics summary
    ax = axes[1, 1]
    ax.axis('off')
    ot_rmse = np.sqrt(np.nanmean(ot_dev**2)) * 100
    ot_range = (np.nanmax(ot_y_valid) - np.nanmin(ot_y_valid)) * 100
    lat_rmse = np.sqrt(np.nanmean((ot_x - ot_x0)**2 + (ot_z - ot_z0)**2)) * 100

    stats = [
        "── Altitude Hold ──",
        f"Mean altitude:         {ot_mean:.3f} m",
        f"Altitude std:          {np.std(ot_dev)*100:.2f} cm",
        f"Altitude RMSE:         {ot_rmse:.2f} cm",
        f"Altitude range:        {ot_range:.2f} cm",
        f"Lateral RMSE:          {lat_rmse:.2f} cm",
        "",
        f"PASS (Pos RMSE < 5cm): {'[PASS]' if ot_rmse < 5 else '[FAIL]'}",
    ]
    if "altitude" in sim:
        sim_rmse = np.sqrt(np.mean(sim_dev**2)) * 100
        stats.insert(5, f"Sim alt std:           {np.std(sim_dev)*100:.4f} cm")
        stats.insert(6, f"Sim alt RMSE:          {sim_rmse:.4f} cm")

    ax.text(0.05, 0.95, "\n".join(stats), transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.tight_layout()
    path = os.path.join(save_dir, "phase2_altitude_stability.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {path}")
    return ot_rmse


def plot_imu_noise_inflight(real, sim, save_dir):
    """Plot 2: IMU noise comparison during hover (motors on)."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("Phase 2 — IMU Noise During Hover (Motors ON)", fontsize=14, fontweight='bold')

    fs_real = real["fs"]
    fs_sim = sim["fs"]

    # Skip first 2s (takeoff transient) for noise analysis
    skip_real = int(2.0 * fs_real)
    skip_sim = int(2.0 * fs_sim)

    accel_axes = [("ax", "Accel X"), ("ay", "Accel Y"), ("az", "Accel Z")]
    gyro_axes  = [("gx", "Gyro X"), ("gy", "Gyro Y"), ("gz", "Gyro Z")]

    noise_table = []

    for i, (key, label) in enumerate(accel_axes):
        ax = axes[0, i]

        # Real
        sig_r = real[key][skip_real:]
        f_r, psd_r = compute_psd(sig_r, fs_real)
        mean_r, std_r, nd_r = compute_noise_density(sig_r, fs_real)

        # Sim
        sig_s = sim[key][skip_sim:]
        f_s, psd_s = compute_psd(sig_s, fs_sim)
        mean_s, std_s, nd_s = compute_noise_density(sig_s, fs_sim)

        ax.semilogy(f_r[1:], psd_r[1:], color=REAL_COLOR, lw=0.5, alpha=0.7,
                     label=f"Real σ={std_r:.4f}")
        ax.semilogy(f_s[1:], psd_s[1:], color=SIM_COLOR, lw=0.5, alpha=0.7,
                     label=f"Sim σ={std_s:.4f}")
        ax.set_xlabel("Freq (Hz)")
        ax.set_ylabel("PSD")
        ax.set_title(f"{label} (g)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        noise_table.append((label, "g", std_r, std_s, nd_r, nd_s))

    for i, (key, label) in enumerate(gyro_axes):
        ax = axes[1, i]

        sig_r = real[key][skip_real:]
        f_r, psd_r = compute_psd(sig_r, fs_real)
        mean_r, std_r, nd_r = compute_noise_density(sig_r, fs_real)

        sig_s = sim[key][skip_sim:]
        f_s, psd_s = compute_psd(sig_s, fs_sim)
        mean_s, std_s, nd_s = compute_noise_density(sig_s, fs_sim)

        ax.semilogy(f_r[1:], psd_r[1:], color=REAL_COLOR, lw=0.5, alpha=0.7,
                     label=f"Real σ={std_r:.2f} °/s")
        ax.semilogy(f_s[1:], psd_s[1:], color=SIM_COLOR, lw=0.5, alpha=0.7,
                     label=f"Sim σ={std_s:.2f} °/s")
        ax.set_xlabel("Freq (Hz)")
        ax.set_ylabel("PSD")
        ax.set_title(f"{label} (°/s)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        noise_table.append((label, "°/s", std_r, std_s, nd_r, nd_s))

    plt.tight_layout()
    path = os.path.join(save_dir, "phase2_imu_noise_hover.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {path}")

    return noise_table


def plot_attitude_comparison(real, sim, save_dir):
    """Plot 3: Roll/pitch/yaw during hover."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("Phase 2 — Attitude During Hover", fontsize=14, fontweight='bold')

    for i, (key_r, key_ot, label) in enumerate([
        ("roll", "ot_roll", "Roll"),
        ("pitch", "ot_pitch", "Pitch"),
        ("yaw", "ot_yaw", "Yaw"),
    ]):
        # Time series
        ax = axes[0, i]
        ax.plot(real["t_s"], real[key_r], color=REAL_COLOR, lw=0.3, alpha=0.7, label="CF onboard")
        if key_ot in real:
            ax.plot(real["t_s"], real[key_ot], color=OT_COLOR, lw=0.3, alpha=0.7, label="OptiTrack")
        ax.plot(sim["t_s"], sim[key_r.lower()], color=SIM_COLOR, lw=0.3, alpha=0.7, label="Sim")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"{label} (°)")
        ax.set_title(f"{label} vs Time")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Histogram
        ax = axes[1, i]
        r_data = real[key_r][~np.isnan(real[key_r])]
        s_data = sim[key_r.lower()]
        ax.hist(r_data, bins=50, color=REAL_COLOR, alpha=0.6, density=True,
                label=f"Real σ={np.std(r_data):.2f}°")
        ax.hist(s_data, bins=50, color=SIM_COLOR, alpha=0.5, density=True,
                label=f"Sim σ={np.std(s_data):.2f}°")
        ax.set_xlabel(f"{label} (°)")
        ax.set_ylabel("Density")
        ax.set_title(f"{label} Distribution")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "phase2_attitude.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {path}")


def plot_motors(real, sim, save_dir):
    """Plot 4: Motor PWM during hover."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Phase 2 — Motor PWM During Hover", fontsize=14, fontweight='bold')

    # Real motor time series
    ax = axes[0]
    for key, label, color in [
        ("m1", "M1", '#1f77b4'), ("m2", "M2", '#ff7f0e'),
        ("m3", "M3", '#2ca02c'), ("m4", "M4", '#d62728'),
    ]:
        ax.plot(real["t_s"], real[key], lw=0.3, alpha=0.7, color=color, label=label)

    if "motor_avg_pwm" in sim:
        ax.axhline(sim["motor_avg_pwm"][0], color=SIM_COLOR, ls='--', lw=2,
                    label=f"Sim hover PWM ({sim['motor_avg_pwm'][0]:.0f})")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Motor PWM (0–65535)")
    ax.set_title("Motor PWM vs Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Motor PWM histogram
    ax = axes[1]
    m_all = np.concatenate([real["m1"], real["m2"], real["m3"], real["m4"]])
    m_all = m_all[~np.isnan(m_all)]
    ax.hist(m_all, bins=80, color=REAL_COLOR, alpha=0.7, density=True,
            label=f"Real (mean={np.mean(m_all):.0f})")
    if "motor_avg_pwm" in sim:
        ax.axvline(sim["motor_avg_pwm"][0], color=SIM_COLOR, ls='--', lw=2,
                    label=f"Sim ({sim['motor_avg_pwm'][0]:.0f})")
    ax.set_xlabel("Motor PWM")
    ax.set_ylabel("Density")
    ax.set_title("Motor PWM Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "phase2_motors.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {path}")

    return np.mean(m_all)


def generate_summary_report(noise_table, alt_rmse, motor_mean, real, save_dir):
    """Generate a text summary of Phase 2 results."""

    lines = [
        "=" * 65,
        "  PHASE 2 — HOVER DYNAMICS VALIDATION REPORT",
        "=" * 65,
        "",
        "── 1. Altitude Stability ──",
        f"  Position RMSE (OptiTrack):  {alt_rmse:.2f} cm",
        f"  Pass criteria (< 5 cm):    {'[PASS]' if alt_rmse < 5 else '[FAIL]'}",
        "",
        "── 2. IMU Noise During Flight (Motors ON) ──",
        f"  {'Axis':<10s} {'Unit':<6s} {'Real σ':<12s} {'Sim σ':<12s} {'Ratio':<8s}",
        "  " + "-" * 50,
    ]

    for name, unit, std_r, std_s, nd_r, nd_s in noise_table:
        ratio = std_r / std_s if std_s > 0 else float('inf')
        lines.append(f"  {name:<10s} {unit:<6s} {std_r:<12.4f} {std_s:<12.4f} {ratio:<8.2f}")

    lines += [
        "",
        "── 3. Attitude ──",
        f"  Roll  std (real):  {np.nanstd(real['roll']):.2f}°",
        f"  Pitch std (real):  {np.nanstd(real['pitch']):.2f}°",
        f"  Yaw   std (real):  {np.nanstd(real['yaw']):.2f}°",
        "",
        "── 4. Motor PWM ──",
        f"  Mean hover PWM (real):  {motor_mean:.0f} / 65535",
        f"  Hover duty %:           {motor_mean/65535*100:.1f}%",
        "",
        "=" * 65,
    ]

    report = "\n".join(lines)
    print(report)

    path = os.path.join(save_dir, "phase2_report.txt")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n  ✓ Report saved: {path}")

    return report


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 2 Hover Analysis")
    parser.add_argument("--real-csv", default=None,
                        help="Path to real sensor_optitrack_hover CSV")
    parser.add_argument("--altitude", type=float, default=0.5,
                        help="Target hover altitude for sim (default: 0.5)")
    args = parser.parse_args()

    print()
    print("PHASE 2 -- HOVER DYNAMICS ANALYSIS")
    print("=" * 55)
    print()

    # Find real data
    if args.real_csv:
        csv_path = args.real_csv
    else:
        # Find most recent hover CSV
        pattern = os.path.join(RESULTS_DIR, "sensor_optitrack_hover_*.csv")
        files = sorted(glob.glob(pattern))
        files = [f for f in files if "DRYRUN" not in f]
        if not files:
            print("  [ERROR] No hover data found. Run exp_sensor_optitrack_v1.py first.")
            sys.exit(1)
        csv_path = files[-1]

    print("── Loading Real Data ──")
    real = load_real_hover(csv_path)

    print("\n── Running Simulation ──")
    duration = real["t_s"][-1] - real["t_s"][0]
    sim = generate_sim_hover(duration=duration, altitude=args.altitude)

    # Create output dir
    os.makedirs(PLOT_DIR, exist_ok=True)
    print(f"\n── Generating Plots → {PLOT_DIR} ──")

    # Generate plots
    alt_rmse = plot_altitude_stability(real, sim, PLOT_DIR)
    noise_table = plot_imu_noise_inflight(real, sim, PLOT_DIR)
    plot_attitude_comparison(real, sim, PLOT_DIR)
    motor_mean = plot_motors(real, sim, PLOT_DIR)

    # Summary report
    print()
    generate_summary_report(noise_table, alt_rmse, motor_mean, real, PLOT_DIR)


if __name__ == "__main__":
    main()
