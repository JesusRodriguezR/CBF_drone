"""
Phase 1 — IMU Noise Characterization
=======================================

Analyses a static sensor capture CSV to extract noise parameters and compare
them against the digital twin's sensor_model.py values.

Produces:
  1. Time-series plots of raw accel/gyro (visual sanity check)
  2. Power Spectral Density (PSD) plots
  3. Allan Deviation plots
  4. Summary table: measured vs sim parameters

Usage:
    python analyse_imu_noise.py --csv ../../exp_results/sensor_log_static_2026-04-14_180000.csv

    # Or with a dry-run CSV for testing the pipeline:
    python analyse_imu_noise.py --csv ../../exp_results/sensor_log_static_XXXX_DRYRUN.csv
"""

import argparse
import csv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime

# Add simulations path for importing sim parameters
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "simulations"))
sys.path.insert(0, SIM_DIR)

try:
    from crazyflie_params import CrazyflieParams
    SIM_PARAMS_AVAILABLE = True
except ImportError:
    SIM_PARAMS_AVAILABLE = False
    print("[WARN] Could not import CrazyflieParams from simulations/.")


# ─────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────

def load_csv(filepath: str) -> dict:
    """Load sensor log CSV into dict of numpy arrays."""
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


# ─────────────────────────────────────────────────────────────────────────
# Noise analysis functions
# ─────────────────────────────────────────────────────────────────────────

def compute_noise_density(signal: np.ndarray, sample_rate: float) -> float:
    """Estimate white noise density [unit/√Hz] from stationary signal.

    For white noise: noise_density = std(signal) / √(sample_rate)
    """
    std = np.nanstd(signal)
    return std / np.sqrt(sample_rate)


def compute_psd(signal: np.ndarray, sample_rate: float):
    """Compute one-sided Power Spectral Density using Welch's method.

    Returns (frequencies, psd) arrays.
    """
    from scipy.signal import welch
    # Use a reasonable segment length (~1 second)
    nperseg = min(len(signal), int(sample_rate))
    freqs, psd = welch(signal, fs=sample_rate, nperseg=nperseg,
                       detrend="linear", scaling="density")
    return freqs, psd


def compute_allan_deviation(signal: np.ndarray, sample_rate: float,
                            n_taus: int = 50):
    """Compute overlapping Allan deviation.

    Parameters
    ----------
    signal : 1D array
        Stationary time series (rate data, e.g. gyro °/s or accel m/s²).
    sample_rate : float
        Sampling rate [Hz].
    n_taus : int
        Number of cluster sizes to evaluate (log-spaced).

    Returns
    -------
    taus : ndarray
        Averaging times [s].
    adev : ndarray
        Allan deviation at each tau.
    """
    dt = 1.0 / sample_rate
    N = len(signal)

    # Log-spaced cluster sizes from 1 sample to N/4
    max_cluster = max(2, N // 4)
    cluster_sizes = np.unique(
        np.logspace(0, np.log10(max_cluster), n_taus).astype(int)
    )

    taus = []
    adevs = []

    for m in cluster_sizes:
        tau = m * dt
        # Overlapping Allan variance
        n_clusters = N - 2 * m
        if n_clusters < 1:
            continue

        # Compute cluster averages
        cumsum = np.cumsum(signal)
        cumsum = np.insert(cumsum, 0, 0)

        # y_bar[i] = mean of signal[i:i+m]
        cluster_avgs = (cumsum[m:] - cumsum[:-m]) / m

        # Allan variance = 0.5 * mean((y_bar[i+m] - y_bar[i])^2)
        diffs = cluster_avgs[m:] - cluster_avgs[:-m]
        if len(diffs) < 1:
            continue

        avar = 0.5 * np.mean(diffs ** 2)
        adev = np.sqrt(avar)

        taus.append(tau)
        adevs.append(adev)

    return np.array(taus), np.array(adevs)


def extract_bias_drift(signal: np.ndarray, sample_rate: float,
                       window_s: float = 5.0) -> tuple:
    """Estimate bias and bias drift from sliding window mean.

    Returns (mean_bias, bias_drift_rate).
    """
    window = int(window_s * sample_rate)
    if window < 1 or window >= len(signal):
        return np.nanmean(signal), 0.0

    # Sliding window mean
    n_windows = len(signal) // window
    if n_windows < 2:
        return np.nanmean(signal), 0.0

    window_means = np.array([
        np.mean(signal[i * window: (i + 1) * window])
        for i in range(n_windows)
    ])

    mean_bias = np.mean(window_means)
    # Bias drift = std of the window means (approximation)
    bias_drift = np.std(window_means)

    return mean_bias, bias_drift


# ─────────────────────────────────────────────────────────────────────────
# Reference sim parameters
# ─────────────────────────────────────────────────────────────────────────

def get_sim_reference() -> dict:
    """Get the expected noise parameters from the simulation model."""
    if not SIM_PARAMS_AVAILABLE:
        return {}

    p = CrazyflieParams()
    return {
        "accel_noise_density": {
            "value": p.accel_noise_density,
            "unit": "m/s²/√Hz",
            "description": "Accelerometer noise density",
        },
        "gyro_noise_density": {
            "value": p.gyro_noise_density,
            "unit": "rad/s/√Hz",
            "description": "Gyroscope noise density",
        },
        "accel_bias_instability": {
            "value": p.accel_bias_instability,
            "unit": "m/s²",
            "description": "Accelerometer bias instability",
        },
        "gyro_bias_instability": {
            "value": p.gyro_bias_instability,
            "unit": "rad/s",
            "description": "Gyroscope bias instability",
        },
        "imu_sample_rate": {
            "value": p.imu_sample_rate,
            "unit": "Hz",
            "description": "IMU sample rate",
        },
    }


# ─────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────

def run_analysis(data: dict, sample_rate: float, save_dir: str):
    """Run full IMU noise analysis and generate all plots."""

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    os.makedirs(save_dir, exist_ok=True)
    sim_ref = get_sim_reference()

    # Time vector
    t = data.get("timestamp_ms", np.array([])) / 1000.0  # → seconds

    accel_keys = ["accel_x", "accel_y", "accel_z"]
    gyro_keys = ["gyro_x", "gyro_y", "gyro_z"]
    axis_labels = ["X", "Y", "Z"]

    # ── Figure 1: Raw time series ────────────────────────────────────
    fig1, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig1.suptitle("Phase 1 — Static IMU Raw Data", fontsize=14, fontweight="bold")

    for i, (key, label) in enumerate(zip(accel_keys, axis_labels)):
        ax = axes[0, i]
        if key in data:
            signal = data[key]
            ax.plot(t[:len(signal)], signal, "b-", linewidth=0.3, alpha=0.7)
            ax.axhline(np.mean(signal), color="r", linestyle="--",
                       linewidth=1, label=f"mean={np.mean(signal):.4f}")
            ax.set_title(f"Accel {label}")
            ax.set_ylabel("m/s²")
            ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for i, (key, label) in enumerate(zip(gyro_keys, axis_labels)):
        ax = axes[1, i]
        if key in data:
            signal = data[key]
            ax.plot(t[:len(signal)], signal, "g-", linewidth=0.3, alpha=0.7)
            ax.axhline(np.mean(signal), color="r", linestyle="--",
                       linewidth=1, label=f"mean={np.mean(signal):.4f}")
            ax.set_title(f"Gyro {label}")
            ax.set_ylabel("°/s")
            ax.set_xlabel("Time [s]")
            ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig1.tight_layout()
    path1 = os.path.join(save_dir, f"phase1_raw_imu_{timestamp}.png")
    fig1.savefig(path1, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: {path1}")

    # ── Figure 2: Power Spectral Density ─────────────────────────────
    try:
        from scipy.signal import welch
        has_scipy = True
    except ImportError:
        has_scipy = False
        print("  [WARN] scipy not available — skipping PSD plots.")

    if has_scipy:
        fig2, axes = plt.subplots(2, 3, figsize=(16, 8))
        fig2.suptitle("Phase 1 — Power Spectral Density", fontsize=14,
                      fontweight="bold")

        for i, (key, label) in enumerate(zip(accel_keys, axis_labels)):
            ax = axes[0, i]
            if key in data:
                signal = data[key] - np.nanmean(data[key])  # remove DC
                freqs, psd = compute_psd(signal, sample_rate)
                ax.loglog(freqs[1:], psd[1:], "b-", linewidth=0.8)
                ax.set_title(f"Accel {label} PSD")
                ax.set_ylabel("(m/s²)²/Hz")

                # Overlay expected white noise floor from sim
                if sim_ref and "accel_noise_density" in sim_ref:
                    nd = sim_ref["accel_noise_density"]["value"]
                    ax.axhline(nd**2, color="r", linestyle="--", linewidth=1,
                               alpha=0.7, label=f"Sim floor={nd**2:.2e}")
                    ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3, which="both")

        for i, (key, label) in enumerate(zip(gyro_keys, axis_labels)):
            ax = axes[1, i]
            if key in data:
                # CF gyro is in °/s, convert to rad/s for comparison
                signal = np.deg2rad(data[key] - np.nanmean(data[key]))
                freqs, psd = compute_psd(signal, sample_rate)
                ax.loglog(freqs[1:], psd[1:], "g-", linewidth=0.8)
                ax.set_title(f"Gyro {label} PSD")
                ax.set_ylabel("(rad/s)²/Hz")
                ax.set_xlabel("Freq [Hz]")

                if sim_ref and "gyro_noise_density" in sim_ref:
                    nd = sim_ref["gyro_noise_density"]["value"]
                    ax.axhline(nd**2, color="r", linestyle="--", linewidth=1,
                               alpha=0.7, label=f"Sim floor={nd**2:.2e}")
                    ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3, which="both")

        fig2.tight_layout()
        path2 = os.path.join(save_dir, f"phase1_psd_{timestamp}.png")
        fig2.savefig(path2, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved: {path2}")

    # ── Figure 3: Allan Deviation ─────────────────────────────────────
    fig3, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig3.suptitle("Phase 1 — Allan Deviation", fontsize=14, fontweight="bold")

    # Accelerometer Allan deviation (all 3 axes)
    ax = axes[0]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, (key, label, color) in enumerate(zip(accel_keys, axis_labels, colors)):
        if key in data:
            signal = data[key]
            taus, adev = compute_allan_deviation(signal, sample_rate)
            ax.loglog(taus, adev, color=color, linewidth=1.5, label=f"Accel {label}")
    ax.set_xlabel("Averaging time τ [s]")
    ax.set_ylabel("Allan deviation [m/s²]")
    ax.set_title("Accelerometer Allan Deviation")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    # Gyroscope Allan deviation (all 3 axes)
    ax = axes[1]
    for i, (key, label, color) in enumerate(zip(gyro_keys, axis_labels, colors)):
        if key in data:
            signal = np.deg2rad(data[key])  # convert to rad/s
            taus, adev = compute_allan_deviation(signal, sample_rate)
            ax.loglog(taus, adev, color=color, linewidth=1.5, label=f"Gyro {label}")
    ax.set_xlabel("Averaging time τ [s]")
    ax.set_ylabel("Allan deviation [rad/s]")
    ax.set_title("Gyroscope Allan Deviation")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    fig3.tight_layout()
    path3 = os.path.join(save_dir, f"phase1_allan_dev_{timestamp}.png")
    fig3.savefig(path3, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: {path3}")

    plt.close("all")

    # ── Summary table ─────────────────────────────────────────────────
    print()
    print("  ┌──────────────────────────────────────────────────────────────────────────────┐")
    print("  │                    PHASE 1 — IMU NOISE CHARACTERIZATION                      │")
    print("  ├─────────────────────────────┬─────────────────┬────────────────┬──────────────┤")
    print("  │ Parameter                   │ Measured        │ Sim Value      │ Match        │")
    print("  ├─────────────────────────────┼─────────────────┼────────────────┼──────────────┤")

    results = {}

    # Accelerometer noise density
    accel_nds = []
    for key in accel_keys:
        if key in data:
            nd = compute_noise_density(data[key], sample_rate)
            accel_nds.append(nd)
    if accel_nds:
        measured_and = np.mean(accel_nds)
        results["accel_noise_density"] = measured_and
        sim_val = sim_ref.get("accel_noise_density", {}).get("value", None)
        if sim_val is not None:
            ratio = measured_and / sim_val
            match_str = "✓" if 0.5 < ratio < 2.0 else "✗ MISMATCH"
            print(f"  │ Accel noise density        │ {measured_and:.6f} m/s²/√Hz │ "
                  f"{sim_val:.6f}      │ {ratio:.1f}x {match_str:>5} │")
        else:
            print(f"  │ Accel noise density        │ {measured_and:.6f} m/s²/√Hz │ "
                  f"     N/A        │              │")

    # Gyroscope noise density (CF reports °/s, sim uses rad/s)
    gyro_nds = []
    for key in gyro_keys:
        if key in data:
            signal_rad = np.deg2rad(data[key])  # convert to rad/s
            nd = compute_noise_density(signal_rad, sample_rate)
            gyro_nds.append(nd)
    if gyro_nds:
        measured_gnd = np.mean(gyro_nds)
        results["gyro_noise_density"] = measured_gnd
        sim_val = sim_ref.get("gyro_noise_density", {}).get("value", None)
        if sim_val is not None:
            ratio = measured_gnd / sim_val
            match_str = "✓" if 0.5 < ratio < 2.0 else "✗ MISMATCH"
            print(f"  │ Gyro noise density         │ {measured_gnd:.6f} rad/s/√Hz│ "
                  f"{sim_val:.6f}      │ {ratio:.1f}x {match_str:>5} │")
        else:
            print(f"  │ Gyro noise density         │ {measured_gnd:.6f} rad/s/√Hz│ "
                  f"     N/A        │              │")

    # Accelerometer bias
    accel_biases = []
    for key in accel_keys:
        if key in data:
            mean_bias, drift = extract_bias_drift(data[key], sample_rate)
            accel_biases.append(abs(mean_bias))
    if accel_biases:
        # For Z axis, expected bias ≈ 9.81 (gravity), so skip it for bias check
        measured_bias_xy = np.mean(accel_biases[:2]) if len(accel_biases) >= 2 else accel_biases[0]
        results["accel_mean_bias_xy"] = measured_bias_xy
        sim_val = sim_ref.get("accel_bias_instability", {}).get("value", None)
        if sim_val is not None:
            ratio = measured_bias_xy / sim_val if sim_val > 0 else 0
            match_str = "✓" if ratio < 2.0 else "⚠ HIGH"
            print(f"  │ Accel bias (XY mean)       │ {measured_bias_xy:.6f} m/s²    │ "
                  f"{sim_val:.6f}      │ {ratio:.1f}x {match_str:>5} │")

    # Gyroscope bias
    gyro_biases = []
    for key in gyro_keys:
        if key in data:
            signal_rad = np.deg2rad(data[key])
            mean_bias, drift = extract_bias_drift(signal_rad, sample_rate)
            gyro_biases.append(abs(mean_bias))
    if gyro_biases:
        measured_gbias = np.mean(gyro_biases)
        results["gyro_mean_bias"] = measured_gbias
        sim_val = sim_ref.get("gyro_bias_instability", {}).get("value", None)
        if sim_val is not None:
            ratio = measured_gbias / sim_val if sim_val > 0 else 0
            match_str = "✓" if ratio < 2.0 else "⚠ HIGH"
            print(f"  │ Gyro bias (mean)           │ {measured_gbias:.6f} rad/s   │ "
                  f"{sim_val:.6f}      │ {ratio:.1f}x {match_str:>5} │")

    print("  └─────────────────────────────┴─────────────────┴────────────────┴──────────────┘")

    # ── Verdict ───────────────────────────────────────────────────────
    print()
    if sim_ref:
        all_ok = True
        for param_name in ["accel_noise_density", "gyro_noise_density"]:
            if param_name in results and param_name in sim_ref:
                ratio = results[param_name] / sim_ref[param_name]["value"]
                if not (0.5 < ratio < 2.0):
                    all_ok = False

        if all_ok:
            print("  ✅ PASS — Measured noise parameters are within acceptable range of sim values.")
            print("           No parameter updates needed.")
        else:
            print("  ⚠️  MISMATCH — Some noise parameters differ significantly from sim values.")
            print("           Consider updating crazyflie_params.py with the measured values.")
            print("           Re-run Phase 1 to confirm after updating.")

    return results


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: IMU noise characterization from static sensor capture",
    )
    parser.add_argument("--csv", required=True,
                        help="Path to static capture CSV (from exp_sensor_logger)")
    parser.add_argument("--sample-rate", type=float, default=100.0,
                        help="Logging sample rate in Hz (default: 100)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for plots (default: same dir as CSV)")

    args = parser.parse_args()
    save_dir = args.output_dir or os.path.dirname(args.csv) or "."

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║    PHASE 1 — IMU NOISE CHARACTERIZATION             ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()
    print(f"  Input CSV:    {args.csv}")
    print(f"  Sample rate:  {args.sample_rate} Hz")
    print(f"  Output dir:   {save_dir}")
    print()

    # Load data
    print("  Loading data...")
    data = load_csv(args.csv)
    n_samples = len(next(iter(data.values())))
    duration = n_samples / args.sample_rate
    print(f"    {n_samples} samples, {duration:.1f}s duration")
    print(f"    Columns: {list(data.keys())}")
    print()

    # Run analysis
    print("  Running noise analysis...")
    results = run_analysis(data, args.sample_rate, save_dir)
    print()
    print("  ✓ Phase 1 analysis complete.")


if __name__ == "__main__":
    main()
