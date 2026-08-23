"""
Validation Plots
=================

Publication-quality comparison figures for sim-to-real validation.

Usage
-----
    # Called automatically by validate_twin.py --plot
    # Or standalone:
    from plot_validation import plot_comparison
    plot_comparison(results, output_dir="path/to/save")
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


# ── Style ──────────────────────────────────────────────────────────────

COLORS = {
    "real": "#2196F3",      # blue
    "sim":  "#FF5722",      # orange-red
    "error": "#9C27B0",     # purple
    "ref":   "#4CAF50",     # green
    "grid":  "#E0E0E0",
}

def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "#FAFAFA",
        "axes.facecolor":   "#FAFAFA",
        "axes.grid":        True,
        "grid.alpha":       0.3,
        "grid.color":       COLORS["grid"],
        "font.family":      "serif",
        "font.size":        10,
        "axes.labelsize":   11,
        "axes.titlesize":   12,
        "legend.fontsize":  9,
        "lines.linewidth":  1.2,
    })


# ═══════════════════════════════════════════════════════════════════════
# Main Comparison Plot
# ═══════════════════════════════════════════════════════════════════════

def plot_comparison(results: dict, output_dir: str = "."):
    """Generate all comparison figures from validation results."""
    setup_style()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    real = results["real"]
    sim  = results["sim"]
    mode = results.get("mode", "open-loop")

    t_real = real["t"]
    t_sim  = sim["t"]

    # ── 1. Position time-series overlay ───────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Position Comparison ({mode})", fontweight="bold", fontsize=14)

    for ax, ch, label, unit in zip(
        axes,
        ["x", "y", "z"],
        ["X", "Y", "Z (altitude)"],
        ["m", "m", "m"],
    ):
        ax.plot(t_real, real[ch], color=COLORS["real"], label="Physical", alpha=0.9)
        ax.plot(t_sim,  sim[ch],  color=COLORS["sim"],  label="Simulation",
                linestyle="--", alpha=0.9)
        if f"{ch}_ref" in real:
            ax.plot(t_real, real[f"{ch}_ref"], color=COLORS["ref"],
                    label="Reference", linestyle=":", alpha=0.6)
        ax.set_ylabel(f"{label} [{unit}]")
        ax.legend(loc="upper right", framealpha=0.8)

    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    path = out / "position_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.name}")

    # ── 2. Velocity comparison ────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Velocity Comparison ({mode})", fontweight="bold", fontsize=14)

    for ax, ch, label in zip(axes, ["vx", "vy", "vz"],
                              ["Vx", "Vy", "Vz"]):
        ax.plot(t_real, real[ch], color=COLORS["real"], label="Physical", alpha=0.9)
        ax.plot(t_sim,  sim[ch],  color=COLORS["sim"],  label="Simulation",
                linestyle="--", alpha=0.9)
        ax.set_ylabel(f"{label} [m/s]")
        ax.legend(loc="upper right", framealpha=0.8)

    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    path = out / "velocity_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.name}")

    # ── 3. Attitude comparison (if available) ─────────────────────────
    if "roll" in real and "roll" in sim:
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f"Attitude Comparison ({mode})", fontweight="bold", fontsize=14)

        for ax, ch, label in zip(axes, ["roll", "pitch", "yaw"],
                                  ["Roll", "Pitch", "Yaw"]):
            ax.plot(t_real, real[ch], color=COLORS["real"],
                    label="Physical", alpha=0.9)
            ax.plot(t_sim,  sim[ch],  color=COLORS["sim"],
                    label="Simulation", linestyle="--", alpha=0.9)
            ax.set_ylabel(f"{label} [°]")
            ax.legend(loc="upper right", framealpha=0.8)

        axes[-1].set_xlabel("Time [s]")
        fig.tight_layout()
        path = out / "attitude_comparison.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {path.name}")

    # ── 4. Error time-series ──────────────────────────────────────────
    channels = ["x", "y", "z", "vx", "vy", "vz"]
    labels = ["X [m]", "Y [m]", "Z [m]", "Vx [m/s]", "Vy [m/s]", "Vz [m/s]"]

    fig, axes = plt.subplots(len(channels), 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Tracking Error: Sim − Real ({mode})",
                 fontweight="bold", fontsize=14)

    for ax, ch, lab in zip(axes, channels, labels):
        # Interpolate sim onto real timestamps
        n = min(len(real[ch]), len(sim[ch]))
        error = sim[ch][:n] - real[ch][:n]
        t = t_real[:n]

        ax.fill_between(t, error, alpha=0.3, color=COLORS["error"])
        ax.plot(t, error, color=COLORS["error"], linewidth=0.8)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="-")
        ax.set_ylabel(f"Δ{lab}")

        rmse = np.sqrt(np.mean(error**2))
        ax.text(0.98, 0.92, f"RMSE={rmse:.4f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.8))

    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    path = out / "error_timeseries.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.name}")

    # ── 5. Scatter: Sim vs Real ───────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(f"Sim vs Real Scatter ({mode})", fontweight="bold", fontsize=14)

    plot_channels = ["x", "y", "z", "vx", "vy", "vz"]
    plot_labels = ["X [m]", "Y [m]", "Z [m]", "Vx [m/s]", "Vy [m/s]", "Vz [m/s]"]

    for ax, ch, lab in zip(axes.flat, plot_channels, plot_labels):
        n = min(len(real[ch]), len(sim[ch]))
        ax.scatter(real[ch][:n], sim[ch][:n], s=2, alpha=0.3,
                   color=COLORS["sim"])
        lims = [min(real[ch][:n].min(), sim[ch][:n].min()),
                max(real[ch][:n].max(), sim[ch][:n].max())]
        margin = (lims[1] - lims[0]) * 0.1 or 0.01
        lims = [lims[0] - margin, lims[1] + margin]
        ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel(f"Real {lab}")
        ax.set_ylabel(f"Sim {lab}")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    path = out / "scatter_sim_vs_real.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.name}")

    # ── 6. Motor PWM comparison ───────────────────────────────────────
    if "m1" in real:
        fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
        fig.suptitle("Motor PWM Commands (Physical)", fontweight="bold", fontsize=14)

        motor_colors = ["#F44336", "#2196F3", "#4CAF50", "#FF9800"]
        for ax, ch, color, name in zip(
            axes, ["m1", "m2", "m3", "m4"], motor_colors,
            ["M1 (CW)", "M2 (CCW)", "M3 (CW)", "M4 (CCW)"]
        ):
            ax.plot(t_real, real[ch], color=color, linewidth=0.8, alpha=0.8)
            ax.set_ylabel(f"{name}\n[uint16]")
            mean_pwm = np.mean(real[ch])
            ax.axhline(mean_pwm, color=color, linestyle=":", alpha=0.4)
            ax.text(0.98, 0.85, f"mean={mean_pwm:.0f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9)

        axes[-1].set_xlabel("Time [s]")
        fig.tight_layout()
        path = out / "motor_pwm.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {path.name}")

    # ── 7. Metrics bar chart ──────────────────────────────────────────
    metrics = results["metrics"]
    if metrics:
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.suptitle(f"Fidelity Metrics ({mode})", fontweight="bold", fontsize=14)

        labels = [m["label"] for m in metrics]
        x = np.arange(len(labels))

        # RMSE
        rmses = [m["rmse"] for m in metrics]
        axes[0].bar(x, rmses, color=COLORS["error"], alpha=0.8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, rotation=45, ha="right")
        axes[0].set_title("RMSE")
        axes[0].set_ylabel("Value")

        # Correlation
        corrs = [m["corr"] if not np.isnan(m["corr"]) else 0 for m in metrics]
        colours = [COLORS["real"] if c > 0.9 else "#FFC107" if c > 0.7
                   else COLORS["sim"] for c in corrs]
        axes[1].bar(x, corrs, color=colours, alpha=0.8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=45, ha="right")
        axes[1].set_title("Correlation")
        axes[1].set_ylim(0, 1.05)
        axes[1].axhline(0.95, color="green", linestyle="--", alpha=0.4)

        # Theil-U
        theils = [m["theil_u"] for m in metrics]
        axes[2].bar(x, theils, color=COLORS["ref"], alpha=0.8)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(labels, rotation=45, ha="right")
        axes[2].set_title("Theil's U")
        axes[2].axhline(0.1, color="green", linestyle="--", alpha=0.4,
                         label="Good (<0.1)")
        axes[2].legend(fontsize=8)

        fig.tight_layout()
        path = out / "metrics_summary.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {path.name}")

    print(f"\n  ✓ All plots saved to {out}/")
