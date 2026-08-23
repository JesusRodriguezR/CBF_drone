"""
08_demo_package.py
------------------
Generate a compact demo package from the current clean-split experiment.
"""

from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from perception_common import (
    MODEL_FILENAME,
    TinyMotionCNN,
    build_input_stack,
    integrate_local_trajectory,
    load_grayscale_image,
)

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_IMAGES = PROCESSED_DIR / "images"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
DEMO_DIR = RESULTS_DIR / "demo"
TABLES_DIR = RESULTS_DIR / "tables"

SEQUENCE_SPLITS = {
    "train": ["indoor_forward_3", "indoor_forward_5", "indoor_forward_6"],
    "val": ["indoor_forward_7", "indoor_forward_9"],
    "test": ["indoor_forward_10"],
}


def main():
    print("=" * 60)
    print("Demo Package Generation")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    model = TinyMotionCNN().to(device)
    model.load_state_dict(torch.load(MODELS_DIR / MODEL_FILENAME, map_location=device, weights_only=True))
    model.eval()

    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    eval_metrics = pd.read_csv(TABLES_DIR / "evaluation_metrics.csv").iloc[0]
    trajectory_metrics_df = pd.read_csv(TABLES_DIR / "trajectory_metrics.csv")
    trajectory_metrics = dict(zip(trajectory_metrics_df["metric"], trajectory_metrics_df["value"]))

    print("Generating demo input frames...")
    sample_indices = np.linspace(0, len(test_df) - 1, 6, dtype=int)
    fig, axes = plt.subplots(3, 6, figsize=(18, 8))
    fig.suptitle("Sample Input Frame Triplets (Test Set)", fontsize=14, fontweight="bold")

    for column, sample_idx in enumerate(sample_indices):
        row = test_df.iloc[sample_idx]
        frame_prev = load_grayscale_image(PROCESSED_IMAGES, row["frame_prev"])
        frame_t = load_grayscale_image(PROCESSED_IMAGES, row["frame_t"])
        frame_t1 = load_grayscale_image(PROCESSED_IMAGES, row["frame_t1"])

        axes[0, column].imshow(frame_prev, cmap="gray")
        axes[0, column].set_title(f"t-1 (#{sample_idx})", fontsize=9)
        axes[0, column].axis("off")

        axes[1, column].imshow(frame_t, cmap="gray")
        axes[1, column].set_title("t", fontsize=9)
        axes[1, column].axis("off")

        axes[2, column].imshow(frame_t1, cmap="gray")
        axes[2, column].set_title("t+1", fontsize=9)
        axes[2, column].axis("off")

    axes[0, 0].set_ylabel("Frame t-1", fontsize=11)
    axes[1, 0].set_ylabel("Frame t", fontsize=11)
    axes[2, 0].set_ylabel("Frame t+1", fontsize=11)
    plt.tight_layout()
    plt.savefig(DEMO_DIR / "demo_input_frames.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Generating prediction overlay...")
    all_preds, all_gts = [], []
    for _, row in test_df.iterrows():
        frame_prev = load_grayscale_image(PROCESSED_IMAGES, row["frame_prev"])
        frame_t = load_grayscale_image(PROCESSED_IMAGES, row["frame_t"])
        frame_t1 = load_grayscale_image(PROCESSED_IMAGES, row["frame_t1"])
        stacked = build_input_stack(frame_prev, frame_t, frame_t1)[np.newaxis]
        tensor = torch.from_numpy(stacked).to(device)
        with torch.no_grad():
            pred = model(tensor).cpu().numpy()[0]
        gt = np.array([row["dx"], row["dy"], row["dyaw"]], dtype=np.float32)
        all_preds.append(pred)
        all_gts.append(gt)

    preds = np.asarray(all_preds)
    gts = np.asarray(all_gts)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    labels = ["dx (m)", "dy (m)", "dyaw (rad)"]
    colors = ["#3498db", "#e74c3c", "#2ecc71"]
    for axis_idx, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        ax.plot(gts[:, axis_idx], "k-", linewidth=1.0, alpha=0.7, label="Ground Truth")
        ax.plot(preds[:, axis_idx], "-", color=color, linewidth=1.0, alpha=0.8, label="Predicted")
        ax.set_xlabel("Frame Index")
        ax.set_ylabel(label)
        ax.set_title(f"Prediction Overlay: {label}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle("CNN Predictions vs Ground Truth (Local Frame)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(DEMO_DIR / "demo_prediction_overlay.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Generating trajectory plot...")
    gt_traj = integrate_local_trajectory(gts, use_yaw=True)
    pred_traj = integrate_local_trajectory(preds, use_yaw=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(gt_traj[:, 0], gt_traj[:, 1], "g-", linewidth=2.5, label="Ground Truth", alpha=0.8)
    ax.plot(pred_traj[:, 0], pred_traj[:, 1], "b-", linewidth=2.0, label="CNN Predicted", alpha=0.8)
    ax.plot(gt_traj[0, 0], gt_traj[0, 1], "go", markersize=12, label="Start", zorder=5)
    ax.plot(gt_traj[-1, 0], gt_traj[-1, 1], "rs", markersize=12, label="End (GT)", zorder=5)
    ax.plot(pred_traj[-1, 0], pred_traj[-1, 1], "b^", markersize=10, label="End (CNN)", zorder=5)
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_title("Trajectory Reconstruction: CNN vs Ground Truth", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(DEMO_DIR / "demo_reconstructed_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Generating error plot...")
    ate = np.linalg.norm(pred_traj[:, :2] - gt_traj[:, :2], axis=1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    ax1.plot(ate, "r-", linewidth=1.5)
    ax1.set_xlabel("Frame Index")
    ax1.set_ylabel("ATE (m)")
    ax1.set_title("Absolute Trajectory Error Over Time", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=np.mean(ate), color="gray", linestyle="--", alpha=0.7, label=f"Mean ATE = {np.mean(ate):.2f} m")
    ax1.legend()

    err_dx = np.abs(preds[:, 0] - gts[:, 0])
    err_dy = np.abs(preds[:, 1] - gts[:, 1])
    err_dyaw = np.abs(preds[:, 2] - gts[:, 2])
    ax2.plot(err_dx, alpha=0.7, label=f"|err dx| (mean={err_dx.mean():.4f})")
    ax2.plot(err_dy, alpha=0.7, label=f"|err dy| (mean={err_dy.mean():.4f})")
    ax2.plot(err_dyaw, alpha=0.7, label=f"|err dyaw| (mean={err_dyaw.mean():.4f})")
    ax2.set_xlabel("Frame Index")
    ax2.set_ylabel("Absolute Error")
    ax2.set_title("Per-Axis Prediction Error", fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(DEMO_DIR / "demo_error_plot.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Saving demo metrics...")
    rmse_dx = float(eval_metrics["RMSE_dx"])
    rmse_dy = float(eval_metrics["RMSE_dy"])
    rmse_dyaw = float(eval_metrics["RMSE_dyaw"])
    mean_ate = float(trajectory_metrics["mean_ATE"])
    final_drift = float(trajectory_metrics["final_drift"])
    gt_path = float(trajectory_metrics["gt_path_length"])
    avg_time = float(eval_metrics["inference_ms"])
    fps = float(eval_metrics["FPS"])

    metrics = {
        "metric": [
            "RMSE_dx", "RMSE_dy", "RMSE_dyaw", "Mean_ATE", "Final_Drift",
            "GT_Path_Length", "Drift_Per_Meter", "Avg_Inference_ms", "FPS",
            "Model_Params", "Test_Samples",
        ],
        "value": [
            f"{rmse_dx:.4f}", f"{rmse_dy:.4f}", f"{rmse_dyaw:.4f}",
            f"{mean_ate:.4f}", f"{final_drift:.4f}", f"{gt_path:.4f}",
            f"{final_drift / gt_path:.4f}", f"{avg_time:.2f}", f"{fps:.0f}",
            "~72K", f"{len(gts)}",
        ],
        "unit": ["m", "m", "rad", "m", "m", "m", "m/m", "ms", "fps", "-", "-"],
    }
    pd.DataFrame(metrics).to_csv(DEMO_DIR / "demo_metrics.csv", index=False)

    print("Writing demo summary...")
    fm_table = ""
    fm_csv = TABLES_DIR / "failure_mode_summary.csv"
    if fm_csv.exists():
        fm_df = pd.read_csv(fm_csv)
        aug_df = fm_df[fm_df["type"] == "augmentation"]
        fm_table = "\n## Failure Mode Summary\n\n"
        fm_table += "| Condition | RMSE dx | RMSE dy | RMSE dyaw |\n"
        fm_table += "|-----------|---------|---------|----------|\n"
        for _, row in aug_df.iterrows():
            fm_table += f"| {row['condition']} | {row['rmse_dx']:.4f} | {row['rmse_dy']:.4f} | {row['rmse_dyaw']:.4f} |\n"

    orb_summary = "| RMSE dx | N/A |\n| RMSE dy | N/A |\n| RMSE dyaw | N/A |\n| FPS | N/A |\n"
    baseline_csv = TABLES_DIR / "baseline_comparison.csv"
    if baseline_csv.exists():
        baseline_df = pd.read_csv(baseline_csv)
        orb_df = baseline_df[baseline_df["method"] == "ORB VO"]
        if not orb_df.empty:
            orb_row = orb_df.iloc[0]
            orb_summary = (
                f"| RMSE dx | {'N/A' if pd.isna(orb_row.get('RMSE_dx')) else orb_row.get('RMSE_dx')} |\n"
                f"| RMSE dy | {'N/A' if pd.isna(orb_row.get('RMSE_dy')) else orb_row.get('RMSE_dy')} |\n"
                f"| RMSE dyaw | {'N/A' if pd.isna(orb_row.get('RMSE_dyaw')) else orb_row.get('RMSE_dyaw')} |\n"
                f"| FPS | {'N/A' if pd.isna(orb_row.get('FPS')) else orb_row.get('FPS')} |\n"
            )

    split_text = (
        f"Train {len(pd.read_csv(PROCESSED_DIR / 'train.csv'))} / "
        f"Val {len(pd.read_csv(PROCESSED_DIR / 'val.csv'))} / "
        f"Test {len(test_df)}"
    )
    train_sequences = ", ".join(SEQUENCE_SPLITS["train"])
    val_sequences = ", ".join(SEQUENCE_SPLITS["val"])
    test_sequences = ", ".join(SEQUENCE_SPLITS["test"])

    with open(DEMO_DIR / "demo_summary.md", "w", encoding="utf-8") as file:
        file.write(f"""# Demo Summary

## Tiny Monocular Relative-Pose Network for Micro-UAV Perception

### Objective
Estimate local-frame relative motion (dx, dy, dyaw) from a short monocular frame context
using a compact CNN, enabling lightweight visual odometry for micro-UAVs.

### Input
- Grayscale frame triplets `[t-1, t, t+1]` resized to 96x96
- Dataset: UZH-FPV indoor-forward Snapdragon public sequences

### Output
- Local-frame relative motion predictions: dx, dy, dyaw
- Reconstructed 2D trajectory
- Performance comparison with classical VO (ORB)

### Dataset
- Source: UZH-FPV Drone Racing Dataset
- Train sequences: {train_sequences}
- Validation sequences: {val_sequences}
- Test sequence: {test_sequences}
- Split: {split_text}
- Total processed triplets: {len(pd.read_csv(PROCESSED_DIR / 'labels.csv'))}

### Results

| Metric | TinyCNN |
|--------|---------|
| RMSE dx | {rmse_dx:.4f} m |
| RMSE dy | {rmse_dy:.4f} m |
| RMSE dyaw | {rmse_dyaw:.4f} rad |
| FPS | {fps:.0f} |

### ORB Baseline Snapshot

| Metric | ORB VO |
|--------|--------|
{orb_summary}

**Trajectory Metrics:**
- Mean ATE: {mean_ate:.2f} m
- Final drift: {final_drift:.2f} m
- GT path length: {gt_path:.2f} m
- Drift rate: {final_drift / gt_path * 100:.1f}%

### Key Observations
- The local-frame label formulation is physically cleaner than the original world-frame regression target.
- The three-frame context provides short-horizon temporal cues beyond isolated image pairs.
- Strong lighting shifts remain the dominant robustness failure mode.
- Real-time inference remains feasible at {fps:.0f} FPS on RTX 4070 Laptop GPU.
{fm_table}
### Next Steps
1. Add IMU fusion to disambiguate aggressive rotations.
2. Increase sequence diversity beyond the current indoor-forward subset.
3. Replace the simple CNN with a temporal encoder once the label pipeline stabilizes.
4. Revisit geometric baselines with a cleaner camera/body-frame comparison.
5. Test transfer on Crazyflie-specific real data.
""")

    print(f"\nDemo package saved to: {DEMO_DIR}")
    print("  demo_input_frames.png")
    print("  demo_prediction_overlay.png")
    print("  demo_reconstructed_trajectory.png")
    print("  demo_error_plot.png")
    print("  demo_metrics.csv")
    print("  demo_summary.md")
    print("Demo package generation complete!")


if __name__ == "__main__":
    main()
