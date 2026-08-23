"""
04_reconstruct_trajectory.py
----------------------------
Integrate local-frame motion predictions into a global trajectory and compare
against the integrated ground-truth local deltas.
"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from perception_common import (
    MODEL_FILENAME,
    MotionTripletDataset,
    TinyMotionCNN,
    integrate_local_trajectory,
)

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_IMAGES = PROCESSED_DIR / "images"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
TABLES_DIR = RESULTS_DIR / "tables"


def compute_ate(pred_traj, gt_traj):
    pred_aligned = pred_traj[:, :2] - pred_traj[0, :2]
    gt_aligned = gt_traj[:, :2] - gt_traj[0, :2]
    n_points = min(len(pred_aligned), len(gt_aligned))
    return np.linalg.norm(pred_aligned[:n_points] - gt_aligned[:n_points], axis=1)


def main():
    print("=" * 60)
    print("Trajectory Reconstruction")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = MODELS_DIR / MODEL_FILENAME
    if not model_path.exists():
        print("ERROR: Run 02_train_tiny_cnn.py first")
        sys.exit(1)

    model = TinyMotionCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    test_csv = PROCESSED_DIR / "test.csv"
    test_dataset = MotionTripletDataset(test_csv, PROCESSED_IMAGES, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    pred_deltas, gt_deltas = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            preds = model(images)
            pred_deltas.append(preds.cpu().numpy()[0])
            gt_deltas.append(labels.numpy()[0])

    pred_deltas = np.asarray(pred_deltas)
    gt_deltas = np.asarray(gt_deltas)

    pred_traj = integrate_local_trajectory(pred_deltas, use_yaw=True)
    gt_traj = integrate_local_trajectory(gt_deltas, use_yaw=True)
    pred_traj_noyaw = integrate_local_trajectory(pred_deltas, use_yaw=False)
    gt_traj_noyaw = integrate_local_trajectory(gt_deltas, use_yaw=False)

    ate_errors = compute_ate(pred_traj, gt_traj)
    mean_ate = float(np.mean(ate_errors))
    final_drift = float(ate_errors[-1]) if len(ate_errors) > 0 else 0.0
    gt_length = float(np.sum(np.linalg.norm(np.diff(gt_traj[:, :2], axis=0), axis=1)))
    drift_per_meter = final_drift / gt_length if gt_length > 0 else 0.0

    print("\nTrajectory Metrics:")
    print(f"  Mean ATE:        {mean_ate:.4f} m")
    print(f"  Final drift:     {final_drift:.4f} m")
    print(f"  GT path length:  {gt_length:.4f} m")
    print(f"  Drift/meter:     {drift_per_meter:.4f}")
    print(f"  Trajectory pts:  {len(pred_traj)}")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame([
        {"metric": "mean_ATE", "value": f"{mean_ate:.6f}", "unit": "m"},
        {"metric": "final_drift", "value": f"{final_drift:.6f}", "unit": "m"},
        {"metric": "gt_path_length", "value": f"{gt_length:.6f}", "unit": "m"},
        {"metric": "drift_per_meter", "value": f"{drift_per_meter:.6f}", "unit": "m/m"},
        {"metric": "num_frames", "value": str(len(pred_traj)), "unit": ""},
    ])
    metrics.to_csv(TABLES_DIR / "trajectory_metrics.csv", index=False)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    ax.plot(gt_traj[:, 0], gt_traj[:, 1], "g-", linewidth=2, label="Ground Truth", alpha=0.8)
    ax.plot(pred_traj[:, 0], pred_traj[:, 1], "b-", linewidth=1.5, label="Predicted (CNN)", alpha=0.8)
    ax.plot(gt_traj[0, 0], gt_traj[0, 1], "go", markersize=10, label="Start")
    ax.plot(gt_traj[-1, 0], gt_traj[-1, 1], "rs", markersize=10, label="End (GT)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Trajectory Reconstruction (with yaw)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    ax = axes[1]
    ax.plot(gt_traj_noyaw[:, 0], gt_traj_noyaw[:, 1], "g-", linewidth=2, label="Ground Truth", alpha=0.8)
    ax.plot(pred_traj_noyaw[:, 0], pred_traj_noyaw[:, 1], "b-", linewidth=1.5, label="Predicted (CNN)", alpha=0.8)
    ax.plot(gt_traj_noyaw[0, 0], gt_traj_noyaw[0, 1], "go", markersize=10, label="Start")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Trajectory Reconstruction (no yaw rotation)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "reconstructed_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ate_errors, "r-", linewidth=1.5)
    ax.fill_between(range(len(ate_errors)), ate_errors, alpha=0.2, color="red")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Position Error (m)")
    ax.set_title("Trajectory Error Over Time")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "trajectory_error_over_time.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nPlots saved to {PLOTS_DIR}")
    print("Trajectory reconstruction complete!")


if __name__ == "__main__":
    main()
