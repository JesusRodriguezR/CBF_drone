"""
03_evaluate_model.py
--------------------
Evaluate the trained Tiny CNN on the held-out test sequence.
"""

import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from perception_common import MODEL_FILENAME, MotionTripletDataset, TinyMotionCNN

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_IMAGES = PROCESSED_DIR / "images"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
TABLES_DIR = RESULTS_DIR / "tables"


def main():
    print("=" * 60)
    print("Model Evaluation - Tiny CNN")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_path = MODELS_DIR / MODEL_FILENAME
    if not model_path.exists():
        print("ERROR: Run 02_train_tiny_cnn.py first")
        sys.exit(1)

    model = TinyMotionCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded model from {model_path}")

    test_csv = PROCESSED_DIR / "test.csv"
    if not test_csv.exists():
        print("ERROR: Run 01_prepare_dataset.py first")
        sys.exit(1)

    test_dataset = MotionTripletDataset(test_csv, PROCESSED_IMAGES, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_dataset)}")

    all_preds, all_labels, inference_times = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            t0 = time.perf_counter()
            preds = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
            inference_times.append(t1 - t0)

    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    inference_times = np.asarray(inference_times)

    errors = preds - labels
    rmse_dx = np.sqrt(np.mean(errors[:, 0] ** 2))
    rmse_dy = np.sqrt(np.mean(errors[:, 1] ** 2))
    rmse_dyaw = np.sqrt(np.mean(errors[:, 2] ** 2))
    mae_dx = np.mean(np.abs(errors[:, 0]))
    mae_dy = np.mean(np.abs(errors[:, 1]))
    mae_dyaw = np.mean(np.abs(errors[:, 2]))
    rmse_pos = np.sqrt(np.mean(errors[:, 0] ** 2 + errors[:, 1] ** 2))
    mae_pos = np.mean(np.sqrt(errors[:, 0] ** 2 + errors[:, 1] ** 2))
    avg_time = np.mean(inference_times)
    fps = 1.0 / avg_time if avg_time > 0 else 0.0

    print("\nMetrics:")
    print(f"  RMSE dx:   {rmse_dx:.6f} m")
    print(f"  RMSE dy:   {rmse_dy:.6f} m")
    print(f"  RMSE dyaw: {rmse_dyaw:.6f} rad")
    print(f"  MAE dx:    {mae_dx:.6f} m")
    print(f"  MAE dy:    {mae_dy:.6f} m")
    print(f"  MAE dyaw:  {mae_dyaw:.6f} rad")
    print(f"  RMSE pos:  {rmse_pos:.6f} m")
    print(f"  MAE pos:   {mae_pos:.6f} m")
    print(f"  Avg inference: {avg_time * 1000:.2f} ms")
    print(f"  FPS: {fps:.0f}")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame([{
        "model": "TinyCNN",
        "input": "3x96x96 grayscale triplet",
        "output": "local-frame dx dy dyaw",
        "RMSE_dx": f"{rmse_dx:.6f}",
        "RMSE_dy": f"{rmse_dy:.6f}",
        "RMSE_dyaw": f"{rmse_dyaw:.6f}",
        "MAE_dx": f"{mae_dx:.6f}",
        "MAE_dy": f"{mae_dy:.6f}",
        "MAE_dyaw": f"{mae_dyaw:.6f}",
        "RMSE_position": f"{rmse_pos:.6f}",
        "MAE_position": f"{mae_pos:.6f}",
        "inference_ms": f"{avg_time * 1000:.2f}",
        "FPS": f"{fps:.0f}",
    }])
    metrics_df.to_csv(TABLES_DIR / "evaluation_metrics.csv", index=False)
    print(f"\nMetrics saved to {TABLES_DIR / 'evaluation_metrics.csv'}")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    components = [("dx", 0), ("dy", 1), ("dyaw", 2)]
    units = ["m", "m", "rad"]

    for (name, idx), unit in zip(components, units):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(labels[:, idx], preds[:, idx], alpha=0.4, s=10, c="#2196F3")
        lims = [
            min(labels[:, idx].min(), preds[:, idx].min()),
            max(labels[:, idx].max(), preds[:, idx].max()),
        ]
        margin = (lims[1] - lims[0]) * 0.05 if lims[1] > lims[0] else 0.05
        lims = [lims[0] - margin, lims[1] + margin]
        ax.plot(lims, lims, "r--", linewidth=1, alpha=0.7, label="Perfect")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f"Ground Truth {name} ({unit})")
        ax.set_ylabel(f"Predicted {name} ({unit})")
        ax.set_title(f"Predicted vs Ground Truth: {name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"predicted_vs_groundtruth_{name}.png", dpi=150)
        plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, idx), unit in zip(axes, components, units):
        ax.hist(errors[:, idx], bins=50, alpha=0.7, color="#4CAF50", edgecolor="black", linewidth=0.5)
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel(f"Error ({unit})")
        ax.set_ylabel("Count")
        ax.set_title(f"{name} Error Distribution")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "error_distributions.png", dpi=150)
    plt.close()

    print(f"Plots saved to {PLOTS_DIR}")
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
