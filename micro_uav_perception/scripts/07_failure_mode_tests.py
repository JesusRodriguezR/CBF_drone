"""
07_failure_mode_tests.py
------------------------
Evaluate the updated perception model under challenging visual conditions.
"""

import sys
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
    load_grayscale_image,
)

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_IMAGES = PROCESSED_DIR / "images"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
TABLES_DIR = RESULTS_DIR / "tables"
NOTES_DIR = PROJECT_ROOT / "notes"


def load_triplet(row):
    frame_prev = load_grayscale_image(PROCESSED_IMAGES, row["frame_prev"])
    frame_t = load_grayscale_image(PROCESSED_IMAGES, row["frame_t"])
    frame_t1 = load_grayscale_image(PROCESSED_IMAGES, row["frame_t1"])
    gt = np.array([row["dx"], row["dy"], row["dyaw"]], dtype=np.float32)
    return frame_prev, frame_t, frame_t1, gt


def predict(model, frames, device):
    stacked = build_input_stack(*frames)[np.newaxis]
    tensor = torch.from_numpy(stacked).to(device)
    with torch.no_grad():
        return model(tensor).cpu().numpy()[0]


def apply_augmentation(frames, aug_type):
    frame_prev, frame_t, frame_t1 = [frame.copy() for frame in frames]
    triplet = [frame_prev, frame_t, frame_t1]

    if aug_type == "normal":
        return triplet
    if aug_type == "low_texture":
        return [cv2.GaussianBlur(frame, (21, 21), 0) for frame in triplet]
    if aug_type == "motion_blur":
        kernel_size = 15
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel[kernel_size // 2, :] = 1.0 / kernel_size
        return [cv2.filter2D(frame, -1, kernel) for frame in triplet]
    if aug_type == "brightness_dark":
        return [np.clip(frame.astype(np.int16) - 80, 0, 255).astype(np.uint8) for frame in triplet]
    if aug_type == "brightness_bright":
        return [np.clip(frame.astype(np.int16) + 80, 0, 255).astype(np.uint8) for frame in triplet]
    if aug_type == "noise":
        augmented = []
        for frame in triplet:
            noise = np.random.normal(0, 25, frame.shape).astype(np.int16)
            augmented.append(np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8))
        return augmented
    return triplet


def main():
    print("=" * 60)
    print("Failure Mode Analysis")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyMotionCNN().to(device)
    model.load_state_dict(torch.load(MODELS_DIR / MODEL_FILENAME, map_location=device, weights_only=True))
    model.eval()

    df = pd.read_csv(PROCESSED_DIR / "test.csv")
    df["speed"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2)
    df["abs_yaw"] = np.abs(df["dyaw"])

    fast_mask = df["speed"] > df["speed"].quantile(0.80)
    slow_mask = df["speed"] < df["speed"].quantile(0.20)
    high_yaw_mask = df["abs_yaw"] > df["abs_yaw"].quantile(0.80)
    forward_mask = (df["dx"].abs() > df["dy"].abs()) & (~fast_mask)

    categories = {
        "all_test": df,
        "fast_motion": df[fast_mask],
        "slow_motion": df[slow_mask],
        "high_yaw": df[high_yaw_mask],
        "forward_dominant": df[forward_mask],
    }
    augmentations = ["normal", "low_texture", "motion_blur", "brightness_dark", "brightness_bright", "noise"]

    print("\n--- Motion Category Analysis ---")
    cat_results = []
    for category_name, category_df in categories.items():
        if len(category_df) < 5:
            continue
        errors_dx, errors_dy, errors_dyaw = [], [], []
        for _, row in category_df.iterrows():
            frame_prev, frame_t, frame_t1, gt = load_triplet(row)
            pred = predict(model, (frame_prev, frame_t, frame_t1), device)
            errors_dx.append(abs(pred[0] - gt[0]))
            errors_dy.append(abs(pred[1] - gt[1]))
            errors_dyaw.append(abs(pred[2] - gt[2]))

        rmse_dx = np.sqrt(np.mean(np.square(errors_dx)))
        rmse_dy = np.sqrt(np.mean(np.square(errors_dy)))
        rmse_dyaw = np.sqrt(np.mean(np.square(errors_dyaw)))
        cat_results.append({
            "condition": category_name,
            "n_samples": len(errors_dx),
            "rmse_dx": rmse_dx,
            "rmse_dy": rmse_dy,
            "rmse_dyaw": rmse_dyaw,
            "type": "motion_category",
        })
        print(f"  {category_name:20s}: n={len(errors_dx):3d}  RMSE dx={rmse_dx:.4f} dy={rmse_dy:.4f} dyaw={rmse_dyaw:.4f}")

    print("\n--- Synthetic Augmentation Analysis ---")
    aug_results = []
    np.random.seed(42)
    for aug_type in augmentations:
        errors_dx, errors_dy, errors_dyaw = [], [], []
        for _, row in df.iterrows():
            frame_prev, frame_t, frame_t1, gt = load_triplet(row)
            augmented = apply_augmentation((frame_prev, frame_t, frame_t1), aug_type)
            pred = predict(model, augmented, device)
            errors_dx.append(abs(pred[0] - gt[0]))
            errors_dy.append(abs(pred[1] - gt[1]))
            errors_dyaw.append(abs(pred[2] - gt[2]))

        rmse_dx = np.sqrt(np.mean(np.square(errors_dx)))
        rmse_dy = np.sqrt(np.mean(np.square(errors_dy)))
        rmse_dyaw = np.sqrt(np.mean(np.square(errors_dyaw)))
        aug_results.append({
            "condition": aug_type,
            "n_samples": len(errors_dx),
            "rmse_dx": rmse_dx,
            "rmse_dy": rmse_dy,
            "rmse_dyaw": rmse_dyaw,
            "type": "augmentation",
        })
        print(f"  {aug_type:20s}: n={len(errors_dx):3d}  RMSE dx={rmse_dx:.4f} dy={rmse_dy:.4f} dyaw={rmse_dyaw:.4f}")

    all_results = cat_results + aug_results
    results_df = pd.DataFrame(all_results)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(TABLES_DIR / "failure_mode_summary.csv", index=False)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    aug_df = results_df[results_df["type"] == "augmentation"]
    x = range(len(aug_df))
    labels = aug_df["condition"].tolist()

    for axis_idx, (column, title) in enumerate([
        ("rmse_dx", "RMSE dx (m)"),
        ("rmse_dy", "RMSE dy (m)"),
        ("rmse_dyaw", "RMSE dyaw (rad)"),
    ]):
        ax = axes[axis_idx]
        values = aug_df[column].tolist()
        colors = ["#2ecc71" if value == min(values) else "#e74c3c" if value == max(values) else "#3498db"
                  for value in values]
        ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(y=values[0], color="gray", linestyle="--", alpha=0.5, label="Baseline")
        ax.legend(fontsize=8)

    plt.suptitle("Failure Mode Analysis: Augmentation Impact", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "failure_mode_errors.png", dpi=150, bbox_inches="tight")
    plt.close()

    cat_df_plot = results_df[results_df["type"] == "motion_category"]
    if len(cat_df_plot) > 1:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        x = range(len(cat_df_plot))
        labels = cat_df_plot["condition"].tolist()
        for axis_idx, (column, title) in enumerate([
            ("rmse_dx", "RMSE dx (m)"),
            ("rmse_dy", "RMSE dy (m)"),
            ("rmse_dyaw", "RMSE dyaw (rad)"),
        ]):
            ax = axes[axis_idx]
            values = cat_df_plot[column].tolist()
            colors = ["#2ecc71" if value == min(values) else "#e74c3c" if value == max(values) else "#3498db"
                      for value in values]
            ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5)
            ax.set_xticks(list(x))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.3)
        plt.suptitle("Failure Mode Analysis: Motion Categories", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "failure_mode_categories.png", dpi=150, bbox_inches="tight")
        plt.close()

    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    with open(NOTES_DIR / "failure_modes.md", "w", encoding="utf-8") as file:
        file.write("# Failure Mode Analysis\n\n")
        file.write("## Motion Category Results\n\n")
        file.write("| Condition | N | RMSE dx | RMSE dy | RMSE dyaw |\n")
        file.write("|-----------|---|---------|---------|----------|\n")
        for result in cat_results:
            file.write(
                f"| {result['condition']} | {result['n_samples']} | "
                f"{result['rmse_dx']:.4f} | {result['rmse_dy']:.4f} | {result['rmse_dyaw']:.4f} |\n"
            )

        file.write("\n## Synthetic Augmentation Results\n\n")
        file.write("| Condition | N | RMSE dx | RMSE dy | RMSE dyaw | Impact |\n")
        file.write("|-----------|---|---------|---------|----------|--------|\n")
        baseline_dx = aug_results[0]["rmse_dx"]
        baseline_dy = aug_results[0]["rmse_dy"]
        for result in aug_results:
            pct_dx = ((result["rmse_dx"] / baseline_dx) - 1.0) * 100.0
            pct_dy = ((result["rmse_dy"] / baseline_dy) - 1.0) * 100.0
            avg_pct = (pct_dx + pct_dy) / 2.0
            if avg_pct < 5:
                impact = "minimal"
            elif avg_pct < 25:
                impact = "moderate"
            elif avg_pct < 50:
                impact = "significant"
            else:
                impact = "severe"
            file.write(
                f"| {result['condition']} | {result['n_samples']} | "
                f"{result['rmse_dx']:.4f} | {result['rmse_dy']:.4f} | {result['rmse_dyaw']:.4f} | "
                f"{impact} ({avg_pct:+.0f}%) |\n"
            )

        file.write("\n## Key Observations\n\n")
        file.write("- Fast motion increases local-motion estimation error due to larger inter-frame displacement.\n")
        file.write("- High yaw still degrades translation estimates, but the local-frame formulation is physically cleaner.\n")
        file.write("- Low texture and motion blur reduce usable visual structure.\n")
        file.write("- Large lighting shifts remain a major generalization problem.\n")
        file.write("- Additive noise is less damaging than strong brightness changes.\n")

    print("\nResults saved:")
    print(f"  {TABLES_DIR / 'failure_mode_summary.csv'}")
    print(f"  {PLOTS_DIR / 'failure_mode_errors.png'}")
    print(f"  {NOTES_DIR / 'failure_modes.md'}")
    print("Failure mode analysis complete!")


if __name__ == "__main__":
    main()
