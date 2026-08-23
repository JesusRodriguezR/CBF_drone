"""
05_visualize_predictions.py
---------------------------
Visualize model predictions on sample three-frame contexts.
"""

import sys
from pathlib import Path

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
DEMO_DIR = RESULTS_DIR / "demo"


def main():
    print("=" * 60)
    print("Visualization - Sample Predictions")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = MODELS_DIR / MODEL_FILENAME
    if not model_path.exists():
        print("ERROR: Run 02_train_tiny_cnn.py first")
        sys.exit(1)

    model = TinyMotionCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    df = pd.read_csv(PROCESSED_DIR / "test.csv")
    n_samples = min(8, len(df))
    indices = np.linspace(0, len(df) - 1, n_samples, dtype=int)

    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(n_samples, 4, figsize=(18, 3 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for row_idx, sample_idx in enumerate(indices):
        row = df.iloc[sample_idx]
        frame_prev = load_grayscale_image(PROCESSED_IMAGES, row["frame_prev"])
        frame_t = load_grayscale_image(PROCESSED_IMAGES, row["frame_t"])
        frame_t1 = load_grayscale_image(PROCESSED_IMAGES, row["frame_t1"])

        stacked = build_input_stack(frame_prev, frame_t, frame_t1)
        input_tensor = torch.from_numpy(stacked).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(input_tensor).cpu().numpy()[0]

        gt = np.array([row["dx"], row["dy"], row["dyaw"]], dtype=np.float32)

        axes[row_idx, 0].imshow(frame_prev, cmap="gray")
        axes[row_idx, 0].set_title("Frame t-1")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(frame_t, cmap="gray")
        axes[row_idx, 1].set_title("Frame t")
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(frame_t1, cmap="gray")
        axes[row_idx, 2].set_title("Frame t+1")
        axes[row_idx, 2].axis("off")

        labels = ["dx", "dy", "dyaw"]
        x_pos = np.arange(len(labels))
        width = 0.35
        axes[row_idx, 3].bar(x_pos - width / 2, gt, width, label="GT", color="#4CAF50", alpha=0.8)
        axes[row_idx, 3].bar(x_pos + width / 2, pred, width, label="Pred", color="#2196F3", alpha=0.8)
        axes[row_idx, 3].set_xticks(x_pos)
        axes[row_idx, 3].set_xticklabels(labels)
        axes[row_idx, 3].set_title(f"Local Motion (sample {sample_idx})")
        axes[row_idx, 3].legend(fontsize=8)
        axes[row_idx, 3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(DEMO_DIR / "demo_prediction_overlay.png", dpi=150, bbox_inches="tight")
    plt.savefig(PLOTS_DIR / "sample_predictions.png", dpi=150, bbox_inches="tight")
    plt.close()

    if len(df) > 0:
        first_row = df.iloc[0]
        frame_prev = load_grayscale_image(PROCESSED_IMAGES, first_row["frame_prev"])
        frame_t = load_grayscale_image(PROCESSED_IMAGES, first_row["frame_t"])
        frame_t1 = load_grayscale_image(PROCESSED_IMAGES, first_row["frame_t1"])
        combined = np.hstack([frame_prev, frame_t, frame_t1])
        import cv2
        cv2.imwrite(str(DEMO_DIR / "demo_input_frames.png"), combined)

    print(f"Visualizations saved to {DEMO_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
