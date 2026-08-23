"""
02_train_tiny_cnn.py
--------------------
Train a compact CNN to estimate local-frame relative motion (dx, dy, dyaw)
from a short grayscale frame context [t-1, t, t+1].

Key changes versus the first prototype:
  - labels are local-frame, not raw world-frame differences
  - training uses photometric augmentation
  - early stopping is driven by sequence-level validation loss
  - input context spans three consecutive frames
"""

import random
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from perception_common import MODEL_FILENAME, MotionTripletDataset, TinyMotionCNN

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_IMAGES = PROCESSED_DIR / "images"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
LOGS_DIR = RESULTS_DIR / "logs"

MAX_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
YAW_WEIGHT = 0.25
EARLY_STOP_PATIENCE = 10
MIN_IMPROVEMENT = 1e-4
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def weighted_mse_loss(pred, target, yaw_weight=YAW_WEIGHT):
    weights = torch.tensor([1.0, 1.0, yaw_weight], device=pred.device)
    return ((pred - target) ** 2 * weights).mean()


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(images)
        loss = weighted_mse_loss(predictions, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            loss = weighted_mse_loss(predictions, labels)
            total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def main():
    print("=" * 60)
    print("Tiny CNN Training - Local-Frame Motion Estimation")
    print("=" * 60)

    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_csv = PROCESSED_DIR / "train.csv"
    val_csv = PROCESSED_DIR / "val.csv"
    if not train_csv.exists() or not val_csv.exists():
        print("ERROR: Run 01_prepare_dataset.py first")
        sys.exit(1)

    train_dataset = MotionTripletDataset(train_csv, PROCESSED_IMAGES, augment=True)
    val_dataset = MotionTripletDataset(val_csv, PROCESSED_IMAGES, augment=False)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = TinyMotionCNN().to(device)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    log_lines = []

    print(f"\nTraining with early stopping (patience={EARLY_STOP_PATIENCE})...")
    print(f"{'Epoch':>5} | {'Train Loss':>12} | {'Val Loss':>12} | {'LR':>10} | {'Time':>6}")
    print("-" * 60)

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        elapsed = time.time() - t0

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        log_line = f"{epoch:5d} | {train_loss:12.8f} | {val_loss:12.8f} | {lr:10.6f} | {elapsed:5.1f}s"
        print(log_line)
        log_lines.append(log_line)

        if val_loss < (best_val_loss - MIN_IMPROVEMENT):
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODELS_DIR / MODEL_FILENAME)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    with open(LOGS_DIR / "training_log.txt", "w", encoding="utf-8") as file:
        file.write("Tiny CNN Local-Frame Motion Estimation\n")
        file.write(f"Parameters: {total_params:,}\n")
        file.write(f"Device: {device}\n")
        file.write(f"Max epochs: {MAX_EPOCHS}, Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}\n")
        file.write(f"Yaw weight: {YAW_WEIGHT}\n")
        file.write(f"Best val loss: {best_val_loss:.8f}\n")
        file.write(f"Best epoch: {best_epoch}\n\n")
        file.write(f"{'Epoch':>5} | {'Train Loss':>12} | {'Val Loss':>12} | {'LR':>10} | {'Time':>6}\n")
        file.write("-" * 60 + "\n")
        for line in log_lines:
            file.write(line + "\n")

    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, "b-", linewidth=1.5, label="Train")
    ax1.plot(epochs, val_losses, "r-", linewidth=1.5, label="Val")
    ax1.axvline(best_epoch, color="gray", linestyle="--", alpha=0.7, label="Best epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (weighted MSE)")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(epochs, train_losses, "b-", linewidth=1.5, label="Train")
    ax2.semilogy(epochs, val_losses, "r-", linewidth=1.5, label="Val")
    ax2.axvline(best_epoch, color="gray", linestyle="--", alpha=0.7, label="Best epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss (log scale)")
    ax2.set_title("Training and Validation Loss (Log)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "training_loss.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, val_losses, "r-", linewidth=2)
    ax.axvline(best_epoch, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss (weighted MSE)")
    ax.set_title("Validation Loss with Early Stopping")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "validation_loss.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nBest validation loss: {best_val_loss:.8f} at epoch {best_epoch}")
    print(f"Model saved to: {MODELS_DIR / MODEL_FILENAME}")
    print(f"Plots saved to: {PLOTS_DIR}")
    print("Training complete!")


if __name__ == "__main__":
    main()
