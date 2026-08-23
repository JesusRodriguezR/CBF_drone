"""
Shared utilities for the micro-UAV perception experiments.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

IMG_SIZE = 96
INPUT_CHANNELS = 3
NUM_OUTPUTS = 3
MODEL_FILENAME = "tiny_cnn_relative_pose.pt"


def wrap_angle(angle):
    """Wrap an angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def world_to_local_delta(dx_world, dy_world, yaw_world):
    """Rotate a planar world-frame delta into the current local/body frame."""
    cos_yaw = math.cos(yaw_world)
    sin_yaw = math.sin(yaw_world)
    dx_local = cos_yaw * dx_world + sin_yaw * dy_world
    dy_local = -sin_yaw * dx_world + cos_yaw * dy_world
    return dx_local, dy_local


def integrate_local_delta(state, delta, use_yaw=True):
    """Integrate a local-frame planar motion increment into world coordinates."""
    x, y, yaw = state
    dx_local, dy_local, dyaw = [float(value) for value in delta]

    if use_yaw:
        x += math.cos(yaw) * dx_local - math.sin(yaw) * dy_local
        y += math.sin(yaw) * dx_local + math.cos(yaw) * dy_local
        yaw = wrap_angle(yaw + dyaw)
    else:
        x += dx_local
        y += dy_local

    return np.array([x, y, yaw], dtype=np.float32)


def integrate_local_trajectory(deltas, use_yaw=True):
    """Integrate a sequence of local-frame increments."""
    state = np.zeros(3, dtype=np.float32)
    trajectory = [state.copy()]
    for delta in deltas:
        state = integrate_local_delta(state, delta, use_yaw=use_yaw)
        trajectory.append(state.copy())
    return np.asarray(trajectory)


def load_grayscale_image(images_dir, filename):
    """Load one processed grayscale frame."""
    image = cv2.imread(str(Path(images_dir) / filename), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    return image


def apply_photometric_augmentation(frames):
    """Apply the same photometric perturbation to every frame in a triplet."""
    augmented = [frame.astype(np.float32) / 255.0 for frame in frames]

    if np.random.rand() < 0.9:
        brightness = np.random.uniform(-0.12, 0.12)
        contrast = np.random.uniform(0.85, 1.15)
        gamma = np.random.uniform(0.85, 1.15)
        augmented = [np.clip(((frame ** gamma) * contrast) + brightness, 0.0, 1.0) for frame in augmented]

    if np.random.rand() < 0.35:
        sigma = np.random.uniform(0.01, 0.05)
        augmented = [np.clip(frame + np.random.normal(0.0, sigma, frame.shape).astype(np.float32), 0.0, 1.0)
                     for frame in augmented]

    if np.random.rand() < 0.30:
        kernel_size = int(np.random.choice([3, 5]))
        augmented = [cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0) for frame in augmented]

    if np.random.rand() < 0.20:
        kernel_size = int(np.random.choice([7, 11, 15]))
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        mode = int(np.random.choice([0, 1, 2]))
        if mode == 0:
            kernel[kernel_size // 2, :] = 1.0
        elif mode == 1:
            kernel[:, kernel_size // 2] = 1.0
        else:
            np.fill_diagonal(kernel, 1.0)
        kernel /= max(kernel.sum(), 1.0)
        augmented = [cv2.filter2D(frame, -1, kernel) for frame in augmented]

    return [(frame * 255.0).astype(np.uint8) for frame in augmented]


def build_input_stack(frame_prev, frame_t, frame_t1):
    """Convert three grayscale frames into a normalized input tensor."""
    frames = [frame_prev, frame_t, frame_t1]
    normalized = [frame.astype(np.float32) / 255.0 for frame in frames]
    return np.stack(normalized, axis=0)


class MotionTripletDataset(Dataset):
    """Three-frame context dataset for local-frame motion estimation."""

    def __init__(self, csv_path, images_dir, augment=False):
        self.df = pd.read_csv(csv_path)
        self.images_dir = Path(images_dir)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        frame_prev = load_grayscale_image(self.images_dir, row["frame_prev"])
        frame_t = load_grayscale_image(self.images_dir, row["frame_t"])
        frame_t1 = load_grayscale_image(self.images_dir, row["frame_t1"])

        if self.augment:
            frame_prev, frame_t, frame_t1 = apply_photometric_augmentation([frame_prev, frame_t, frame_t1])

        stacked = build_input_stack(frame_prev, frame_t, frame_t1)
        label = np.array([row["dx"], row["dy"], row["dyaw"]], dtype=np.float32)
        return torch.from_numpy(stacked), torch.from_numpy(label)


class TinyMotionCNN(nn.Module):
    """Compact CNN for short-horizon local-frame motion estimation."""

    def __init__(self, input_channels=INPUT_CHANNELS, num_outputs=NUM_OUTPUTS):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 24, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 96, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Sequential(
            nn.Linear(96, 96),
            nn.ReLU(inplace=True),
            nn.Linear(96, num_outputs),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.regressor(x)
