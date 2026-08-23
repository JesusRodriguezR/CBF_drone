"""
09_live_dashboard.py
--------------------
Live dashboard for the micro-UAV perception demo.

Streams the processed split sample by sample and updates:
  - current frame triplet
  - TinyCNN convolution activations
  - GT vs TinyCNN vs ORB trajectories
  - relative-motion estimates over time
  - position error and convolution-response trends

Usage examples:
  python scripts/09_live_dashboard.py
  python scripts/09_live_dashboard.py --list-sequences
  python scripts/09_live_dashboard.py --max-frames 300 --playback-speed 2.0
  python scripts/09_live_dashboard.py --no-display
"""

import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

import matplotlib
from matplotlib.widgets import Button, RadioButtons

from perception_common import (
    MODEL_FILENAME,
    TinyMotionCNN,
    build_input_stack,
    integrate_local_delta,
    load_grayscale_image,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_IMAGES = PROCESSED_DIR / "images"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LIVE_DIR = RESULTS_DIR / "live_dashboard"

IMG_SIZE = 96
FOCAL_LENGTH = 50.0
CX, CY = IMG_SIZE / 2.0, IMG_SIZE / 2.0
K = np.array([[FOCAL_LENGTH, 0, CX],
              [0, FOCAL_LENGTH, CY],
              [0, 0, 1]], dtype=np.float64)

def parse_args():
    parser = argparse.ArgumentParser(description="Live TinyCNN/ORB trajectory dashboard")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Processed split to replay")
    parser.add_argument("--sequence", default=None,
                        help="Optional sequence filter, e.g. indoor_forward_7")
    parser.add_argument("--list-sequences", action="store_true",
                        help="Print the available split/sequence combinations and exit")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Limit the number of triplets processed")
    parser.add_argument("--stride", type=int, default=1,
                        help="Use every Nth triplet")
    parser.add_argument("--playback-speed", type=float, default=1.0,
                        help="1.0 matches dataset time, 2.0 is 2x faster")
    parser.add_argument("--refresh-every", type=int, default=1,
                        help="Update the dashboard every N processed triplets")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable the interactive window and save only the final snapshot")
    parser.add_argument("--save-snapshot", action="store_true",
                        help="Save the final dashboard image even in interactive mode")
    return parser.parse_args()


def load_split_dataframe(split_name):
    split_csv = PROCESSED_DIR / f"{split_name}.csv"
    if not split_csv.exists():
        raise FileNotFoundError(f"Split file not found: {split_csv}")
    return pd.read_csv(split_csv)


def available_sequences_by_split():
    sequences = {}
    for split_name in ["train", "val", "test"]:
        df = load_split_dataframe(split_name)
        sequences[split_name] = sorted(df["sequence"].dropna().unique().tolist())
    return sequences


def print_available_sequences():
    print("Available replay targets:")
    for split_name, sequence_names in available_sequences_by_split().items():
        joined = ", ".join(sequence_names) if sequence_names else "(none)"
        print(f"  {split_name}: {joined}")


def load_replay_dataframe(split_name, sequence_name):
    df = load_split_dataframe(split_name)
    if sequence_name:
        df = df[df["sequence"] == sequence_name].reset_index(drop=True)
        if df.empty:
            available = ", ".join(sorted(load_split_dataframe(split_name)["sequence"].dropna().unique().tolist()))
            raise ValueError(
                f"Sequence '{sequence_name}' not found in split '{split_name}'. "
                f"Available: {available}"
            )
    return df


def load_model(device):
    model_path = MODELS_DIR / MODEL_FILENAME
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run 02_train_tiny_cnn.py first.")

    model = TinyMotionCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model


def infer_with_activations(model, tensor):
    activations = []
    x = tensor
    for layer in model.features:
        x = layer(x)
        if layer.__class__.__name__ == "ReLU":
            activations.append(x.detach().cpu().numpy()[0])

    pooled = model.pool(x)
    pooled = pooled.view(pooled.size(0), -1)
    pred = model.regressor(pooled)
    return pred.detach().cpu().numpy()[0], activations


def orb_relative_pose(img1, img2, intrinsics):
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return np.zeros(3, dtype=np.float32), 0, False

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda match: match.distance)

    if len(matches) < 8:
        return np.zeros(3, dtype=np.float32), len(matches), False

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    essential, _ = cv2.findEssentialMat(
        pts1, pts2, intrinsics, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    if essential is None:
        return np.zeros(3, dtype=np.float32), len(matches), False

    _, rotation, translation, _ = cv2.recoverPose(essential, pts1, pts2, intrinsics)
    dyaw = math.atan2(rotation[1, 0], rotation[0, 0])
    result = np.array([translation[0, 0], translation[1, 0], dyaw], dtype=np.float32)
    return result, len(matches), True


def scale_orb_translation(orb_delta, gt_delta):
    gt_mag = math.hypot(gt_delta[0], gt_delta[1])
    orb_mag = math.hypot(orb_delta[0], orb_delta[1])
    scaled = orb_delta.copy()
    if orb_mag > 1e-8:
        scale = gt_mag / orb_mag
        scaled[0] *= scale
        scaled[1] *= scale
    return scaled


def build_montage(activation, max_channels=16):
    channels = min(activation.shape[0], max_channels)
    tiles = []

    for idx in range(channels):
        feature = activation[idx]
        feature = feature - feature.min()
        max_value = feature.max()
        if max_value > 1e-8:
            feature = feature / max_value
        tiles.append(feature)

    if not tiles:
        return np.zeros((64, 64), dtype=np.float32)

    grid_cols = int(math.ceil(math.sqrt(channels)))
    grid_rows = int(math.ceil(channels / grid_cols))
    height, width = tiles[0].shape
    canvas = np.zeros((grid_rows * height, grid_cols * width), dtype=np.float32)

    for idx, tile in enumerate(tiles):
        row = idx // grid_cols
        col = idx % grid_cols
        canvas[row * height:(row + 1) * height, col * width:(col + 1) * width] = tile

    return canvas


def initialize_dashboard(plt):
    fig = plt.figure(figsize=(18, 10))
    fig.subplots_adjust(left=0.06, right=0.84, top=0.92, bottom=0.08, wspace=0.35, hspace=0.35)
    grid = fig.add_gridspec(3, 5, height_ratios=[1.1, 1.0, 1.0])

    ax_frame_prev = fig.add_subplot(grid[0, 0])
    ax_frame_t = fig.add_subplot(grid[0, 1])
    ax_frame_t1 = fig.add_subplot(grid[0, 2])
    ax_traj = fig.add_subplot(grid[0, 3:])
    ax_conv_1 = fig.add_subplot(grid[1, 0:2])
    ax_conv_4 = fig.add_subplot(grid[1, 2:4])
    ax_delta = fig.add_subplot(grid[1, 4])
    ax_conv_graph = fig.add_subplot(grid[2, 0:2])
    ax_error = fig.add_subplot(grid[2, 2:5])

    frame_placeholder = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    conv_placeholder = np.zeros((64, 64), dtype=np.float32)

    frame_prev_img = ax_frame_prev.imshow(frame_placeholder, cmap="gray", vmin=0.0, vmax=1.0)
    frame_t_img = ax_frame_t.imshow(frame_placeholder, cmap="gray", vmin=0.0, vmax=1.0)
    frame_t1_img = ax_frame_t1.imshow(frame_placeholder, cmap="gray", vmin=0.0, vmax=1.0)
    conv_1_img = ax_conv_1.imshow(conv_placeholder, cmap="magma", vmin=0.0, vmax=1.0)
    conv_4_img = ax_conv_4.imshow(conv_placeholder, cmap="magma", vmin=0.0, vmax=1.0)

    ax_frame_prev.set_title("Frame t-1")
    ax_frame_t.set_title("Frame t")
    ax_frame_t1.set_title("Frame t+1")
    ax_conv_1.set_title("Conv Block 1 Activations")
    ax_conv_4.set_title("Conv Block 4 Activations")

    for axis in (ax_frame_prev, ax_frame_t, ax_frame_t1, ax_conv_1, ax_conv_4):
        axis.axis("off")

    gt_line, = ax_traj.plot([], [], color="#2e7d32", linewidth=2.5, label="Ground Truth")
    cnn_line, = ax_traj.plot([], [], color="#1565c0", linewidth=2.0, label="TinyCNN")
    orb_line, = ax_traj.plot([], [], color="#ad1457", linewidth=1.8, label="ORB VO")
    gt_head, = ax_traj.plot([], [], "o", color="#2e7d32", markersize=7)
    cnn_head, = ax_traj.plot([], [], "^", color="#1565c0", markersize=7)
    orb_head, = ax_traj.plot([], [], "s", color="#ad1457", markersize=6)
    ax_traj.set_title("Estimated Position vs Ground Truth")
    ax_traj.set_xlabel("X (m)")
    ax_traj.set_ylabel("Y (m)")
    ax_traj.grid(True, alpha=0.3)
    ax_traj.legend(loc="upper right")
    ax_traj.set_aspect("equal", adjustable="box")
    info_text = ax_traj.text(
        0.02, 0.98, "", transform=ax_traj.transAxes, va="top", ha="left",
        fontsize=9, bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"}
    )

    delta_lines = {
        "gt_dx": ax_delta.plot([], [], color="#2e7d32", linewidth=1.4, linestyle="-", label="GT dx")[0],
        "cnn_dx": ax_delta.plot([], [], color="#1565c0", linewidth=1.2, linestyle="-", label="CNN dx")[0],
        "orb_dx": ax_delta.plot([], [], color="#ad1457", linewidth=1.2, linestyle="-", label="ORB dx")[0],
        "gt_dy": ax_delta.plot([], [], color="#2e7d32", linewidth=1.4, linestyle="--", label="GT dy")[0],
        "cnn_dy": ax_delta.plot([], [], color="#1565c0", linewidth=1.2, linestyle="--", label="CNN dy")[0],
        "orb_dy": ax_delta.plot([], [], color="#ad1457", linewidth=1.2, linestyle="--", label="ORB dy")[0],
    }
    ax_delta.set_title("Local Translation")
    ax_delta.set_xlabel("Frame Index")
    ax_delta.set_ylabel("Translation (m)")
    ax_delta.grid(True, alpha=0.3)
    ax_delta.legend(loc="upper right", ncol=2, fontsize=8)

    conv_lines = [
        ax_conv_graph.plot([], [], linewidth=1.5, label=f"Conv {idx + 1}")[0]
        for idx in range(4)
    ]
    ax_conv_graph.set_title("Convolution Response Trend")
    ax_conv_graph.set_xlabel("Frame Index")
    ax_conv_graph.set_ylabel("Mean |activation|")
    ax_conv_graph.grid(True, alpha=0.3)
    ax_conv_graph.legend(loc="upper right", fontsize=8)

    cnn_error_line, = ax_error.plot([], [], color="#1565c0", linewidth=1.8, label="TinyCNN error")
    orb_error_line, = ax_error.plot([], [], color="#ad1457", linewidth=1.8, label="ORB error")
    ax_error.set_title("Position Error vs Ground Truth")
    ax_error.set_xlabel("Frame Index")
    ax_error.set_ylabel("Position error (m)")
    ax_error.grid(True, alpha=0.3)
    ax_error.legend(loc="upper left")

    fig.suptitle("Micro-UAV Perception Live Dashboard", fontsize=15, fontweight="bold")

    ax_selector = fig.add_axes([0.855, 0.56, 0.13, 0.30])
    ax_selector.set_title("Sequence", fontsize=10, pad=10)
    ax_replay = fig.add_axes([0.855, 0.48, 0.13, 0.05])
    ax_status = fig.add_axes([0.855, 0.12, 0.13, 0.32])
    ax_status.axis("off")
    status_text = ax_status.text(
        0.0, 1.0, "", va="top", ha="left", fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"}
    )

    return {
        "fig": fig,
        "frame_prev_img": frame_prev_img,
        "frame_t_img": frame_t_img,
        "frame_t1_img": frame_t1_img,
        "conv_1_img": conv_1_img,
        "conv_4_img": conv_4_img,
        "gt_line": gt_line,
        "cnn_line": cnn_line,
        "orb_line": orb_line,
        "gt_head": gt_head,
        "cnn_head": cnn_head,
        "orb_head": orb_head,
        "delta_lines": delta_lines,
        "conv_lines": conv_lines,
        "cnn_error_line": cnn_error_line,
        "orb_error_line": orb_error_line,
        "ax_traj": ax_traj,
        "ax_delta": ax_delta,
        "ax_conv_graph": ax_conv_graph,
        "ax_error": ax_error,
        "info_text": info_text,
        "ax_selector": ax_selector,
        "ax_replay": ax_replay,
        "status_text": status_text,
    }


def autoscale_axis(axis, x_values, y_values, x_margin=1.0, y_margin=1.0):
    if not x_values or not y_values:
        return

    min_x = min(x_values)
    max_x = max(x_values)
    min_y = min(y_values)
    max_y = max(y_values)

    if math.isclose(min_x, max_x):
        min_x -= x_margin
        max_x += x_margin
    else:
        span_x = max_x - min_x
        min_x -= 0.08 * span_x
        max_x += 0.08 * span_x

    if math.isclose(min_y, max_y):
        min_y -= y_margin
        max_y += y_margin
    else:
        span_y = max_y - min_y
        min_y -= 0.08 * span_y
        max_y += 0.08 * span_y

    axis.set_xlim(min_x, max_x)
    axis.set_ylim(min_y, max_y)


def refresh_dashboard(dashboard, history):
    idxs = history["frame_idx"]

    dashboard["frame_prev_img"].set_data(history["last_frame_prev"])
    dashboard["frame_t_img"].set_data(history["last_frame_t"])
    dashboard["frame_t1_img"].set_data(history["last_frame_t1"])
    dashboard["conv_1_img"].set_data(history["last_conv_1"])
    dashboard["conv_4_img"].set_data(history["last_conv_4"])

    gt_x = history["gt_x"]
    gt_y = history["gt_y"]
    cnn_x = history["cnn_x"]
    cnn_y = history["cnn_y"]
    orb_x = history["orb_x"]
    orb_y = history["orb_y"]

    dashboard["gt_line"].set_data(gt_x, gt_y)
    dashboard["cnn_line"].set_data(cnn_x, cnn_y)
    dashboard["orb_line"].set_data(orb_x, orb_y)
    dashboard["gt_head"].set_data([gt_x[-1]], [gt_y[-1]])
    dashboard["cnn_head"].set_data([cnn_x[-1]], [cnn_y[-1]])
    dashboard["orb_head"].set_data([orb_x[-1]], [orb_y[-1]])

    all_x = gt_x + cnn_x + orb_x
    all_y = gt_y + cnn_y + orb_y
    autoscale_axis(dashboard["ax_traj"], all_x, all_y)

    dashboard["delta_lines"]["gt_dx"].set_data(idxs, history["gt_dx"])
    dashboard["delta_lines"]["cnn_dx"].set_data(idxs, history["cnn_dx"])
    dashboard["delta_lines"]["orb_dx"].set_data(idxs, history["orb_dx"])
    dashboard["delta_lines"]["gt_dy"].set_data(idxs, history["gt_dy"])
    dashboard["delta_lines"]["cnn_dy"].set_data(idxs, history["cnn_dy"])
    dashboard["delta_lines"]["orb_dy"].set_data(idxs, history["orb_dy"])

    if idxs:
        dashboard["ax_delta"].set_xlim(0, max(idxs[-1], 1))
        delta_values = (
            history["gt_dx"] + history["cnn_dx"] + history["orb_dx"] +
            history["gt_dy"] + history["cnn_dy"] + history["orb_dy"]
        )
        if delta_values:
            min_delta = min(delta_values)
            max_delta = max(delta_values)
            if math.isclose(min_delta, max_delta):
                min_delta -= 0.1
                max_delta += 0.1
            else:
                span = max_delta - min_delta
                min_delta -= 0.08 * span
                max_delta += 0.08 * span
            dashboard["ax_delta"].set_ylim(min_delta, max_delta)

    for line, values in zip(dashboard["conv_lines"], history["conv_means"]):
        line.set_data(idxs, values)
    if idxs:
        dashboard["ax_conv_graph"].set_xlim(0, max(idxs[-1], 1))
        conv_values = [value for series in history["conv_means"] for value in series]
        if conv_values:
            max_conv = max(conv_values)
            dashboard["ax_conv_graph"].set_ylim(0.0, max(0.1, max_conv * 1.15))

    dashboard["cnn_error_line"].set_data(idxs, history["cnn_pos_error"])
    dashboard["orb_error_line"].set_data(idxs, history["orb_pos_error"])
    if idxs:
        dashboard["ax_error"].set_xlim(0, max(idxs[-1], 1))
        error_values = history["cnn_pos_error"] + history["orb_pos_error"]
        if error_values:
            dashboard["ax_error"].set_ylim(0.0, max(0.1, max(error_values) * 1.15))

    dashboard["info_text"].set_text(
        f"Split/sequence: {history['selection_label']}\n"
        f"Frame triplet: {history['frame_triplet_label']}\n"
        f"CNN error: {history['cnn_pos_error'][-1]:.3f} m\n"
        f"ORB error: {history['orb_pos_error'][-1]:.3f} m\n"
        f"ORB matches: {history['last_matches']}\n"
        f"CNN inference: {history['last_inference_ms']:.2f} ms"
    )
    dashboard["status_text"].set_text(
        f"Loaded:\n{history['selection_label']}\n\n"
        f"Frames: {history['num_frames']}\n"
        f"Current index: {history['frame_idx'][-1] if history['frame_idx'] else 0}\n"
        f"CNN err: {history['cnn_pos_error'][-1]:.3f} m\n"
        f"ORB err: {history['orb_pos_error'][-1]:.3f} m"
    )


def slugify_label(label):
    return label.replace("/", "_").replace("\\", "_").replace(" ", "_")


def save_outputs(dashboard, records, selection_label):
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    suffix = slugify_label(selection_label)
    dashboard["fig"].savefig(LIVE_DIR / f"live_dashboard_{suffix}_final.png", dpi=150, bbox_inches="tight")
    pd.DataFrame(records).to_csv(LIVE_DIR / f"live_dashboard_{suffix}_trace.csv", index=False)


def make_sequence_options():
    options = []
    for split_name, sequence_names in available_sequences_by_split().items():
        for sequence_name in sequence_names:
            options.append({
                "label": f"{split_name}: {sequence_name}",
                "split": split_name,
                "sequence": sequence_name,
            })
    return options


def create_history(selection_label, num_frames):
    return {
        "frame_idx": [],
        "gt_x": [0.0],
        "gt_y": [0.0],
        "cnn_x": [0.0],
        "cnn_y": [0.0],
        "orb_x": [0.0],
        "orb_y": [0.0],
        "gt_dx": [],
        "gt_dy": [],
        "cnn_dx": [],
        "cnn_dy": [],
        "orb_dx": [],
        "orb_dy": [],
        "conv_means": [[], [], [], []],
        "cnn_pos_error": [],
        "orb_pos_error": [],
        "last_frame_prev": np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32),
        "last_frame_t": np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32),
        "last_frame_t1": np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32),
        "last_conv_1": np.zeros((64, 64), dtype=np.float32),
        "last_conv_4": np.zeros((64, 64), dtype=np.float32),
        "last_matches": 0,
        "last_inference_ms": 0.0,
        "frame_triplet_label": "N/A",
        "selection_label": selection_label,
        "num_frames": num_frames,
    }


def prepare_replay_dataframe(split_name, sequence_name, stride, max_frames):
    df = load_replay_dataframe(split_name, sequence_name)
    if stride > 1:
        df = df.iloc[::stride].reset_index(drop=True)
    if max_frames is not None:
        df = df.iloc[:max_frames].reset_index(drop=True)
    if df.empty:
        raise ValueError("No triplets selected after applying stride/max-frames.")
    return df


def run_replay(args, plt, model, device, dashboard, split_name, sequence_name):
    df = prepare_replay_dataframe(split_name, sequence_name, args.stride, args.max_frames)
    selected_sequences = sorted(df["sequence"].dropna().unique().tolist())
    selection_label = f"{split_name}/{' + '.join(selected_sequences)}"
    history = create_history(selection_label, len(df))
    records = []

    gt_state = np.zeros(3, dtype=np.float32)
    cnn_state = np.zeros(3, dtype=np.float32)
    orb_state = np.zeros(3, dtype=np.float32)

    print("=" * 60)
    print("Live Micro-UAV Dashboard")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Split: {split_name}")
    print(f"Sequence filter: {sequence_name if sequence_name else 'all in split'}")
    print(f"Resolved sequences: {', '.join(selected_sequences)}")
    print(f"Frame triplets: {len(df)}")
    print(f"Playback speed: {args.playback_speed}x")
    print(f"Interactive display: {not args.no_display}")

    for idx, row in df.iterrows():
        img_prev = load_grayscale_image(PROCESSED_IMAGES, row["frame_prev"])
        img_t = load_grayscale_image(PROCESSED_IMAGES, row["frame_t"])
        img_t1 = load_grayscale_image(PROCESSED_IMAGES, row["frame_t1"])

        img_t_norm = img_t.astype(np.float32) / 255.0
        img_t1_norm = img_t1.astype(np.float32) / 255.0
        stacked = build_input_stack(img_prev, img_t, img_t1)[np.newaxis]
        tensor = torch.from_numpy(stacked).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            cnn_delta, activations = infer_with_activations(model, tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - t0) * 1000.0

        gt_delta = np.array([row["dx"], row["dy"], row["dyaw"]], dtype=np.float32)
        orb_delta, matches, orb_ok = orb_relative_pose(img_t, img_t1, K)
        if orb_ok:
            orb_delta = scale_orb_translation(orb_delta, gt_delta)

        gt_state = integrate_local_delta(gt_state, gt_delta, use_yaw=True)
        cnn_state = integrate_local_delta(cnn_state, cnn_delta, use_yaw=True)
        orb_state = integrate_local_delta(orb_state, orb_delta, use_yaw=True)

        history["frame_idx"].append(idx)
        history["gt_x"].append(float(gt_state[0]))
        history["gt_y"].append(float(gt_state[1]))
        history["cnn_x"].append(float(cnn_state[0]))
        history["cnn_y"].append(float(cnn_state[1]))
        history["orb_x"].append(float(orb_state[0]))
        history["orb_y"].append(float(orb_state[1]))
        history["gt_dx"].append(float(gt_delta[0]))
        history["gt_dy"].append(float(gt_delta[1]))
        history["cnn_dx"].append(float(cnn_delta[0]))
        history["cnn_dy"].append(float(cnn_delta[1]))
        history["orb_dx"].append(float(orb_delta[0]))
        history["orb_dy"].append(float(orb_delta[1]))

        conv_means = [float(np.mean(np.abs(activation))) for activation in activations]
        for series, value in zip(history["conv_means"], conv_means):
            series.append(value)

        cnn_error = float(np.linalg.norm(cnn_state[:2] - gt_state[:2]))
        orb_error = float(np.linalg.norm(orb_state[:2] - gt_state[:2]))
        history["cnn_pos_error"].append(cnn_error)
        history["orb_pos_error"].append(orb_error)

        history["last_frame_prev"] = img_prev.astype(np.float32) / 255.0
        history["last_frame_t"] = img_t_norm
        history["last_frame_t1"] = img_t1_norm
        history["last_conv_1"] = build_montage(activations[0], max_channels=16)
        history["last_conv_4"] = build_montage(activations[-1], max_channels=16)
        history["last_matches"] = matches
        history["last_inference_ms"] = inference_ms
        history["frame_triplet_label"] = f"{row['frame_prev']} | {row['frame_t']} | {row['frame_t1']}"

        records.append({
            "frame_index": idx,
            "split": row.get("split", split_name),
            "sequence": row.get("sequence", ""),
            "frame_prev": row["frame_prev"],
            "frame_t": row["frame_t"],
            "frame_t1": row["frame_t1"],
            "timestamp_prev": row.get("timestamp_prev", np.nan),
            "timestamp_t": row["timestamp_t"],
            "timestamp_t1": row["timestamp_t1"],
            "gt_x": float(gt_state[0]),
            "gt_y": float(gt_state[1]),
            "cnn_x": float(cnn_state[0]),
            "cnn_y": float(cnn_state[1]),
            "orb_x": float(orb_state[0]),
            "orb_y": float(orb_state[1]),
            "gt_dx": float(gt_delta[0]),
            "gt_dy": float(gt_delta[1]),
            "cnn_dx": float(cnn_delta[0]),
            "cnn_dy": float(cnn_delta[1]),
            "orb_dx": float(orb_delta[0]),
            "orb_dy": float(orb_delta[1]),
            "gt_dyaw": float(gt_delta[2]),
            "cnn_dyaw": float(cnn_delta[2]),
            "orb_dyaw": float(orb_delta[2]),
            "cnn_pos_error": cnn_error,
            "orb_pos_error": orb_error,
            "orb_matches": matches,
            "cnn_inference_ms": inference_ms,
            "conv1_mean_abs": conv_means[0],
            "conv2_mean_abs": conv_means[1],
            "conv3_mean_abs": conv_means[2],
            "conv4_mean_abs": conv_means[3],
        })

        should_refresh = ((idx + 1) % args.refresh_every == 0) or (idx == len(df) - 1)
        if should_refresh:
            refresh_dashboard(dashboard, history)
            dashboard["fig"].canvas.draw_idle()
            if args.no_display:
                dashboard["fig"].canvas.draw()
            else:
                dashboard["fig"].canvas.flush_events()
                plt.pause(0.001)

        if not args.no_display and idx < len(df) - 1:
            dt = max(float(row["timestamp_t1"] - row["timestamp_t"]), 0.0)
            delay = dt / max(args.playback_speed, 1e-6)
            if delay > 0:
                plt.pause(delay)

        if (idx + 1) % 50 == 0 or idx == len(df) - 1:
            print(
                f"Processed {idx + 1}/{len(df)} | "
                f"CNN err {cnn_error:.3f} m | ORB err {orb_error:.3f} m | "
                f"ORB matches {matches}"
            )

    if args.no_display or args.save_snapshot:
        save_outputs(dashboard, records, selection_label)
        print(f"Saved snapshot and trace to {LIVE_DIR}")

    print("Live dashboard run complete.")
    return history, records


def main():
    args = parse_args()
    if args.list_sequences:
        print_available_sequences()
        return

    if args.no_display:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    dashboard = initialize_dashboard(plt)

    if not args.no_display:
        plt.ion()
        dashboard["fig"].show()

    sequence_options = make_sequence_options()
    option_labels = [item["label"] for item in sequence_options]
    default_label = f"{args.split}: {args.sequence}" if args.sequence else f"{args.split}: {sequence_options[-1]['sequence']}" if args.split == "test" else option_labels[0]
    if default_label not in option_labels:
        default_label = option_labels[0]

    selected_target = {"label": default_label}

    radio = RadioButtons(
        dashboard["ax_selector"],
        option_labels,
        active=option_labels.index(default_label),
    )
    replay_button = Button(dashboard["ax_replay"], "Replay Selection")

    def on_sequence_selected(label):
        selected_target["label"] = label

    def on_replay_clicked(_event):
        label = selected_target["label"]
        option = next(item for item in sequence_options if item["label"] == label)
        run_replay(args, plt, model, device, dashboard, option["split"], option["sequence"])

    radio.on_clicked(on_sequence_selected)
    replay_button.on_clicked(on_replay_clicked)

    initial_option = next(item for item in sequence_options if item["label"] == default_label)
    run_replay(args, plt, model, device, dashboard, initial_option["split"], initial_option["sequence"])

    if not args.no_display:
        print("Close the dashboard window to exit.")
        plt.ioff()
        plt.show()
    else:
        plt.close(dashboard["fig"])


if __name__ == "__main__":
    main()
