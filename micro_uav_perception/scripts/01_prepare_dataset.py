"""
01_prepare_dataset.py
---------------------
Prepare a clean sequence-level split from the public UZH-FPV indoor-forward
Snapdragon sequences.

Split policy:
  - train: indoor_forward_3, indoor_forward_5, indoor_forward_6
  - val:   indoor_forward_7, indoor_forward_9
  - test:  indoor_forward_10

For each sequence:
  - Read left_images.txt for image timestamps
  - Read groundtruth.txt for poses
  - Match images to nearest GT pose within a threshold
  - Compute local-frame relative motion labels (dx, dy, dyaw)
  - Resize images to 96x96 grayscale
  - Save flattened processed frame files with sequence-prefixed names

Outputs:
  data/processed/images/
  data/processed/labels.csv
  data/processed/train.csv
  data/processed/val.csv
  data/processed/test.csv
  results/tables/dataset_summary.csv
"""

import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from perception_common import world_to_local_delta, wrap_angle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "datasets" / "uzh_fpv"
SEQUENCES_DIR = DATASET_DIR / "sequences"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_IMAGES = PROCESSED_DIR / "images"
RESULTS_TABLES_DIR = PROJECT_ROOT / "results" / "tables"

IMG_SIZE = 96
MATCH_THRESHOLD = 0.05  # max time diff (seconds) for matching image to GT

SEQUENCE_SPLITS = {
    "train": ["indoor_forward_3", "indoor_forward_5", "indoor_forward_6"],
    "val": ["indoor_forward_7", "indoor_forward_9"],
    "test": ["indoor_forward_10"],
}


def quaternion_to_yaw(qx, qy, qz, qw):
    """Extract yaw angle from quaternion (rotation about Z axis)."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def load_image_list(path):
    """Parse left_images.txt into a list of (timestamp, image_path)."""
    entries = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                entries.append((float(parts[1]), parts[2]))
    return entries


def load_groundtruth(path):
    """Parse groundtruth.txt into arrays of timestamps, positions, quaternions."""
    timestamps, positions, quaternions = [], [], []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 8:
                timestamps.append(float(parts[0]))
                positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
                quaternions.append([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])
    return np.array(timestamps), np.array(positions), np.array(quaternions)


def match_image_to_gt(img_ts, gt_timestamps):
    """Find the nearest GT index for an image timestamp."""
    idx = np.searchsorted(gt_timestamps, img_ts)
    candidates = []
    if idx > 0:
        candidates.append(idx - 1)
    if idx < len(gt_timestamps):
        candidates.append(idx)
    if not candidates:
        return None, float("inf")
    best_idx = min(candidates, key=lambda gt_idx: abs(gt_timestamps[gt_idx] - img_ts))
    return best_idx, abs(gt_timestamps[best_idx] - img_ts)


def clear_previous_outputs():
    """Remove stale processed files so the split is reproducible."""
    PROCESSED_IMAGES.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    for image_path in PROCESSED_IMAGES.glob("*.png"):
        image_path.unlink()

    for csv_name in ["labels.csv", "train.csv", "val.csv", "test.csv"]:
        csv_path = PROCESSED_DIR / csv_name
        if csv_path.exists():
            csv_path.unlink()

    summary_path = RESULTS_TABLES_DIR / "dataset_summary.csv"
    if summary_path.exists():
        summary_path.unlink()


def validate_sequence_dirs():
    """Ensure every configured sequence folder exists."""
    if not SEQUENCES_DIR.exists():
        print(f"ERROR: Sequence directory not found: {SEQUENCES_DIR}")
        sys.exit(1)

    missing = []
    for split_name, sequence_names in SEQUENCE_SPLITS.items():
        for sequence_name in sequence_names:
            sequence_dir = SEQUENCES_DIR / sequence_name
            if not sequence_dir.exists():
                missing.append((split_name, sequence_name))

    if missing:
        print("ERROR: Missing required sequence folders:")
        for split_name, sequence_name in missing:
            print(f"  {split_name}: {sequence_name}")
        sys.exit(1)


def process_sequence(sequence_name, split_name):
    """Prepare matched frame triplets and processed images for one sequence."""
    sequence_dir = SEQUENCES_DIR / sequence_name
    left_images_file = sequence_dir / "left_images.txt"
    gt_file = sequence_dir / "groundtruth.txt"
    img_dir = sequence_dir / "img"

    for path in [left_images_file, gt_file, img_dir]:
        if not path.exists():
            print(f"ERROR: Missing {path}")
            sys.exit(1)

    print(f"\n[{split_name}] {sequence_name}")
    print("-" * 60)

    image_entries = load_image_list(left_images_file)
    gt_timestamps, gt_positions, gt_quaternions = load_groundtruth(gt_file)

    print(f"Images in list: {len(image_entries)}")
    print(f"GT samples:     {len(gt_timestamps)}")

    matched = []
    skipped = 0

    for img_idx, (img_ts, img_path) in enumerate(image_entries):
        gt_idx, dt = match_image_to_gt(img_ts, gt_timestamps)
        if gt_idx is None or dt > MATCH_THRESHOLD:
            skipped += 1
            continue

        tx, ty, tz = gt_positions[gt_idx]
        qx, qy, qz, qw = gt_quaternions[gt_idx]
        yaw = quaternion_to_yaw(qx, qy, qz, qw)

        matched.append({
            "img_idx": img_idx,
            "timestamp": img_ts,
            "img_path": img_path,
            "tx": tx,
            "ty": ty,
            "tz": tz,
            "yaw": yaw,
            "gt_dt": dt,
        })

    print(f"Matched frames: {len(matched)} / {len(image_entries)} (skipped {skipped})")
    if len(matched) < 10:
        print(f"ERROR: Too few matched frames in {sequence_name}")
        sys.exit(1)

    written_frames = set()
    samples = []

    for idx in tqdm(range(1, len(matched) - 1), desc=f"Processing {sequence_name}", leave=False):
        prev = matched[idx - 1]
        curr = matched[idx]
        nxt = matched[idx + 1]

        img_prev_path = sequence_dir / prev["img_path"]
        img_curr_path = sequence_dir / curr["img_path"]
        img_nxt_path = sequence_dir / nxt["img_path"]

        img_prev = cv2.imread(str(img_prev_path), cv2.IMREAD_GRAYSCALE)
        img_curr = cv2.imread(str(img_curr_path), cv2.IMREAD_GRAYSCALE)
        img_nxt = cv2.imread(str(img_nxt_path), cv2.IMREAD_GRAYSCALE)
        if img_prev is None or img_curr is None or img_nxt is None:
            continue

        fname_prev = f"{sequence_name}_frame_{prev['img_idx']:05d}.png"
        fname_curr = f"{sequence_name}_frame_{curr['img_idx']:05d}.png"
        fname_nxt = f"{sequence_name}_frame_{nxt['img_idx']:05d}.png"

        if fname_prev not in written_frames:
            img_prev_r = cv2.resize(img_prev, (IMG_SIZE, IMG_SIZE))
            cv2.imwrite(str(PROCESSED_IMAGES / fname_prev), img_prev_r)
            written_frames.add(fname_prev)

        if fname_curr not in written_frames:
            img_curr_r = cv2.resize(img_curr, (IMG_SIZE, IMG_SIZE))
            cv2.imwrite(str(PROCESSED_IMAGES / fname_curr), img_curr_r)
            written_frames.add(fname_curr)

        if fname_nxt not in written_frames:
            img_nxt_r = cv2.resize(img_nxt, (IMG_SIZE, IMG_SIZE))
            cv2.imwrite(str(PROCESSED_IMAGES / fname_nxt), img_nxt_r)
            written_frames.add(fname_nxt)

        dx_world = nxt["tx"] - curr["tx"]
        dy_world = nxt["ty"] - curr["ty"]
        dx_local, dy_local = world_to_local_delta(dx_world, dy_world, curr["yaw"])
        dyaw = wrap_angle(nxt["yaw"] - curr["yaw"])

        samples.append({
            "split": split_name,
            "sequence": sequence_name,
            "frame_prev": fname_prev,
            "frame_t": fname_curr,
            "frame_t1": fname_nxt,
            "dx": dx_local,
            "dy": dy_local,
            "dyaw": dyaw,
            "dx_world": dx_world,
            "dy_world": dy_world,
            "yaw_t": curr["yaw"],
            "yaw_t1": nxt["yaw"],
            "timestamp_prev": prev["timestamp"],
            "timestamp_t": curr["timestamp"],
            "timestamp_t1": nxt["timestamp"],
            "source_img_prev": prev["img_path"],
            "source_img_t": curr["img_path"],
            "source_img_t1": nxt["img_path"],
        })

    print(f"Generated triplets: {len(samples)}")

    sequence_summary = {
        "split": split_name,
        "sequence": sequence_name,
        "num_images_listed": len(image_entries),
        "num_images_matched": len(matched),
        "num_pairs": len(samples),
        "num_processed_frames": len(written_frames),
        "image_resolution": f"{IMG_SIZE}x{IMG_SIZE}",
        "label_type": "local_dx,local_dy,dyaw",
        "mean_abs_gt_dt_ms": float(np.mean([item["gt_dt"] for item in matched]) * 1000.0),
        "max_abs_gt_dt_ms": float(np.max([item["gt_dt"] for item in matched]) * 1000.0),
    }
    return samples, sequence_summary


def build_dataset_summary(labels_df, sequence_summaries):
    """Create a compact summary table for the prepared dataset."""
    summary_rows = list(sequence_summaries)

    for split_name in ["train", "val", "test"]:
        split_df = labels_df[labels_df["split"] == split_name]
        split_unique_frames = int(pd.unique(pd.concat([split_df["frame_prev"], split_df["frame_t"], split_df["frame_t1"]])).size)
        summary_rows.append({
            "split": split_name,
            "sequence": "__split_total__",
            "num_images_listed": split_unique_frames,
            "num_images_matched": "",
            "num_pairs": len(split_df),
            "num_processed_frames": split_unique_frames,
            "image_resolution": f"{IMG_SIZE}x{IMG_SIZE}",
            "label_type": "local_dx,local_dy,dyaw",
            "mean_abs_gt_dt_ms": "",
            "max_abs_gt_dt_ms": "",
        })

    all_unique_frames = int(pd.unique(pd.concat([labels_df["frame_prev"], labels_df["frame_t"], labels_df["frame_t1"]])).size)
    summary_rows.append({
        "split": "all",
        "sequence": "__all__",
        "num_images_listed": all_unique_frames,
        "num_images_matched": "",
        "num_pairs": len(labels_df),
        "num_processed_frames": all_unique_frames,
        "image_resolution": f"{IMG_SIZE}x{IMG_SIZE}",
        "label_type": "local_dx,local_dy,dyaw",
        "mean_abs_gt_dt_ms": "",
        "max_abs_gt_dt_ms": "",
    })

    return pd.DataFrame(summary_rows)


def main():
    print("=" * 60)
    print("Dataset Preparation - UZH-FPV Clean Sequence Split")
    print("=" * 60)
    print("Split policy:")
    for split_name, sequence_names in SEQUENCE_SPLITS.items():
        print(f"  {split_name}: {', '.join(sequence_names)}")

    validate_sequence_dirs()
    clear_previous_outputs()

    all_pairs = []
    sequence_summaries = []

    for split_name in ["train", "val", "test"]:
        for sequence_name in SEQUENCE_SPLITS[split_name]:
            samples, summary = process_sequence(sequence_name, split_name)
            all_pairs.extend(samples)
            sequence_summaries.append(summary)

    if not all_pairs:
        print("ERROR: No frame triplets generated.")
        sys.exit(1)

    labels_df = pd.DataFrame(all_pairs)
    labels_df.to_csv(PROCESSED_DIR / "labels.csv", index=False)

    train_df = labels_df[labels_df["split"] == "train"].reset_index(drop=True)
    val_df = labels_df[labels_df["split"] == "val"].reset_index(drop=True)
    test_df = labels_df[labels_df["split"] == "test"].reset_index(drop=True)

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    summary_df = build_dataset_summary(labels_df, sequence_summaries)
    summary_df.to_csv(RESULTS_TABLES_DIR / "dataset_summary.csv", index=False)

    print("\nLabel statistics:")
    for column in ["dx", "dy", "dyaw"]:
        print(
            f"  {column}: mean={labels_df[column].mean():.6f}  "
            f"std={labels_df[column].std():.6f}  "
            f"min={labels_df[column].min():.6f}  "
            f"max={labels_df[column].max():.6f}"
        )

    print("\nPrepared splits:")
    print(f"  train: {len(train_df)} triplets from {sorted(train_df['sequence'].unique().tolist())}")
    print(f"  val:   {len(val_df)} triplets from {sorted(val_df['sequence'].unique().tolist())}")
    print(f"  test:  {len(test_df)} triplets from {sorted(test_df['sequence'].unique().tolist())}")

    print("\nOutputs:")
    print(f"  {PROCESSED_DIR / 'labels.csv'}")
    print(f"  {PROCESSED_DIR / 'train.csv'}")
    print(f"  {PROCESSED_DIR / 'val.csv'}")
    print(f"  {PROCESSED_DIR / 'test.csv'}")
    print(f"  {RESULTS_TABLES_DIR / 'dataset_summary.csv'}")
    print("=" * 60)
    print("Dataset preparation complete.")


if __name__ == "__main__":
    main()
