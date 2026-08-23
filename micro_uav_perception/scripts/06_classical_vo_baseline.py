"""
06_classical_vo_baseline.py
---------------------------
Classical visual odometry baseline using ORB features for comparison.

Important note:
`recoverPose()` returns a unit translation direction in the camera frame.
The prepared dataset labels `dx, dy` are local/body-frame planar displacements.
Those axes are still not directly comparable per frame, so this script evaluates:
  - relative yaw RMSE
  - aligned trajectory error after integrating ORB motion in its own frame
"""

import sys
import math
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_IMAGES = PROCESSED_DIR / "images"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
TABLES_DIR = RESULTS_DIR / "tables"

IMG_SIZE = 96
FOCAL_LENGTH = 50.0
CX, CY = IMG_SIZE / 2.0, IMG_SIZE / 2.0
K = np.array([[FOCAL_LENGTH, 0, CX],
              [0, FOCAL_LENGTH, CY],
              [0, 0, 1]], dtype=np.float64)


def orb_relative_pose(img1, img2, intrinsics):
    """
    Estimate relative pose between two images using ORB + essential matrix.

    Returns camera-frame translation direction plus relative yaw.
    """
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches = matcher.knnMatch(des1, des2, k=2)

    good_matches = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < 0.75 * second.distance:
            good_matches.append(best)

    if len(good_matches) < 8:
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    essential, mask = cv2.findEssentialMat(
        pts1, pts2, intrinsics, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    if essential is None:
        return None

    _, rotation, translation, mask_pose = cv2.recoverPose(
        essential, pts1, pts2, intrinsics, mask=mask
    )

    inliers = int(mask_pose.astype(bool).sum()) if mask_pose is not None else 0
    dyaw = math.atan2(rotation[1, 0], rotation[0, 0])

    return {
        "translation": translation[:, 0].astype(np.float64),
        "dyaw": dyaw,
        "matches": len(good_matches),
        "inliers": inliers,
    }


def integrate_trajectory(deltas, use_yaw=True):
    """Integrate relative motions into a 2D trajectory."""
    x, y, yaw = 0.0, 0.0, 0.0
    trajectory = [(x, y, yaw)]

    for dx, dy, dyaw in deltas:
        if use_yaw:
            x += math.cos(yaw) * dx - math.sin(yaw) * dy
            y += math.sin(yaw) * dx + math.cos(yaw) * dy
            yaw += dyaw
        else:
            x += dx
            y += dy
        trajectory.append((x, y, yaw))

    return np.array(trajectory)


def align_trajectory_similarity(pred_xy, gt_xy):
    """
    Align a predicted 2D trajectory to GT with similarity transform.

    This makes the ORB trajectory plot meaningful even though monocular VO
    lives in its own local coordinate frame and scale.
    """
    pred_mean = pred_xy.mean(axis=0)
    gt_mean = gt_xy.mean(axis=0)
    pred_centered = pred_xy - pred_mean
    gt_centered = gt_xy - gt_mean

    pred_norm = np.linalg.norm(pred_centered)
    gt_norm = np.linalg.norm(gt_centered)
    if pred_norm < 1e-8 or gt_norm < 1e-8:
        return pred_xy.copy()

    pred_unit = pred_centered / pred_norm
    gt_unit = gt_centered / gt_norm

    u, _, vt = np.linalg.svd(pred_unit.T @ gt_unit)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = u @ vt

    scale = gt_norm / pred_norm
    return scale * (pred_xy - pred_mean) @ rotation + gt_mean


def main():
    print("=" * 60)
    print("Classical VO Baseline - ORB Features")
    print("=" * 60)

    test_csv = PROCESSED_DIR / "test.csv"
    if not test_csv.exists():
        print("ERROR: Run 01_prepare_dataset.py first")
        sys.exit(1)

    df = pd.read_csv(test_csv)
    print(f"Test triplets: {len(df)}")

    orb_deltas = []
    gt_deltas = []
    failed = 0
    total_matches = 0
    total_inliers = 0

    for idx in range(len(df)):
        row = df.iloc[idx]
        img_t = cv2.imread(str(PROCESSED_IMAGES / row["frame_t"]), cv2.IMREAD_GRAYSCALE)
        img_t1 = cv2.imread(str(PROCESSED_IMAGES / row["frame_t1"]), cv2.IMREAD_GRAYSCALE)

        gt_delta = np.array([row["dx"], row["dy"], row["dyaw"]], dtype=np.float64)
        gt_deltas.append(gt_delta)

        if img_t is None or img_t1 is None:
            orb_deltas.append([0.0, 0.0, 0.0])
            failed += 1
            continue

        result = orb_relative_pose(img_t, img_t1, K)
        if result is None:
            orb_deltas.append([0.0, 0.0, 0.0])
            failed += 1
        else:
            # Keep ORB in its local camera frame; only scale translation magnitude.
            # Per-frame local/body-frame dx/dy labels are not directly comparable to these axes.
            local_xy = result["translation"][:2].copy()
            gt_mag = math.hypot(gt_delta[0], gt_delta[1])
            orb_mag = math.hypot(local_xy[0], local_xy[1])
            if orb_mag > 1e-8:
                local_xy *= gt_mag / orb_mag

            orb_deltas.append([local_xy[0], local_xy[1], result["dyaw"]])
            total_matches += result["matches"]
            total_inliers += result["inliers"]

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(df)} triplets")

    orb_deltas = np.array(orb_deltas)
    gt_deltas = np.array(gt_deltas)

    yaw_errors = orb_deltas[:, 2] - gt_deltas[:, 2]
    rmse_dyaw = np.sqrt(np.mean(yaw_errors ** 2))

    print("\nResults:")
    print(f"  Failed triplets: {failed}/{len(df)}")
    print(f"  Avg matches:  {total_matches / max(len(df) - failed, 1):.0f}")
    print(f"  Avg inliers:  {total_inliers / max(len(df) - failed, 1):.0f}")
    print(f"  RMSE dyaw:    {rmse_dyaw:.6f}")
    print("  Note: dx/dy RMSE is not reported for ORB because recoverPose translation")
    print("        is camera-frame direction, while dataset labels are body/local-frame.")

    gt_traj = integrate_trajectory(gt_deltas, use_yaw=True)
    orb_traj_local = integrate_trajectory(orb_deltas, use_yaw=True)
    orb_traj_aligned_xy = align_trajectory_similarity(orb_traj_local[:, :2], gt_traj[:, :2])
    aligned_ate = np.linalg.norm(orb_traj_aligned_xy - gt_traj[:, :2], axis=1)
    mean_ate = float(np.mean(aligned_ate))
    final_drift = float(aligned_ate[-1])

    print(f"  Mean ATE:     {mean_ate:.6f}")
    print(f"  Final drift:  {final_drift:.6f}")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    ax.plot(gt_traj[:, 0], gt_traj[:, 1], "g-", linewidth=2, label="Ground Truth", alpha=0.8)
    ax.plot(orb_traj_local[:, 0], orb_traj_local[:, 1], "m-", linewidth=1.5, label="ORB local frame", alpha=0.8)
    ax.plot(gt_traj[0, 0], gt_traj[0, 1], "go", markersize=10, label="Start")
    ax.set_title("Raw Trajectory Frames (Not Directly Comparable)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect("equal")

    ax = axes[1]
    ax.plot(gt_traj[:, 0], gt_traj[:, 1], "g-", linewidth=2, label="Ground Truth", alpha=0.8)
    ax.plot(orb_traj_aligned_xy[:, 0], orb_traj_aligned_xy[:, 1], "m-", linewidth=1.5,
            label="ORB trajectory (similarity aligned)", alpha=0.8)
    ax.plot(gt_traj[0, 0], gt_traj[0, 1], "go", markersize=10, label="Start")
    ax.set_title("Classical VO Trajectory (Aligned for Evaluation)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "orb_vo_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    cnn_metrics_path = TABLES_DIR / "evaluation_metrics.csv"
    cnn_row = {
        "method": "TinyCNN",
        "RMSE_dx": "N/A",
        "RMSE_dy": "N/A",
        "RMSE_dyaw": "N/A",
        "FPS": "N/A",
        "strengths": "compact, trainable, GPU-accelerated",
        "weaknesses": "requires training data, drift",
    }
    if cnn_metrics_path.exists():
        cnn_df = pd.read_csv(cnn_metrics_path)
        if len(cnn_df) > 0:
            row = cnn_df.iloc[0]
            cnn_row.update({
                "RMSE_dx": row.get("RMSE_dx", "N/A"),
                "RMSE_dy": row.get("RMSE_dy", "N/A"),
                "RMSE_dyaw": row.get("RMSE_dyaw", "N/A"),
                "FPS": row.get("FPS", "N/A"),
            })

    comparison = pd.DataFrame([
        {
            "method": "ORB VO",
            "input": "image pair (t to t+1)",
            "output": "local-frame relative pose",
            "RMSE_dx": "N/A",
            "RMSE_dy": "N/A",
            "RMSE_dyaw": f"{rmse_dyaw:.6f}",
            "mean_ATE_aligned": f"{mean_ate:.6f}",
            "final_drift_aligned": f"{final_drift:.6f}",
            "FPS": "N/A",
            "strengths": "interpretable, no training needed",
            "weaknesses": "monocular scale ambiguity, frame mismatch vs body-frame labels",
        },
        {
            "method": cnn_row["method"],
            "input": "3x96x96 grayscale triplet",
            "output": "local-frame dx dy dyaw",
            "RMSE_dx": cnn_row["RMSE_dx"],
            "RMSE_dy": cnn_row["RMSE_dy"],
            "RMSE_dyaw": cnn_row["RMSE_dyaw"],
            "mean_ATE_aligned": "N/A",
            "final_drift_aligned": "N/A",
            "FPS": cnn_row["FPS"],
            "strengths": cnn_row["strengths"],
            "weaknesses": cnn_row["weaknesses"],
        }
    ])
    comparison.to_csv(TABLES_DIR / "baseline_comparison.csv", index=False)

    print(f"\nComparison saved to {TABLES_DIR / 'baseline_comparison.csv'}")
    print("Classical VO baseline complete!")


if __name__ == "__main__":
    main()
