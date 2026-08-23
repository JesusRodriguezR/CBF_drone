# Implementation Plan — Lightweight Monocular Perception Front-End for Micro-UAV Localization

## Objective

Implement a first research-oriented perception front-end for micro-UAV localization using a lightweight monocular deep neural network.

The goal is not to implement full SLAM immediately. The goal is to create a measurable perception module that receives visual input and estimates relative motion between frames. This creates a bridge between raw onboard sensing and the state information later required by navigation and control modules.

The first target is:

image_t, image_t+1 -> relative motion estimate

The first version should estimate planar relative motion:

Δx, Δy, Δyaw

A simpler first implementation may estimate only:

Δx, Δy

The system should produce trajectory reconstruction, error metrics, inference-time measurements, and failure-mode analysis.

---

## Motivation

Previous control experiments assume access to ground-truth state information, usually from simulation or motion-capture systems. However, in real micro-UAV autonomy, position, velocity, and obstacle information must be estimated from onboard sensors under strict constraints:

- limited onboard computation,
- limited payload,
- low-resolution camera input,
- sensor noise,
- motion blur,
- lighting variation,
- texture-poor scenes,
- drift,
- latency.

This implementation focuses on the perception/localization front-end before returning to the control and safety layers.

---

## Literature-Aligned Direction

The implementation is inspired by lightweight perception systems for nano-UAVs and micro-UAVs, especially works involving:

- DNN-based visual navigation on nano-drones,
- monocular relative pose estimation,
- visual-inertial odometry,
- CNN-based ego-motion estimation,
- compact neural networks for onboard inference,
- Crazyflie-class autonomy.

This prototype should be framed as a first step toward a compact learned visual odometry module, not as a full replacement for mature SLAM systems such as ORB-SLAM3 or VINS-Mono.

---

## Core Research Question

Can a lightweight monocular neural network estimate useful frame-to-frame motion for micro-UAV localization under constrained sensing conditions?

Secondary questions:

1. How accurate is a small CNN compared with simple classical visual odometry?
2. How much drift accumulates when predicted relative motions are integrated?
3. What failure modes appear under motion blur, low texture, rotation, and lighting changes?
4. What inference speed can be achieved on a standard Windows PC?
5. Can the same pipeline later be adapted to Crazyflie AI Deck or PC-side Crazyflie perception?

---

## Expected Final Output

input image pair
        ↓
tiny CNN relative-motion estimator
        ↓
predicted Δx, Δy, Δyaw
        ↓
trajectory reconstruction
        ↓
comparison with ground truth or pseudo-ground truth
        ↓
metrics and plots

---

# Dataset Choice

## Recommended Dataset for First Implementation

Use the UZH-FPV Drone Racing Dataset first.

Reason:

- It is UAV-specific.
- It contains aggressive flight.
- It includes image data, IMU data, and ground truth.
- It is more aligned with micro-UAV perception than generic robotics datasets.
- It is suitable for visual odometry and visual-inertial odometry experiments.

Official dataset website:

https://fpv.ifi.uzh.ch/

Paper:

Are We Ready for Autonomous Drone Racing? The UZH-FPV Drone Racing Dataset

Recommended first download:

Use one indoor forward-facing sequence first, preferably a small indoor sequence for quick testing.

Suggested starting sequence:

Indoor forward-facing sequence 03, 05, 06, or 10.

For the first prototype, use only one sequence until the pipeline works.

Later split example:

Training:
- indoor_forward_03
- indoor_forward_05
- indoor_forward_06

Validation:
- indoor_forward_07
- indoor_forward_09

Testing:
- indoor_forward_10

If the download page provides multiple formats, prioritize the format that includes:

- camera images,
- IMU data,
- ground truth pose,
- timestamps.

Avoid downloading the full dataset at first unless storage and download time are not an issue.

---

# Phase 1 — Environment Setup

## Goal

Prepare a clean Python environment on Windows for training and testing a lightweight monocular relative-pose network.

## Required Software

- Windows PC
- Python 3.10 or newer
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- Pandas
- tqdm
- scikit-learn

## Environment Creation

Run in Windows PowerShell or CMD:

python -m venv perception_env
perception_env\Scripts\activate
python -m pip install --upgrade pip
pip install torch torchvision torchaudio
pip install opencv-python numpy matplotlib pandas tqdm scikit-learn

If CUDA is available and PyTorch does not detect the GPU, install the CUDA-compatible PyTorch version from the official PyTorch installation page:

https://pytorch.org/get-started/locally/

## Verification Script

Create:

scripts/00_check_environment.py

Content:

import torch
import cv2
import numpy as np
import sys

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)

## Deliverable

results/environment_check.txt

This file should contain:

- Python version,
- PyTorch version,
- CUDA availability,
- OpenCV version,
- NumPy version.

---

# Phase 2 — Project Structure

## Goal

Create a clean project structure for datasets, scripts, trained models, and results.

## Folder Structure

micro_uav_perception/
  data/
    raw/
    processed/
    datasets/
      uzh_fpv/
      crazyflie/
      webcam/
  scripts/
    00_check_environment.py
    01_prepare_dataset.py
    02_train_tiny_cnn.py
    03_evaluate_model.py
    04_reconstruct_trajectory.py
    05_visualize_predictions.py
    06_classical_vo_baseline.py
  models/
    tiny_cnn_relative_pose.pt
  results/
    plots/
    videos/
    tables/
    logs/
    demo/
  notes/
    implementation_log.md
    failure_modes.md
  README.md

## Deliverable

A project folder with the structure above.

---

# Phase 3 — Dataset Download and Organization

## Goal

Download one UZH-FPV sequence and organize it for preprocessing.

## Download Instructions

1. Go to:

https://fpv.ifi.uzh.ch/

2. Open the dataset download section.

3. Download one indoor forward-facing sequence first.

Recommended:

- Indoor forward-facing sequence 03, or
- Indoor forward-facing sequence 10 if you want to reserve it as a test sequence later.

4. Extract the downloaded files into:

micro_uav_perception/data/datasets/uzh_fpv/

Example structure after extraction:

micro_uav_perception/data/datasets/uzh_fpv/
  indoor_forward_03/
    images/
    imu/
    groundtruth/
    timestamps/

The exact folder names may differ depending on the dataset package. The important files are:

- image frames,
- image timestamps,
- IMU measurements,
- ground-truth poses.

## If the Dataset Format Is Different

Create a local mapping note:

notes/dataset_format_notes.md

Include:

- where images are stored,
- where timestamps are stored,
- where ground truth is stored,
- where IMU data is stored,
- units used by the ground truth,
- coordinate convention if documented.

## Deliverables

data/datasets/uzh_fpv/[selected_sequence]/
notes/dataset_format_notes.md

---

# Phase 4 — Dataset Preparation

## Goal

Create image pairs and relative-motion labels.

The expected dataset format is:

frame_t, frame_t1, dx, dy, dyaw

Example labels.csv:

frame_t,frame_t1,dx,dy,dyaw
000001.png,000002.png,0.0021,-0.0004,0.0010
000002.png,000003.png,0.0023,-0.0002,0.0007

## Image Preprocessing

For the first version:

- convert images to grayscale,
- resize to 96 x 96,
- normalize pixel values to [0, 1],
- create consecutive frame pairs.

Input tensor:

2 x 96 x 96

where channel 1 is frame_t and channel 2 is frame_t+1.

## Label Computation

Given two poses:

pose_t
pose_t+1

compute relative motion:

relative_pose = inverse(pose_t) * pose_t+1

For the first planar version, extract:

dx
dy
dyaw

If full 6-DoF labels are available but the first model is planar, ignore vertical motion and roll/pitch initially.

## Practical First Version

To avoid getting blocked by full quaternion handling on day one, start with translation-only labels:

dx = x_{t+1} - x_t
dy = y_{t+1} - y_t

Then add dyaw once pose parsing is working.

## Dataset Split

For one sequence:

- 70% training,
- 15% validation,
- 15% testing.

For multiple sequences later:

- train on some sequences,
- validate on separate sequences,
- test on a sequence never used during training.

## Deliverables

data/processed/images/
data/processed/labels.csv
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
results/tables/dataset_summary.csv

The dataset summary should include:

number of images
number of image pairs
image resolution
label type
train/validation/test split

---

# Phase 5 — Tiny CNN Relative-Pose Model

## Goal

Implement a compact CNN that estimates relative motion from two consecutive grayscale frames.

## Input

frame_t, frame_t+1

as a stacked tensor:

[2, 96, 96]

## Output

First version:

[dx, dy, dyaw]

Alternative minimal version:

[dx, dy]

## Proposed Architecture

Input: 2 x 96 x 96

Conv2D: 2 -> 16, kernel 5, stride 2
ReLU
Conv2D: 16 -> 32, kernel 3, stride 2
ReLU
Conv2D: 32 -> 64, kernel 3, stride 2
ReLU
Conv2D: 64 -> 64, kernel 3, stride 2
ReLU
Global Average Pooling
Fully Connected: 64 -> 64
ReLU
Fully Connected: 64 -> 3

If only dx and dy are estimated, the final layer should be:

Fully Connected: 64 -> 2

## Loss Function

Use mean squared error:

L = MSE(predicted_delta, ground_truth_delta)

If using yaw, optionally weight yaw:

L = MSE(dx_pred, dx_gt) + MSE(dy_pred, dy_gt) + λ MSE(dyaw_pred, dyaw_gt)

Initial value:

λ = 0.1

## Training Settings

Initial values:

epochs: 50
batch size: 32
optimizer: Adam
learning rate: 1e-3
train/val/test split: 70/15/15
image size: 96 x 96

## Deliverables

scripts/02_train_tiny_cnn.py
models/tiny_cnn_relative_pose.pt
results/plots/training_loss.png
results/plots/validation_loss.png
results/logs/training_log.txt

---

# Phase 6 — Model Evaluation

## Goal

Evaluate the trained model on unseen image pairs.

## Metrics

Compute:

RMSE_dx
RMSE_dy
RMSE_dyaw
MAE_dx
MAE_dy
MAE_dyaw
inference_time_per_frame_pair
FPS

If using only dx, dy, compute:

RMSE_position_delta
MAE_position_delta

## Evaluation Output

Create a table:

model,input,output,RMSE_dx,RMSE_dy,RMSE_dyaw,FPS
TinyCNN,2 grayscale frames,dx dy dyaw,...

## Deliverables

scripts/03_evaluate_model.py
results/tables/evaluation_metrics.csv
results/plots/predicted_vs_groundtruth_dx.png
results/plots/predicted_vs_groundtruth_dy.png
results/plots/predicted_vs_groundtruth_dyaw.png

---

# Phase 7 — Trajectory Reconstruction

## Goal

Integrate frame-to-frame predictions into a trajectory.

## Simple Planar Reconstruction

Initialize:

x_0 = 0
y_0 = 0
yaw_0 = 0

For each predicted relative motion:

yaw_{t+1} = yaw_t + Δyaw
x_{t+1} = x_t + cos(yaw_t) Δx - sin(yaw_t) Δy
y_{t+1} = y_t + sin(yaw_t) Δx + cos(yaw_t) Δy

If yaw is not estimated, use:

x_{t+1} = x_t + Δx
y_{t+1} = y_t + Δy

## Compare Against Ground Truth

Plot:

- ground-truth trajectory,
- predicted trajectory,
- trajectory error over time.

## Metrics

Compute:

Absolute Trajectory Error
final drift
drift per second
drift per meter

## Deliverables

scripts/04_reconstruct_trajectory.py
results/plots/reconstructed_trajectory.png
results/plots/trajectory_error_over_time.png
results/tables/trajectory_metrics.csv

---

# Phase 8 — Classical Baseline

## Goal

Implement a simple classical visual odometry baseline for comparison.

This is not the main contribution. It is used to show why the learned method is meaningful and to provide a baseline.

## Baseline Method

Use:

- ORB feature detection,
- descriptor matching,
- essential matrix estimation,
- pose recovery,
- trajectory integration.

## Expected Limitations

Document the following:

- scale ambiguity,
- drift,
- feature loss in low-texture scenes,
- instability under motion blur,
- rotation/translation ambiguity.

## Deliverables

scripts/06_classical_vo_baseline.py
results/plots/orb_vo_trajectory.png
results/tables/baseline_comparison.csv
notes/failure_modes.md

The comparison table should include:

method,input,output,strengths,weaknesses,FPS,error_metric
ORB VO,image pair,relative pose,interpretable,scale ambiguity,...
Tiny CNN,image pair,relative motion,compact and trainable,requires data,...

---

# Phase 9 — Failure Mode Tests

## Goal

Evaluate how the perception module behaves under conditions relevant to micro-UAV flight.

## Test Conditions

Prepare or select sequences with:

1. normal textured scene,
2. low-texture wall or floor,
3. fast motion,
4. motion blur,
5. yaw rotation,
6. forward motion,
7. lighting change,
8. vibration or shaky input.

## Expected Observations

Create a table:

condition,expected_problem,observed_behavior,impact_on_localization

Example:

low texture,few visual features,higher prediction error,drift increases
fast yaw,rotation dominates image motion,translation estimate degrades,false displacement
motion blur,feature and texture loss,noisy predictions,trajectory instability

## Deliverables

notes/failure_modes.md
results/tables/failure_mode_summary.csv
results/plots/failure_mode_errors.png

---

# Phase 10 — One-Day Demo Package

## Goal

Prepare a compact demo that can be shown in a thesis meeting.

## Demo Structure

The demo should include:

1. input image pair,
2. predicted relative motion,
3. reconstructed trajectory,
4. ground-truth trajectory,
5. error plot,
6. inference speed,
7. short failure-mode table.

## Suggested Demo Title

Lightweight Learned Visual Odometry Front-End for Crazyflie-Class UAVs

Alternative title:

Tiny Monocular Relative-Pose Network for Micro-UAV Perception

## Deliverables

results/demo/
  demo_input_frames.png
  demo_prediction_overlay.png
  demo_reconstructed_trajectory.png
  demo_error_plot.png
  demo_metrics.csv
  demo_summary.md

## Demo Summary Template

# Demo Summary

## Objective

Implement a lightweight monocular neural network for relative motion estimation from image pairs.

## Input

Two consecutive grayscale images resized to 96 x 96.

## Output

Predicted relative motion: Δx, Δy, Δyaw.

## Dataset

UZH-FPV Drone Racing Dataset.

## Results

| Metric | Value |
|---|---|
| RMSE dx | |
| RMSE dy | |
| RMSE dyaw | |
| FPS | |
| Final drift | |

## Observations

- The network captures useful frame-to-frame motion trends.
- Drift accumulates during trajectory reconstruction.
- Performance degrades under low texture, fast yaw, and motion blur.
- The model provides a measurable perception front-end for later control integration.

## Next Step

Add IMU measurements to reduce ambiguity and improve robustness.

---

# Phase 11 — Next Upgrade: Visual-Inertial Version

## Goal

Extend the monocular model with IMU information.

## Input

frame_t
frame_t+1
gyro samples between t and t+1
accelerometer samples between t and t+1

## Output

Δx, Δy, Δyaw

## Proposed Architecture

Image branch:

Tiny CNN

IMU branch:

MLP or 1D CNN

Fusion:

concatenate visual features + IMU features
fully connected layers
relative motion output

## Why This Matters

Monocular vision alone suffers from ambiguity between:

- translation,
- rotation,
- depth,
- scene scale.

IMU helps disambiguate fast rotations, acceleration, and short-term motion.

## Deliverables

scripts/07_train_visual_inertial_model.py
models/tiny_cnn_imu_relative_pose.pt
results/tables/visual_vs_visual_inertial_comparison.csv
results/plots/visual_inertial_trajectory.png

---

# Minimum Viable Deliverables

At the end of the first implementation cycle, the minimum acceptable deliverables are:

1. Dataset downloaded and organized.
2. Dataset format notes written.
3. Dataset preparation script.
4. Tiny CNN training script.
5. Trained model file.
6. Evaluation script.
7. Trajectory reconstruction plot.
8. Error metrics table.
9. Classical baseline comparison.
10. Failure-mode notes.
11. Demo summary markdown.

---

# Recommended First-Day Checklist

## Morning

- Create Python environment.
- Create project folder.
- Download one UZH-FPV sequence from https://fpv.ifi.uzh.ch/.
- Organize the sequence inside data/datasets/uzh_fpv/.
- Identify image folder, timestamps, IMU file, and ground-truth file.
- Prepare a small image-pair dataset.
- Confirm image loading and preprocessing.

## Midday

- Implement tiny CNN.
- Train on a small subset.
- Save training and validation loss plots.

## Afternoon

- Evaluate predictions.
- Reconstruct trajectory.
- Plot predicted vs ground-truth trajectory.

## End of Day

- Create a short demo summary.
- Write observed failure modes.
- List next steps.

---

# First-Day Success Criteria

The first implementation is successful if it can:

load image pairs
train a small CNN
predict relative motion
reconstruct a trajectory
compute error metrics
generate plots

It does not need to be highly accurate on day one.

The objective is to establish the full perception pipeline.

---

# Research Framing for Presentation

Use the following phrasing:

The first implementation focuses on a lightweight learned visual front-end for micro-UAV localization. Instead of assuming perfect state feedback, the system estimates frame-to-frame relative motion directly from monocular image pairs. This produces a measurable localization signal that can be evaluated through trajectory reconstruction, drift, and inference latency. The current version is PC-based and will later be extended with IMU fusion and Crazyflie/OptiTrack data.

---

# Expected Limitations

Document these honestly:

- The first model is not full SLAM.
- Monocular scale may be ambiguous depending on the dataset.
- The model requires labeled motion data.
- Trajectory drift will accumulate.
- Generalization to new environments may be weak.
- Real onboard deployment requires further compression and latency testing.

---

# Next Steps After First Prototype

After the first prototype works:

1. Add IMU input.
2. Train with Crazyflie/OptiTrack data.
3. Compare against ORB-based visual odometry.
4. Measure inference latency.
5. Test under motion blur and low texture.
6. Estimate uncertainty bounds.
7. Feed estimated pose into navigation/control pipeline later.

---

# Final Deliverable List

micro_uav_perception/
  README.md
  data/datasets/uzh_fpv/[selected_sequence]/
  data/processed/labels.csv
  data/processed/train.csv
  data/processed/val.csv
  data/processed/test.csv
  scripts/00_check_environment.py
  scripts/01_prepare_dataset.py
  scripts/02_train_tiny_cnn.py
  scripts/03_evaluate_model.py
  scripts/04_reconstruct_trajectory.py
  scripts/06_classical_vo_baseline.py
  models/tiny_cnn_relative_pose.pt
  results/plots/training_loss.png
  results/plots/validation_loss.png
  results/plots/reconstructed_trajectory.png
  results/plots/trajectory_error_over_time.png
  results/tables/dataset_summary.csv
  results/tables/evaluation_metrics.csv
  results/tables/trajectory_metrics.csv
  results/tables/baseline_comparison.csv
  notes/dataset_format_notes.md
  notes/failure_modes.md
  results/demo/demo_summary.md