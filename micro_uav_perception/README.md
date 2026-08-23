# Micro-UAV Perception Front-End

Compact monocular perception front-end for short-horizon relative motion estimation on micro-UAV footage.

## Objective

Estimate local-frame relative motion `(dx, dy, dyaw)` from short grayscale frame triplets `[t-1, t, t+1]` using a compact CNN, then integrate those increments into a trajectory for comparison against ground truth.

## Dataset

UZH-FPV Drone Racing Dataset:

- Paper: *Are We Ready for Autonomous Drone Racing? The UZH-FPV Drone Racing Dataset* (ICRA 2019)
- Website: https://fpv.ifi.uzh.ch/

## Pipeline

```text
frame_(t-1), frame_t, frame_(t+1)
            ->
TinyMotionCNN
            ->
predicted local dx, dy, dyaw
            ->
trajectory reconstruction
            ->
metrics, plots, dashboard
```

## Clean Split

The processed dataset uses a sequence-level split:

- `train`: `indoor_forward_3`, `indoor_forward_5`, `indoor_forward_6`
- `val`: `indoor_forward_7`, `indoor_forward_9`
- `test`: `indoor_forward_10`

This avoids temporal leakage between train, validation, and test.

## Setup

```powershell
cd micro_uav_perception
perception_env\Scripts\activate
python scripts/00_check_environment.py
python scripts/01_prepare_dataset.py
```

## Scripts

| Script | Purpose |
|--------|---------|
| `00_check_environment.py` | Verify dependencies |
| `01_prepare_dataset.py` | Create frame triplets and local-frame labels |
| `02_train_tiny_cnn.py` | Train the triplet CNN with augmentation and early stopping |
| `03_evaluate_model.py` | Evaluate on the held-out test sequence |
| `04_reconstruct_trajectory.py` | Integrate predicted local motion into a trajectory |
| `05_visualize_predictions.py` | Visualize triplet inputs and predictions |
| `06_classical_vo_baseline.py` | ORB-based baseline |
| `07_failure_mode_tests.py` | Stress-test the trained model under degradations |
| `08_demo_package.py` | Build the demo artifact bundle |
| `09_live_dashboard.py` | Interactive GT vs TinyCNN vs ORB dashboard with in-window sequence selection |

## Model

```text
Input: [3, 96, 96] stacked grayscale triplet
Conv2D(3->24, k=5, s=2) + ReLU
Conv2D(24->48, k=3, s=2) + ReLU
Conv2D(48->96, k=3, s=2) + ReLU
Conv2D(96->96, k=3, s=2) + ReLU
Global Average Pooling
FC(96->96) + ReLU
FC(96->3) -> [dx, dy, dyaw]
```

Training changes in the current version:

- local/body-frame labels instead of world-frame deltas
- stronger photometric augmentation
- early stopping on validation loss
- three-frame temporal context instead of isolated pairs

## Live Dashboard

```powershell
python scripts/09_live_dashboard.py --list-sequences
python scripts/09_live_dashboard.py
python scripts/09_live_dashboard.py --split test --sequence indoor_forward_10 --max-frames 300
```

The window includes an in-app sequence selector and replays frame triplets while showing trajectories, activations, local translation traces, and error curves.

## References

- Delmerico et al., "Are We Ready for Autonomous Drone Racing?", ICRA 2019
- Cioffi et al., "Continuous-Time vs. Discrete-Time Vision-based SLAM", RA-L 2022
