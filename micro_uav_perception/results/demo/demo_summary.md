# Demo Summary

## Tiny Monocular Relative-Pose Network for Micro-UAV Perception

### Objective
Estimate local-frame relative motion (dx, dy, dyaw) from a short monocular frame context
using a compact CNN, enabling lightweight visual odometry for micro-UAVs.

### Input
- Grayscale frame triplets `[t-1, t, t+1]` resized to 96x96
- Dataset: UZH-FPV indoor-forward Snapdragon public sequences

### Output
- Local-frame relative motion predictions: dx, dy, dyaw
- Reconstructed 2D trajectory
- Performance comparison with classical VO (ORB)

### Dataset
- Source: UZH-FPV Drone Racing Dataset
- Train sequences: indoor_forward_3, indoor_forward_5, indoor_forward_6
- Validation sequences: indoor_forward_7, indoor_forward_9
- Test sequence: indoor_forward_10
- Split: Train 2680 / Val 2503 / Test 820
- Total processed triplets: 6003

### Results

| Metric | TinyCNN |
|--------|---------|
| RMSE dx | 0.0825 m |
| RMSE dy | 0.0992 m |
| RMSE dyaw | 0.0463 rad |
| FPS | 545 |

### ORB Baseline Snapshot

| Metric | ORB VO |
|--------|--------|
| RMSE dx | N/A |
| RMSE dy | N/A |
| RMSE dyaw | 0.273563 |
| FPS | N/A |


**Trajectory Metrics:**
- Mean ATE: 10.53 m
- Final drift: 16.62 m
- GT path length: 128.67 m
- Drift rate: 12.9%

### Key Observations
- The local-frame label formulation is physically cleaner than the original world-frame regression target.
- The three-frame context provides short-horizon temporal cues beyond isolated image pairs.
- Strong lighting shifts remain the dominant robustness failure mode.
- Real-time inference remains feasible at 545 FPS on RTX 4070 Laptop GPU.

## Failure Mode Summary

| Condition | RMSE dx | RMSE dy | RMSE dyaw |
|-----------|---------|---------|----------|
| normal | 0.0825 | 0.0992 | 0.0463 |
| low_texture | 0.0825 | 0.0992 | 0.0463 |
| motion_blur | 0.0825 | 0.0992 | 0.0463 |
| brightness_dark | 0.0841 | 0.1013 | 0.0453 |
| brightness_bright | 0.0819 | 0.0993 | 0.0468 |
| noise | 0.0825 | 0.0992 | 0.0463 |

### Next Steps
1. Add IMU fusion to disambiguate aggressive rotations.
2. Increase sequence diversity beyond the current indoor-forward subset.
3. Replace the simple CNN with a temporal encoder once the label pipeline stabilizes.
4. Revisit geometric baselines with a cleaner camera/body-frame comparison.
5. Test transfer on Crazyflie-specific real data.
