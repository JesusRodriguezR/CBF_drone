# Dataset Format Notes - UZH-FPV Drone Racing Dataset

## Source

- Dataset: UZH-FPV Drone Racing Dataset
- Website: https://fpv.ifi.uzh.ch/
- Paper: "Are We Ready for Autonomous Drone Racing?" (ICRA 2019)
- License: CC BY-NC-SA 3.0

## Public Sequences Used

- Training: `indoor_forward_3_snapdragon_with_gt`, `indoor_forward_5_snapdragon_with_gt`, `indoor_forward_6_snapdragon_with_gt`
- Validation: `indoor_forward_7_snapdragon_with_gt`, `indoor_forward_9_snapdragon_with_gt`
- Testing: `indoor_forward_10_snapdragon_with_gt`
- Sensor: Qualcomm Snapdragon Flight board (stereo cameras + IMU)
- Environment: indoor forward-facing racing trajectories

## Data Format

### Ground Truth

- Format: space-separated text file
- Columns: `timestamp tx ty tz qx qy qz qw`
- Units: seconds (timestamp), meters (translation)
- Quaternion: Hamilton convention, `w` component last
- Frame: IMU pose in world frame (`T_wi`)
- Source: Leica Nova MS60 laser tracker

### Images

- Sensor: Snapdragon stereo cameras (we use left images)
- Format: PNG files
- Naming: nanosecond timestamps as filenames
- Original resolution: about `640x480`
- Processed resolution: `96x96` grayscale

### IMU

- Format: text file
- Columns: `timestamp gx gy gz ax ay az`
- Units: rad/s (gyro), m/s^2 (accelerometer)
- Rate: about `200 Hz`

## Coordinate Convention

- World frame: fixed reference
- Body frame: IMU-centered
- Poses specify `T_wi` (world-from-IMU transformation)
- Translation in meters
- Rotation as unit quaternion

## Processing Notes

- Images are converted to grayscale and resized to `96x96`
- Samples are built as frame triplets `[t-1, t, t+1]`
- Relative translation from `t` to `t+1` is rotated into the local/body frame at time `t`
- Yaw target is `dyaw = wrap(yaw_{t+1} - yaw_t)`
- Yaw is extracted from quaternion with `atan2(2*(qw*qz + qx*qy), 1 - 2*(qy^2 + qz^2))`
- Image-to-GT matching uses the nearest timestamp within a `50 ms` threshold
- Processed frame files are flattened into `data/processed/images/` with sequence-prefixed names
- Train, validation, and test are split by full sequence, not by time slices within one sequence
