# Reinstall and restore guide

This directory preserves the active Crazyflie greenhouse ROS source state as
of 2026-11-08. The active source originally lived outside this repository at:

```text
/home/yisus/crazyflie_gz_test_ws/src/ros_gz_crazyflie
```

The copied source includes the uncommitted monitoring-mission, greenhouse,
bridge, perception, impairment, and visualizer changes. Generated `build/`,
`install/`, `log/`, Python caches, and nested `.git` metadata are intentionally
excluded and should be regenerated.

## Preserved contents

- `ros_workspace/src/ros_gz_crazyflie/`: complete active ROS source tree.
- `external/crazyflie_simulation_models/crazyflie-simulation/`: local
  Crazyflie Gazebo model dependency used by `GZ_SIM_RESOURCE_PATH`.
- `PROJECT_STATUS.md` and `summary.md`: project status and next-phase
  mixed-reality proposal.

The source repositories were based on these commits before local changes:

```text
ros_gz_crazyflie:       e2792d0727a22aba3cd8e67f018d95c95f76c487
crazyflie-simulation:  7e93752dbc803af2488c1db46bb79b3da55f5d8c
```

## Restore after reinstall

From the restored copy:

```bash
cd /path/to/vgreen/recovery/ros_workspace
mkdir -p src
# The source is already in src/ros_gz_crazyflie.

source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

Set the model path using the restored dependency:

```bash
export GZ_SIM_RESOURCE_PATH=/path/to/vgreen/recovery/external/crazyflie_simulation_models/crazyflie-simulation/simulator_files/gazebo:${GZ_SIM_RESOURCE_PATH:-}
```

If the restored project is moved elsewhere, update that path rather than
copying the build or install directories.

## Run the current monitoring scenario

```bash
ros2 launch ros_gz_crazyflie_bringup crazyflie_simulation.launch.py \
  camera_view:=false impairment_gui:=false \
  perception_log_path:=/tmp/perception_monitoring.csv
```

In another terminal:

```bash
source /opt/ros/humble/setup.bash
source /path/to/vgreen/recovery/ros_workspace/install/setup.bash
ros2 run ros_gz_crazyflie_control greenhouse_fsm_visualizer
```

In a third terminal:

```bash
ros2 run ros_gz_crazyflie_control greenhouse_fsm_trial --ros-args \
  -p n_lanes:=1 -p observation_duration:=1.5 \
  -p observation_min_frames:=3
```

For the full bundle, use `camera_view:=true impairment_gui:=true`. The
impairment node can then be configured live, for example:

```bash
ros2 param set /image_impairment delay_mode jitter
ros2 param set /image_impairment base_delay_ms 80.0
ros2 param set /image_impairment jitter_ms 25.0
```

The default monitoring points are in:

```text
ros_gz_crazyflie_control/config/monitoring_points.json
```

Analyze a run with:

```bash
python3 tools/analyze_monitoring.py \
  --input /tmp/perception_monitoring.csv \
  --output-dir /tmp/monitoring_analysis
```

## Planned UE/mixed-reality phase

Do not install Unreal until storage and GPU setup are ready. The intended
architecture is:

```text
real drone + mocap → ROS 2 → Gazebo digital twin
                         ↘ UE virtual crops
real camera + UE render → image compositor → /crazyflie/camera/image_mixed
```

Keep one physics authority: Gazebo for simulation, the real drone for hardware
flight. UE should initially be render-only. The preferred future layout is to
keep ROS/Gazebo on Ubuntu and place Unreal Engine, UE projects, and asset caches
on `/data` or a larger Linux partition. Verify the NVIDIA driver before UE
installation. Do not resize dual-boot partitions without a backup and a fresh
partition-layout check.

## Important omissions

- No user-created Blender greenhouse assets were found under `/home/yisus`
  during this backup. The only `.blend` files found were the upstream
  Crazyflie model meshes, which are included in `external/`.
- Runtime CSV logs under `/tmp` were not present at backup time.
- `build/`, `install/`, and `log/` are intentionally not portable source
  artifacts; rebuild them after reinstall.
