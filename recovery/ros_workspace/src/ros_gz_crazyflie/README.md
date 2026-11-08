# ros-gz-crazyflie

> Disclaimer: This project is experimental and prone to errors as it is not regularly maintained. Please start a ticket if you encounter any issues but do expect limited support

> The modules for the multiranger, like the simple mapper and wall following are moved to: https://github.com/knmcguire/crazyflie_ros2_multiranger

Tested on:
* ROS 2 Humble with Gazebo Harmonic (2024/09/02)
* ROS 2 Jazzy with Gazebo Harmonic (2025/06/02)
* ROS 2 Ionic with Gazebo Ionic (2025/06/02)

## Usage

Make a workspace and clone this gazebo-ros crazyflie simulation repo

    mkdir ~/ros2_ws/src
    cd ~/ros2_ws/src
    git clone https://github.com/knmcguire/ros_gz_crazyflie


Clone the crazyflie simulation repo and source the crazyflie model

    mkdir ~/simulation_models/
    cd ~/simulation_models/
    git clone git@github.com:bitcraze/crazyflie-simulation.git
    export GZ_SIM_RESOURCE_PATH=~/simulation_models/crazyflie-simulation/simulator_files/gazebo/"

Build the ros workspace with

    cd  ~/ros2_ws/
    colcon build --cmake-args -DBUILD_TESTING=ON

Run with

    source install/local_setup.bash
    ros2 launch ros_gz_crazyflie_bringup crazyflie_simulation.launch.py


In a different terminal, open ROS 2 teleop twist keyboard node with

    source /opt/ros/DISTRO/setup.bash
    ros2 run teleop_twist_keyboard teleop_twist_keyboard

Take off with pressing 't' and then control the crazyflie with the keyboard according to the teleop_twist_keyboard instructions.

## Passive perception characterization

The greenhouse world already contains lightweight tutor wires: five rows at
`y = -3.0, -1.5, 0.0, 1.5, 3.0 m`, with vertical wires from `z = 0.65` to
`z = 2.25 m` and known `x` positions spaced by 0.30 m. The passive baseline
detector uses these positions as ground truth; it does not modify the FSM,
controller, or camera impairment node.

The launch file starts the detector by default. It subscribes to the impaired
stream `/crazyflie/camera/image_delayed`, detects vertical contrast in each
frame independently, and writes a CSV containing timestamps, AoI, detection
validity, image position, distance estimate, wire ground truth, error, odometry,
FSM state, aisle, and velocity. The raw stream is never used for perception.

Start a baseline run:

```bash
source /opt/ros/humble/setup.bash
source /home/yisus/crazyflie_gz_test_ws/install/setup.bash
export GZ_SIM_RESOURCE_PATH=/home/yisus/crazyflie_simulation_models/crazyflie-simulation/simulator_files/gazebo:${GZ_SIM_RESOURCE_PATH:-}
ros2 launch ros_gz_crazyflie_bringup crazyflie_simulation.launch.py \
  perception_log_path:=/tmp/perception_baseline.csv
```

The existing impairment GUI opens with the launch. Apply one of the repeatable
conditions in `perception_experiments.yaml` for a degraded run, then use a new
log path, for example:

```bash
ros2 launch ros_gz_crazyflie_bringup crazyflie_simulation.launch.py \
  perception_log_path:=/tmp/perception_combined.csv
```

The same degraded condition can be applied without the GUI. For example, a
100 ms fixed-delay plus 10% packet-loss run is:

```bash
ros2 param set /image_impairment delay_mode fixed
ros2 param set /image_impairment base_delay_ms 100.0
ros2 param set /image_impairment packet_loss_enabled true
ros2 param set /image_impairment packet_loss_probability 10.0
```

For a jitter run, use `delay_mode:=jitter`, set `base_delay_ms` and `jitter_ms`,
and choose `jitter_distribution` as `uniform` or `gaussian`.

Generate plots and summary statistics:

```bash
python3 tools/analyze_perception.py \
  --input /tmp/perception_baseline.csv \
  --impairment-log /tmp/crazyflie_image_impairment.csv \
  --output-dir /tmp/perception_baseline_analysis
```

The analysis directory contains detection success, distance-vs-ground-truth,
error-vs-time, AoI-vs-time, AoI-vs-error, confidence-vs-error plots, plus
`summary.csv` and `summary.json`. The impairment node's own CSV remains at
`/tmp/crazyflie_image_impairment.csv` unless its `log_path` parameter is changed.

## Monitoring mission

The FSM loads three deterministic monitoring points from the installed
`ros_gz_crazyflie_control/config/monitoring_points.json` file. Each point has
an aisle, longitudinal position, crop side, observation yaw offset, and crop
section identifier. The mission inserts `APPROACH_MONITOR_POINT` and `OBSERVE`
between `IN_LANE` traversal and the existing headland transition. During
`OBSERVE`, the vehicle stops horizontally, turns to the configured yaw, holds
for the configured duration, and the passive detector tags all arriving frames
with the same monitoring event ID.

After starting Gazebo and the supporting nodes, run the one-aisle scenario:

```bash
source /opt/ros/humble/setup.bash
source /home/yisus/crazyflie_gz_test_ws/install/setup.bash
ros2 run ros_gz_crazyflie_control greenhouse_fsm_visualizer
```

In another terminal:

```bash
source /opt/ros/humble/setup.bash
source /home/yisus/crazyflie_gz_test_ws/install/setup.bash
ros2 run ros_gz_crazyflie_control greenhouse_fsm_trial --ros-args \
  -p n_lanes:=1 -p observation_duration:=1.5
```

The existing perception CSV (for example `/tmp/perception_monitoring.csv`)
now includes the event ID, point/aisle/section IDs, observation flag, pose,
yaw, AoI, detector output, and ground truth. Analyze the event sequences with:

```bash
python3 tools/analyze_monitoring.py \
  --input /tmp/perception_monitoring.csv \
  --output-dir /tmp/monitoring_analysis
```

The analysis writes per-event CSV/JSON summaries and distance/validity SVG
plots.


