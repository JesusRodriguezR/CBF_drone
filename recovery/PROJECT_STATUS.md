# Crazyflie Gazebo Project Status

**Status date:** 2026-10-28  
**Current platform:** ROS 2 Humble with Gazebo Fortress / Ignition Gazebo 6

**Next planned direction:** mixed-reality photorealistic sensing using the real
Crazyflie and mocap pose, Gazebo as the ROS/digital-twin layer, and Unreal
Engine as the virtual-crop renderer.

## Current project position

The project has a working simulated Crazyflie inspection pipeline in a greenhouse environment. The simulation launches successfully, the drone takes off under Gazebo's multicopter controller, and the odometry-driven finite-state machine (FSM) begins traversing the valid greenhouse aisles. The active ROS workspace also includes a configurable camera-network impairment layer and GUI for testing delayed, jittered, and lost image deliveries.

It now also includes a passive single-frame tutor-wire perception characterization pipeline. This consumes `/crazyflie/camera/image_delayed`, logs perception and timing data, and does not influence the existing FSM or flight controller.

The current system is best described as an **odometry-based autonomous flight and aisle-traversal prototype**. It is not yet a complete perception-driven crop-inspection system: the camera and sensors are simulated and available, but the FSM currently uses simulated odometry rather than camera perception to make navigation decisions.

## What has been implemented

### Greenhouse world

- The default launch starts `greenhouse_world.sdf`.
- The world contains 20 crop-row models, named `crop_row_01` through `crop_row_20`.
- There are 19 traversable internal aisles.
- The twentieth computed path is intentionally excluded because its centerline would lie on the north wall at `y = 15.0 m`.
- Each crop row is currently a static rectangular box with one visual and one matching collision.
- Procedural stems, leaves, crop overlays, visual-fidelity controls, and the selector GUI plugin were removed from the active world.

### Flight control

- Gazebo's `MulticopterVelocityControl` provides the inner attitude, angular-rate, and velocity stabilization loop.
- The Crazyflie model includes attitude and angular-rate gains plus four motor-model plugins.
- The custom `control_services` ROS node provides bounded takeoff, hover-height, landing-altitude PID behavior, and horizontal/yaw command handling.
- An odometry-based outer lateral position correction keeps the vehicle near the active FSM lane target.
- World-frame FSM velocities are converted to the body frame expected by the Crazyflie/Gazebo velocity interface.
- Odometry-based yaw hold targets the forward heading while airborne.

### FSM behavior

- The FSM supports 19 valid internal aisles.
- Lane headings alternate between `0 rad` eastbound and `pi rad` westbound.
- The drone turns 180 degrees at headlands so that the camera faces the direction of travel on every lane.
- The active trial reports states such as `ENTER_LANE`, `IN_LANE`, `APPROACH_MONITOR_POINT`, and `OBSERVE` and remains odometry-driven.
- The monitoring extension loads configurable observation points, stops at each point, commands a side-looking yaw, waits for a configured duration/minimum frame count, and resumes traversal.

### Monitoring mission

- The default first scenario contains three deterministic points in aisle 1, configured in `ros_gz_crazyflie_control/config/monitoring_points.json`.
- Each point specifies aisle, longitudinal position, crop side, observation yaw offset, and crop-section ID.
- The FSM publishes a shared monitoring event ID and point metadata through `/greenhouse/fsm_state`.
- The perception logger preserves event/point/section IDs, capture and arrival timestamps, AoI, vehicle pose/yaw, desired yaw, detector output, and tutor-wire ground truth for every frame.
- `tools/analyze_monitoring.py` summarizes frames, detection rate, distance estimates, MAE/RMSE, confidence, observation duration, and AoI per event, with SVG plots.

### Simulated sensing and ROS interfaces

- IMU simulation: 1000 Hz, bridged to `/crazyflie/imu`.
- FlowDeck v2 simulation: publishes `/crazyflie/flow` and `/crazyflie/flow/velocity`.
- Downward range channel: simulated at 30 Hz and bridged to `/crazyflie/flow/range`.
- AI-deck camera: 320 x 320 pixels at 30 Hz, bridged to `/crazyflie/camera/image`.
- Odometry, TF, lidar, clock, command velocity, IMU, FlowDeck range, and camera bridges are configured in `ros_gz_crazyflie_bringup/config/ros_gz_crazyflie_bridge.yaml`.

### Camera-network impairment testing

- The `image_impairment` node subscribes to `/crazyflie/camera/image` and republishes an impaired stream on `/crazyflie/camera/image_delayed`.
- The impairment layer supports disabled or fixed delay, uniform or Gaussian jitter, configurable packet-loss probability, and rejection of stale/out-of-order frames.
- It publishes live status on `/crazyflie/camera/impairment_status` and records per-frame delivery data, configured delay, measured delay, age of information, and drop status to CSV.
- The `image_impairment_gui` Tk window provides live controls for delay, jitter, packet loss, frame ordering, and statistics reset.
- The simulation launch file includes both nodes and defaults `impairment_gui` to `True`.

### Passive perception characterization

- The greenhouse already contains lightweight tutor-wire geometry with known row, height, and longitudinal positions.
- `baseline_perception` independently detects vertical wire contrast from each impaired camera frame.
- It logs frame IDs, original/arrival/processing timestamps, AoI, detection validity, image position, estimated distance, confidence, wire ground truth, perception error, odometry pose, FSM state, aisle, and velocity.
- `tools/analyze_perception.py` generates SVG plots and CSV/JSON summary metrics for detection success, distance error, AoI, AoI/error correlation, and confidence/error correlation.
- Repeatable baseline, fixed-delay, jitter, packet-loss, and combined conditions are listed in `perception_experiments.yaml`.

## What has been verified

- The workspace was rebuilt successfully before the current clean run.
- Gazebo starts with the greenhouse world.
- The bridge, custom controller, FlowDeck simulator, and camera viewer start from the launch file.
- The camera topic was previously verified at approximately 30 Hz.
- The current run started the Gazebo simulation, visualizer, and FSM trial successfully.
- During the current run, the controller reported `Takeoff completed`.
- A clean one-aisle monitoring run completed all three observation events and landed successfully, producing 7,186 perception frames; events contained 135, 138, and 135 frames.
- A full-bundle one-aisle run completed all three observation events under uniform camera jitter (`80 ms` base delay, `25 ms` jitter, no packet loss) and landed successfully; events contained 123, 126, and 128 tagged frames.
- The full-bundle run included Gazebo, camera viewer, impairment node/GUI, baseline perception, FSM visualizer, and monitoring FSM.

## Current limitations and next work

- Navigation and lane following currently use perfect simulated odometry; camera-based perception has not yet been integrated into the FSM.
- The crop rows are simplified box geometry rather than detailed plant models.
- The system has not yet been demonstrated on real Crazyflie hardware.
- A complete quantitative evaluation still needs recorded trajectories, lane-tracking error, yaw error, and multi-condition perception comparisons.
- The current monitoring policy is fixed stop-and-observe; adaptive viewpoint selection, temporal filtering, structure-aware inference, and uncertainty calibration remain future work.
- The full-bundle log is currently at `/tmp/perception_monitoring_full.csv` and is ephemeral; copy logs to a persistent experiment directory for later analysis.
- The baseline perception node and analysis scripts are available, but multi-condition experiment results still need to be collected systematically.

### Planned mixed-reality extension

The next major phase is to fly the real Crazyflie inside the mocap sensing
area and use the measured pose to drive both a Gazebo digital twin and an
Unreal virtual camera. The real camera image will be combined with Unreal's
virtual crop render and published as a mixed image for the existing perception
pipeline.

Planned data flow:

```text
real Crazyflie + mocap
        ↓
      ROS 2
   ┌────┼─────────────┐
   ↓    ↓             ↓
Gazebo UE virtual   real camera
twin   crop render       │
   └────┬───────────────┘
        ↓
 image compositor
        ↓
 /crazyflie/camera/image_mixed
        ↓
 existing perception + monitoring logger
```

The first implementation should keep Gazebo authoritative for the simulated
digital twin but use UE for photorealistic rendering. UE should not introduce a
second physics authority. The real drone should remain authoritative during
hardware flights.

Before setup resumes, storage should be reorganized. The current Ubuntu root
has approximately 37 GB free and `/data` approximately 52 GB free. The planned
layout is to keep ROS/Gazebo on Ubuntu and place the Unreal Engine installation,
UE project, and asset cache under `/data` or a larger Linux partition. Do not
resize or move dual-boot partitions without a backup and an explicit partition
layout check.

The other prerequisite is graphics support: NVIDIA hardware is present, but
the NVIDIA driver was not active during the last check. Verify the driver
before installing UE. Prefer a precompiled Linux Unreal build first; avoid a
full source build until storage and GPU support are confirmed.

## How to run the current system

Start Gazebo and the supporting nodes:

```bash
source /opt/ros/humble/setup.bash
source /home/yisus/crazyflie_gz_test_ws/install/setup.bash
export GZ_SIM_RESOURCE_PATH=/home/yisus/crazyflie_simulation_models/crazyflie-simulation/simulator_files/gazebo:${GZ_SIM_RESOURCE_PATH:-}
ros2 launch ros_gz_crazyflie_bringup crazyflie_simulation.launch.py
```

The launch file also starts the camera impairment node and its configuration window by default. To control this explicitly, use `impairment_gui:=true` or `impairment_gui:=false`.

In a second terminal, start the top-down monitor:

```bash
source /opt/ros/humble/setup.bash
source /home/yisus/crazyflie_gz_test_ws/install/setup.bash
ros2 run ros_gz_crazyflie_control greenhouse_fsm_visualizer
```

In a third terminal, start the odometry-based FSM trial:

```bash
source /opt/ros/humble/setup.bash
source /home/yisus/crazyflie_gz_test_ws/install/setup.bash
ros2 run ros_gz_crazyflie_control greenhouse_fsm_trial
```

Do not start a second Gazebo launch or a second `control_services` node. Duplicate controllers or worlds can produce ambiguous odometry and command behavior.

## Recommended project description

> We have a working ROS 2 Humble and Gazebo Fortress simulation of a Crazyflie inspecting a simplified greenhouse. The system includes a 20-row greenhouse world, simulated IMU, FlowDeck range/optical-flow sensing, lidar, odometry, TF, and a 30 Hz AI-deck camera. Gazebo provides the inner multicopter stabilization, while a custom ROS controller handles bounded takeoff, altitude control, yaw, and odometry-based lateral correction. A 19-aisle FSM autonomously commands alternating eastbound and westbound traversals with corresponding 180-degree headland turns. The current prototype is operational in simulation and has reached takeoff and lane traversal, but it still uses simulated odometry for navigation; camera-based perception, full-run quantitative evaluation, and hardware validation remain to be completed.

