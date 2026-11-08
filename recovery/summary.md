# Project Summary

This project simulates a Crazyflie micro-drone inspecting a simplified greenhouse using ROS 2 Humble and Gazebo Fortress.

The simulation provides:

- A greenhouse world containing 20 crop rows and 19 traversable aisles.
- Crazyflie takeoff, hover, landing, velocity, altitude, and yaw control.
- An autonomous finite-state machine that alternates eastbound and westbound aisle traversals.
- Simulated IMU, FlowDeck optical-flow and range sensing, lidar, odometry, TF, and AI-deck camera data.
- A camera-network impairment layer that can add fixed delay, jitter, packet loss, and out-of-order frame rejection.
- A GUI for changing impairment settings live and monitoring delay, age of information, received frames, dropped frames, and loss percentage.
- A passive single-frame tutor-wire perception node that consumes the impaired stream without controlling the drone.
- Machine-readable perception logs and analysis scripts for detection rate, distance error, AoI, and confidence relationships.
- A visualizer for the planned path, drone position, FSM state, lane target, lane error, and commands.
- A monitoring mission layer with configurable observation points, side-looking yaw actions, temporal event tagging, and resume-after-observation behavior.
- Monitoring-event analysis for frame counts, detection rate, tutor-wire distance error, confidence, observation duration, and AoI.

Gazebo supplies the inner multicopter stabilization, while custom ROS nodes manage the flight behavior and aisle-traversal mission. The current navigation system uses simulated odometry to follow lane targets, and the system has been verified to launch, take off, and enter the first lane.

The camera impairment node receives the raw camera stream and publishes a separate delayed/impaired stream for perception experiments. Its configuration window is included in the simulation launch and supports fixed delay, uniform or Gaussian jitter, configurable packet-loss probability, stale-frame rejection, and live statistics.

The passive perception experiment consumes `/crazyflie/camera/image_delayed`, detects the known tutor-wire geometry independently in each frame, and logs timestamps, AoI, estimates, ground truth, errors, odometry, FSM state, and velocity. It is intentionally disconnected from the FSM and flight controller.

The monitoring mission now inserts `APPROACH_MONITOR_POINT` and `OBSERVE` into the existing aisle FSM. The default aisle-1 scenario uses three points from `monitoring_points.json`; each observation holds position, turns toward the configured crop side, acquires a short sequence, and resumes traversal. Frames share a `monitoring_event_id` and include point, aisle, crop-section, pose, yaw, AoI, detector, and ground-truth fields.

The complete one-aisle bundle was demonstrated with Gazebo, camera viewer, impairment node and GUI, baseline perception, visualizer, and FSM. Under uniform camera jitter with 80 ms base delay and 25 ms jitter (no packet loss), all three monitoring events completed with 123, 126, and 128 tagged frames, followed by landing. The resulting log was `/tmp/perception_monitoring_full.csv`.

The project remains an odometry-driven autonomous flight and structured-observation prototype rather than a final perception-driven crop-inspection system. Camera-based navigation, temporal filtering, structure-aware inference, uncertainty calibration, adaptive viewpoint selection, and real Crazyflie validation remain future work.

## Next planned phase: mixed reality

The proposed next phase is to fly the real Crazyflie inside the mocap sensing
area. Mocap will publish the real drone pose to ROS 2; Gazebo will mirror that
pose as a digital twin, while Unreal Engine will render virtual crops from the
matching camera viewpoint. The real camera image and UE virtual-crop render
will then be composited into `/crazyflie/camera/image_mixed` for the existing
perception and monitoring logger.

The intended architecture has one physics authority: Gazebo for simulation and
the real drone for hardware flights. UE is initially a render-only virtual
environment, not a second flight simulator. The first milestone is a stationary
or slow-hovering drone, one calibrated UE crop asset, pose-synchronized
rendering, and a correctly aligned mixed image.

Work is paused until storage is reorganized. Current free space is about 37 GB
on Ubuntu root and 52 GB on `/data`; the preferred layout is ROS/Gazebo on
Ubuntu and Unreal Engine, projects, and asset caches under `/data` or a larger
Linux partition. The NVIDIA GPU is present but its driver was not active during
the last check, so GPU setup must be verified before Unreal installation.
