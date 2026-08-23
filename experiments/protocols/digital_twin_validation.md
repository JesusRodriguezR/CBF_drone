# Digital Twin Validation Protocol

Step-by-step workflow to validate the Crazyflie 2.1 digital twin against physical flights.

---

## Equipment Required

| Item | Phase 1 | Phase 2 | Phase 3 |
|---|:---:|:---:|:---:|
| Crazyflie 2.1 + battery | ✅ | ✅ | ✅ |
| Crazyradio PA | ✅ | ✅ | ✅ |
| Game controller | — | optional | optional |
| **OptiTrack** | ❌ | ✅ | ✅ |
| Flow deck v2 | ❌ | ❌ | ❌ (OptiTrack provides position) |

---

## Phase 1 — IMU Noise Characterization

**Goal:** Validate `sensor_model.py` noise parameters against real BMI088 measurements.
**OptiTrack: NOT needed.**

### Steps

1. **Static IMU capture**
   Place CF on a flat, level, rigid surface. **Do not move it.**
   ```bash
   python exp_sensor_logger_v1.py --mission hover --duration 0 --dry-run
   # But for real collection, we need a "static" mode, so just log while CF is off:
   # Manually log with cflib for 60s with motors OFF — see below.
   ```
   > Actually, use the sensor logger with a custom script (no takeoff) or just connect
   > and start logging without launching a mission. You can modify the script to add
   > `--mission static` which skips the MotionCommander entirely.

2. **Collect 60 seconds** of IMU data at 1000 Hz (or 100 Hz from the log).

3. **Compute from real data:**
   - Accelerometer noise density σ_a (from PSD or std of stationary signal)
   - Gyroscope noise density σ_g
   - Bias drift rate (from Allan variance or sliding window mean)

4. **Compare to sim parameters:**
   | Parameter | Sim Value | Real Value | Match? |
   |---|---|---|---|
   | `accel_noise_density` | 150 µg/√Hz | measure | |
   | `gyro_noise_density` | 0.014 °/s/√Hz | measure | |
   | `accel_bias_instability` | 40 mg | measure | |
   | `gyro_bias_instability` | 1 °/s | measure | |

5. **Update `crazyflie_params.py`** if values differ significantly (>20%).

### Pass criteria
- Noise density within ±30% of datasheet values.
- Allan variance curves have similar shape/knee point.

---

## Phase 2 — Open-Loop Dynamics Validation

**Goal:** Validate rigid body dynamics, motor model, and aerodynamics.
**OptiTrack: REQUIRED** — provides ground-truth position & attitude at 120+ Hz.

### OptiTrack Setup
1. Create a rigid body for the CF in Motive (use reflective markers on the CF).
2. Stream rigid body pose via NatNet to the PC.
3. Option A: Stream to CF via `locSrv` for onboard position estimate.
4. Option B: Log Motive CSV separately and time-sync in post-processing (recommended for validation since it avoids coupling).

### Experiments

#### 2A. Hover Stability
```bash
python exp_sensor_logger_v1.py --mission hover --duration 15 --altitude 0.5
```
Then run the **same hover** in simulation:
```bash
cd simulations
python run_digital_twin.py   # hover test at 1m (adjust for 0.5m)
```
**Compare:** altitude stability, euler angle drift, motor PWMs.

#### 2B. Step Response
Apply a pitch/roll step via gamepad (small, ~5° for 0.2s):
```bash
python exp_gamepad_flight_v1.py
```
Record the step input and angular rate response.
**Compare:** angular rate time constant, peak overshoot, settling time vs sim disturbance test.

#### 2C. Free Fall (optional, careful!)
From a low hover (~0.3m), cut thrust and let it drop.
**Compare:** impact time vs `run_digital_twin.py` free-fall scenario (adjust starting altitude).

### Analysis
```bash
cd experiments/exp_analysis
python compare_sim_vs_real.py --sim ../simulations/simulation_results/hover_log.npz --real ../exp_results/sensor_log_hover_XXXX.csv
```

### Pass Criteria
- Position RMSE < 5 cm over a 10s hover.
- Attitude time constant within ±20% of sim.
- Motor PWM distributions overlap significantly.

---

## Phase 3 — Closed-Loop Trajectory Validation

**Goal:** Validate the full digital twin with position controller in the loop.
**OptiTrack: REQUIRED** — ground truth AND position feedback for the CF.

### OptiTrack Setup
- Stream to CF via `locSrv` so its onboard EKF uses OptiTrack as a position source.
- Also record Motive CSV for post-analysis.

### Experiments

#### 3A. Square Waypoint Tracking
```bash
python exp_sensor_logger_v1.py --mission square --duration 60
```
Run the **same square** in simulation (add waypoint mission to `run_digital_twin.py`).
**Compare:** XY trajectory overlay, tracking error histogram.

#### 3B. Aggressive Maneuver
Use gamepad to fly a figure-8 or rapid altitude change:
```bash
python exp_gamepad_flight_v1.py
```
Replay the **recorded commands** into the simulation to compare outputs.

### Pass Criteria
- Trajectory RMSE < 10 cm for square mission.
- Spectral content of attitude signals matches between sim and real (within 1 octave).
- Command-to-response latency agrees within ±5 ms.

---

## Summary: When You Need OptiTrack

| Task | OptiTrack? | Why |
|---|---|---|
| IMU noise characterization | ❌ | Only comparing onboard sensor noise |
| Static barometer check | ❌ | Known altitude reference is enough |
| Hover altitude stability | ✅ | Need ground-truth altitude (baro drifts) |
| Step response dynamics | ✅ | Need ground-truth attitude angles |
| Free fall trajectory | ✅ | Need ground-truth position |
| Waypoint tracking | ✅ | Need ground-truth XY + position feedback |
| Gamepad flight replay | ✅ | Need ground-truth for comparison |

---

## Post-Validation Actions

1. **If sim matches real** (all pass criteria met):
   - Tag the sim parameters as "validated v1" in git.
   - Document the validated flight envelope (max angle, max speed tested).

2. **If sim diverges:**
   - Identify which subsystem diverges (plot overlay will show it).
   - Tune the specific parameter (motor time constant, drag coefficient, noise density).
   - Re-run the relevant phase to confirm the fix.
   - Iterate until convergence.
