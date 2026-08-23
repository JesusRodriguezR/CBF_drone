# Sim-to-Real Validation Findings

This document summarizes the insights and findings from the Phase 1 and Phase 2 experiments conducted to validate the Crazyflie 2.1 digital twin simulation against real physical hardware.

---

## Stage 1: IMU Noise Characterization (Static Test)

**Objective:** Validate that the simulated IMU's noise characteristics (white noise density and bias random walks) match the physical accelerometer and gyroscope. 

**Method:** 60 seconds of stationary log data collected at 100 Hz.

### Key Findings
1. **Accelerometer Noise Density Mismatch:**
   - **Measured:** 0.000155 m/s²/√Hz
   - **Simulated:** 0.001471 m/s²/√Hz
   - **Insight:** The simulation was modeled directly from the BMI088 datasheet specifications (150 µg/√Hz). However, the real Crazyflie firmware applies internal low-pass filtering before outputting the logs. This made the simulation **~10x too noisy**.
   - **Action:** Scaled down `accel_noise_density` in `crazyflie_params.py` by a factor of 10 to match reality.
2. **Gyroscope:**
   - **Measured:** 0.000297 rad/s/√Hz vs. **Simulated:** 0.000244 rad/s/√Hz. 
   - **Insight:** Good match! The simulated gyro matched the real hardware without major adjustments.
3. **Biases:**
   - Both accelerometer and gyro drift biases were found to be well within modeled stability limits. 

---

## Stage 2: Open-Loop Hover Dynamics

**Objective:** Compare real-world hover dynamics against the digital twin by correlating onboard IMU streams with external OptiTrack positional ground-truth. 

**Method:** An autonomous ~15-second PID-controlled hover test targeting 0.5m altitude.

### Key Findings
1. **Altitude Stability and Base Thrust ([PASS / ISSUE])**
   - The drone demonstrated excellent stability holding altitude with an **RMSE of 1.73 cm** (well below the 5 cm pass criteria).
   - However, the actual hover altitude plateaued at ~0.05m instead of the target 0.5m. The PID loop maintained stability, but the baseline thrust compensation or takeoff ramp was systematically too low to push the drone up into free-air.
   - **Action:** Base thrust coefficients require tuning to prevent ground-effect trapping.
2. **Attitude and Yaw Drift ([PASS / EXPECTED])**
   - Pitch and roll were rock solid with a standard deviation of ~0.6 degrees.
   - Yaw drifted locally over the 15-second flight (~5-10 degrees).
   - **Insight:** This was expected as the baseline testing PID loop did not actively hold magnetometer/vision-based yaw. OptiTrack tracking and onboard attitude matched perfectly.
3. **In-Flight Vibration Noise ([MISSING])**
   - While static IMU noise matched perfectly in Stage 1, turning the motors on resulted in massive harmonic vibration peaks for both the Gyroscope (at 5, 10, and 25 Hz) and the Z-axis Accelerometer (~15x higher noise than static).
   - **Insight:** The simulation only models white noise, entirely omitting motor-induced acoustic vibration characteristics. 
   - **Action:** A motor vibration harmonic noise model must be added to `sensor_model.py` for high-fidelity control testing.
4. **Motor PWM Mapping Discrepancy ([MISMATCH])**
   - **Measured Hover PWM:** ~39,817 (out of 65,535; ~60.8% duty cycle)
   - **Insight:** The simulation was structured around normalized PWM fractions (0.0 to 1.0) and theoretical mappings that vastly mismatched the real 16-bit operational scale.
   - **Action:** Recalibration of the structural thrust curves (`pwm_thrust_a` and `pwm_thrust_b` parameters) is required to map empirical real-world bit-scales to Newtonian thrust.
