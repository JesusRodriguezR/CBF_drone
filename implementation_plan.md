# Crazyflie 2.1 Digital Twin Simulation

Build a high-fidelity digital twin of the Crazyflie 2.1 quadrotor inside `simulations/`. The simulation models full 6-DOF rigid body dynamics, realistic motor dynamics with the firmware-accurate mixing matrix, aerodynamic effects, and sensor noise — enabling sim-to-real transfer of controllers.

## Proposed Changes

### Physical Parameters Module

#### [NEW] [crazyflie_params.py](file:///c:/digital%20model/simulations/crazyflie_params.py)

Centralized dataclass with all Crazyflie 2.1 constants sourced from official Bitcraze specs and published system identification:

| Parameter | Value | Source |
|---|---|---|
| Mass | 0.027 kg (no battery) / 0.033 kg (with battery) | Bitcraze official |
| Inertia Ixx | 16.572e-6 kg·m² | System ID |
| Inertia Iyy | 16.656e-6 kg·m² | System ID |
| Inertia Izz | 29.262e-6 kg·m² | System ID |
| Arm length | 0.046 m (center to motor) | 92mm diagonal / 2 |
| Max thrust per motor | ~0.15 N (~15g) | Bitcraze spec |
| PWM-to-thrust | `0.091492681·PWM² + 0.067673604·PWM` | Empirical fit |
| Motor time constant | ~0.02 s | Typical brushed DC |
| Thrust coeff (k_f) | 3.16e-10 N/RPM² | Literature |
| Torque coeff (k_m) | 7.94e-12 N·m/RPM² | Estimated ~k_f/40 |

---

### Rigid Body Dynamics

#### [NEW] [rigid_body_dynamics.py](file:///c:/digital%20model/simulations/rigid_body_dynamics.py)

Full 6-DOF equations of motion using **quaternion** attitude representation (no gimbal lock):

- **Translational:** `m·a = F_thrust + F_gravity + F_drag` in the world frame
- **Rotational:** `I·ω̇ = τ - ω × (I·ω)` (Euler's rotation equation) in the body frame
- **Quaternion kinematics:** `q̇ = 0.5·q ⊗ [0, ω]`
- 4th-order Runge-Kutta integration with configurable timestep (default 1 ms)

State vector (13 elements): `[x, y, z, vx, vy, vz, qw, qx, qy, qz, ωx, ωy, ωz]`

---

### Motor Model

#### [NEW] [motor_model.py](file:///c:/digital%20model/simulations/motor_model.py)

Models the complete motor subsystem for all 4 motors:

- **First-order lag dynamics:** `τ_m · ω̇_motor = ω_cmd - ω_motor` (time constant ~20 ms)
- **PWM → RPM mapping** via the empirical thrust curve (inverted)
- **Thrust generation:** `F_i = k_f · ω_i²`
- **Reactive torque:** `M_i = k_m · ω_i²` (CW/CCW sign convention)
- **Voltage sag model** (optional): thrust reduction under load as battery depletes
- **Motor saturation:** clamps to `[0, max_RPM]`

---

### Motor Mixing

#### [NEW] [motor_mixing.py](file:///c:/digital%20model/simulations/motor_mixing.py)

Firmware-accurate X-configuration motor mixing that maps `[thrust, roll, pitch, yaw]` commands to individual motor powers:

```
Motor layout (top view, nose = +X):

     M4 (CCW)   M1 (CW)
         \      /
          \    /
           \/
           /\
          /  \
         /    \
     M3 (CW)   M2 (CCW)

Mixing (matching Crazyflie firmware):
  M1 = thrust + pitch + yaw
  M2 = thrust - roll  - yaw
  M3 = thrust - pitch + yaw
  M4 = thrust + roll  - yaw
```

Includes both:
- **Forward mixing:** command → motor PWMs (for controller output)
- **Inverse mixing:** motor thrusts → net force/torques (for dynamics)

---

### Aerodynamics Model

#### [NEW] [aerodynamics_model.py](file:///c:/digital%20model/simulations/aerodynamics_model.py)

- **Body drag:** `F_drag = -k_drag · v` (linear drag model, small-velocity regime)
- **Rotor drag torque:** already captured in motor model via `k_m`
- **Ground effect** (optional): thrust amplification factor when `z < 2 · rotor_radius`

---

### Sensor Model

#### [NEW] [sensor_model.py](file:///c:/digital%20model/simulations/sensor_model.py)

Simulated IMU (BMI088-equivalent) output with realistic noise characteristics:

| Sensor | Noise density | Bias instability |
|---|---|---|
| Accelerometer | 150 µg/√Hz | ±40 mg |
| Gyroscope | 0.014 °/s/√Hz | ±1 °/s |

- Adds white noise + slowly-drifting bias (random walk)
- Configurable sample rate (default 1000 Hz, matching Crazyflie)
- Optional: barometer model (BMP388) for altitude

---

### Main Digital Twin Class

#### [NEW] [crazyflie_sim.py](file:///c:/digital%20model/simulations/crazyflie_sim.py)

Orchestrator class `CrazyflieSim` that ties all subsystems together:

```python
class CrazyflieSim:
    def __init__(self, params, dt=0.001)
    def reset(self, initial_state=None) -> state
    def step(self, command) -> state, sensors
    def get_state(self) -> state
    def get_sensor_readings(self) -> dict
```

- Single `step()` call advances the full simulation pipeline: mixing → motor dynamics → forces/torques → rigid body integration → sensor output
- Supports both `[thrust, roll, pitch, yaw]` and direct `[m1, m2, m3, m4]` PWM command modes
- Configurable timestep, with sub-stepping for stability
- Data logging built-in (timestamps, states, commands)

---

### Example Runner

#### [NEW] [run_digital_twin.py](file:///c:/digital%20model/simulations/run_digital_twin.py)

Demonstration script that:
1. Instantiates the digital twin
2. Runs an open-loop hover command (thrust = weight) to verify equilibrium
3. Applies a step disturbance to test dynamics
4. Produces time-series plots (position, attitude, motor RPMs) saved to `simulation_results/`

---

## Verification Plan

### Automated Tests

Run from project root:
```bash
python -m pytest simulations/tests/ -v
```

Tests to implement in `simulations/tests/test_digital_twin.py`:

1. **Motor mixing matrix** — verify M1–M4 signs match firmware convention
2. **Hover equilibrium** — at `thrust = m·g`, verify net acceleration ≈ 0
3. **Quaternion normalization** — verify quaternion stays unit-norm over 10s simulation
4. **Symmetry** — equal motor commands produce zero angular acceleration
5. **Conservation** — zero-command free-fall matches `g` downward

### Visual Verification

Run the example script and inspect the generated plots:
```bash
python simulations/run_digital_twin.py
```
Verify the output plots in `simulation_results/` show physically plausible behavior (stable hover, correct gravity response, reasonable motor RPMs).
