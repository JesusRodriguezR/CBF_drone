# Data

Shared datasets, lookup tables, calibration data, and reference data used across the project.

## File Naming Convention

```
<type>_<descriptor>.<ext>
```

**Examples:**
- `lut_motor_thrust_vs_pwm.csv`
- `calibration_imu_offsets.json`
- `reference_trajectory_circle_1m.csv`

## Conventions

- Data here is **shared and reusable** — not tied to a specific simulation or experiment run.
- Experiment-specific raw data belongs in `experiments/exp_results/`.
- Simulation-specific raw data belongs in `simulations/simulation_results/`.
- Include a header row and units comment in all CSV files.
- Document the source and date of calibration data.
