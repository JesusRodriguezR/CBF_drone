# Crazyflie Digital Model

Research and development project for the Crazyflie 2.x quadrotor platform — covering simulation, real-world experiments, and control architecture design.

## Project Structure

```
digital model/
├── simulations/              # Simulation environments and scripts
│   ├── simulation_results/   # Raw simulation output data
│   └── simulation_analysis/  # Post-processing and analysis of sim data
├── experiments/              # Real-world flight experiment scripts
│   ├── exp_results/          # Raw experimental data logs
│   └── exp_analysis/        # Post-processing and analysis of exp data
├── control_arq_scripts/      # Control architecture scripts and algorithms
├── models/                   # System models (dynamics, aerodynamics, sensors)
├── config/                   # Configuration files (params, gains, hardware)
├── docs/                     # Project documentation, references, and notes
├── tools/                    # Utility scripts and helper tools
└── data/                     # Shared datasets (LUTs, sensor calibrations, etc.)
```

## General Conventions

- **Python 3.10+** is the default language unless otherwise specified.
- Use `snake_case` for all file and folder names.
- Every folder contains a `README.md` describing its purpose, file naming, and conventions.
- Dates in filenames use `YYYY-MM-DD` format (e.g., `2026-03-23_hover_test.csv`).
- Keep raw data immutable — analysis scripts read from results folders, never overwrite them.
