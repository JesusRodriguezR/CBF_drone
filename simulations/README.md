# Simulations

Simulation environments, launch scripts, and configuration for Crazyflie digital-twin work.

## Purpose

This folder contains the simulation scripts themselves — the environments, scenario definitions, and launch configurations used to run simulated flights.

## File Naming Convention

```
<sim_type>_<descriptor>_v<version>.py
```

**Examples:**
- `hover_baseline_v1.py`
- `trajectory_tracking_circle_v2.py`
- `pid_tuning_step_response_v1.py`

## Internal Structure

```
simulations/
├── simulation_results/   # Raw output data from simulation runs
├── simulation_analysis/  # Analysis notebooks and scripts
├── envs/                 # (optional) Custom simulation environments
└── scenarios/            # (optional) Scenario definition files
```

## Conventions

- Each simulation script should be self-contained or clearly import from `control_arq_scripts/`.
- Document simulation parameters (timestep, duration, initial conditions) at the top of each file.
- Use a `if __name__ == "__main__":` guard for all runnable scripts.
- Output results to `simulation_results/` using the naming convention defined there.
