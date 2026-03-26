# Simulation Results

Raw output data from simulation runs. **This folder is treated as immutable** — never modify files here after they are generated.

## File Naming Convention

```
<YYYY-MM-DD>_<sim_name>_<run_id>.<ext>
```

**Examples:**
- `2026-03-23_hover_baseline_run01.csv`
- `2026-03-23_trajectory_circle_run03.npz`
- `2026-03-23_pid_step_response_run01.json`

## Supported Formats

| Format  | Use Case                              |
|---------|---------------------------------------|
| `.csv`  | Tabular time-series data              |
| `.npz`  | Large numpy arrays (state vectors)    |
| `.json` | Metadata, parameter snapshots         |
| `.bag`  | ROS bag files (if using ROS/Gazebo)   |

## Conventions

- Each run should produce at minimum a data file and a companion metadata file (`.json`) documenting the simulation parameters used.
- Use incrementing `run##` suffixes for repeated runs of the same simulation.
- Do **not** place analysis outputs or plots here — those go in `simulation_analysis/`.
