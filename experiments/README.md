# Experiments

Scripts and configurations for real-world Crazyflie flight experiments.

## Purpose

This folder holds the experiment launch scripts, flight plans, and hardware interface code used to run physical Crazyflie testing sessions.

## File Naming Convention

```
exp_<descriptor>_v<version>.py
```

**Examples:**
- `exp_hover_indoor_v1.py`
- `exp_waypoint_follow_v2.py`
- `exp_disturbance_rejection_v1.py`

## Internal Structure

```
experiments/
├── exp_results/    # Raw flight logs and sensor data
├── exp_analysis/   # Post-flight analysis scripts and plots
└── protocols/      # (optional) Written experiment protocols / checklists
```

## Conventions

- Always include a pre-flight checklist comment block at the top of each experiment script (battery level, firmware version, arena setup).
- Log the Crazyflie firmware version and radio channel in the metadata for every experiment.
- Controllers used should be imported from `control_arq_scripts/`.
- Coordinate with `config/` for hardware parameters and gain values.
