# Control Architecture Scripts

Core control algorithms and architecture implementations for the Crazyflie platform.

## Purpose

This is the **central library** of controllers and control-related utilities. Both simulation and experiment scripts import from here to ensure consistency between simulated and real-world behavior.

## File Naming Convention

```
<controller_type>_<descriptor>.py
```

**Examples:**
- `pid_attitude_controller.py`
- `lqr_hover_controller.py`
- `mpc_trajectory_tracker.py`
- `mixer_motor_commands.py`
- `state_estimator_complementary.py`

## Internal Structure

```
control_arq_scripts/
├── controllers/    # (optional) Controller implementations
├── estimators/     # (optional) State estimation algorithms
├── mixers/         # (optional) Control allocation / motor mixing
└── utils/          # (optional) Shared control utilities
```

## Conventions

- Each controller should implement a consistent interface (e.g., `compute(state, reference) -> command`).
- Document control gains, expected input/output formats, and frame conventions in docstrings.
- Use SI units throughout (meters, radians, seconds).
- Frame convention: **NED** (North-East-Down) or **ENU** (East-North-Up) — document which one is used.
- Include unit tests or validation scripts for each controller where feasible.
