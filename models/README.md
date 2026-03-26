# Models

Mathematical and computational models of the Crazyflie system.

## Purpose

Contains dynamic models, aerodynamic models, sensor models, and any system identification results used across simulations and control design.

## File Naming Convention

```
<model_type>_<descriptor>.py
```

**Examples:**
- `dynamics_6dof_rigid_body.py`
- `aerodynamics_thrust_model.py`
- `sensor_imu_noise_model.py`
- `sysid_motor_response.py`

## Conventions

- Models should be usable both standalone and as imports in simulation scripts.
- Document all physical parameters, units, and assumptions in docstrings.
- Store identified parameter values in `config/` rather than hardcoding them.
- Reference relevant papers or datasheets where applicable.
