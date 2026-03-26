# Experiment Results

Raw data logs from real-world Crazyflie flights. **This folder is treated as immutable.**

## File Naming Convention

```
<YYYY-MM-DD>_<exp_name>_<run_id>.<ext>
```

**Examples:**
- `2026-03-23_hover_indoor_run01.csv`
- `2026-03-23_waypoint_follow_run02.usd`
- `2026-03-23_hover_indoor_run01_meta.json`

## Supported Formats

| Format  | Use Case                                       |
|---------|-------------------------------------------------|
| `.csv`  | Exported time-series data (position, attitude)  |
| `.usd`  | Crazyflie onboard SD-card log (USD deck)        |
| `.json` | Experiment metadata (params, conditions, notes)  |
| `.bag`  | ROS bag recordings (if applicable)              |

## Conventions

- Every experiment run **must** have a companion `_meta.json` file recording:
  - Date and time
  - Crazyflie ID and firmware version
  - Controller gains used
  - Environmental notes (room, lighting, obstacles)
- Use incrementing `run##` suffixes for repeated runs.
- Do **not** place analysis outputs or plots here.
