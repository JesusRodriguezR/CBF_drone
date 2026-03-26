# Experiment Analysis

Post-flight analysis scripts, notebooks, and generated figures.

## File Naming Convention

```
analysis_<exp_name>_<descriptor>.<ext>
```

**Examples:**
- `analysis_hover_indoor_position_drift.py`
- `analysis_waypoint_follow_tracking_error.ipynb`

## Output Naming (Plots / Figures)

```
fig_<exp_name>_<descriptor>_<YYYY-MM-DD>.<ext>
```

**Examples:**
- `fig_hover_indoor_xy_drift_2026-03-23.png`
- `fig_waypoint_follow_3d_trajectory_2026-03-23.svg`

## Conventions

- Read raw data from `exp_results/` — never modify source files.
- Compare experimental results against simulation baselines when possible.
- Use consistent color schemes and labeling across related plots.
- Include error metrics (RMSE, MAE) in analysis summaries.
