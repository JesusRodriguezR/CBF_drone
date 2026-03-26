# Simulation Analysis

Post-processing scripts, notebooks, and generated plots for analyzing simulation results.

## File Naming Convention

```
analysis_<sim_name>_<descriptor>.<ext>
```

**Examples:**
- `analysis_hover_baseline_comparison.py`
- `analysis_pid_tuning_bode_plots.ipynb`
- `analysis_trajectory_tracking_error.py`

## Output Naming (Plots / Figures)

```
fig_<sim_name>_<descriptor>_<YYYY-MM-DD>.<ext>
```

**Examples:**
- `fig_hover_baseline_position_error_2026-03-23.png`
- `fig_pid_step_response_comparison_2026-03-23.svg`

## Conventions

- Analysis scripts should read from `simulation_results/` and never modify those files.
- Jupyter notebooks (`.ipynb`) are encouraged for exploratory analysis.
- Python scripts (`.py`) are preferred for reproducible, automated analysis pipelines.
- Save generated figures in this folder alongside the script that produced them.
- Include axis labels, units, legends, and titles on all plots.
