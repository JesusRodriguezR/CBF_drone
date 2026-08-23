# Diffussion Crop-Row Inspection

Prototype package for a drone system that inspects `N` crop rows under partial observability.

The folder name intentionally matches the requested spelling: `diffussion`.

## Objective

- Full coverage: every row is inspected once.
- No redundant revisits: the planner targets only unvisited rows until coverage is complete.
- Robustness to visually identical rows: belief combines perception with topological odometry and visited-row memory.
- Smooth, feasible motion: the local trajectory generator samples and scores bounded trajectories.

## Modules

### 1. Belief and Memory

File: `belief_memory.py`

Tracks:

- Probabilistic estimated current row.
- Visited rows.
- Localization confidence.

Input:

- `PerceptionInput`
- `OdometryDelta`

Output:

- `BeliefState`

### 2. High-Level Planner

File: `high_level_planner.py`

Decides:

- `target_row`
- `task_mode`: `inspect_row`, `transition_between_rows`, or `complete`
- Local `motion_intent`: `follow_row`, `shift_left`, `shift_right`, or `hold`

The planner owns global coverage decisions. It avoids revisiting inspected rows unless all rows are already complete.

### 3. Diffusion Trajectory Generator

File: `trajectory_generator.py`

Role:

- Generate local trajectories under uncertainty.
- Handle ambiguous local visual observations.
- Generate smooth, feasible paths.
- Support multi-modal local trajectories.

Hard constraints:

- Must not decide the next row.
- Must not track mission progress.
- Must not receive `target_row`, `visited_rows`, or row belief state.

Inputs:

- Current local observation.
- Motion intent: `follow_row`, `shift_left`, `shift_right`, or `hold`.

Current implementation:

- A deterministic diffusion-style conditional sampler.
- It samples candidate futures from local observation hypotheses, injects uncertainty-scaled noise, scores smoothness and endpoint alignment, then returns feasible trajectory modes.
- `generate(...)` returns the best local path.
- `generate_modes(...)` returns multiple diverse local paths for ambiguous observations.
- A trained denoising model can replace `_sample_candidate` later while preserving the interface.

## Demo

Run from the repo root:

```powershell
python -m diffussion.run_demo
```

## Tests

Run:

```powershell
pytest diffussion/tests
```
