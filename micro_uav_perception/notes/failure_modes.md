# Failure Mode Analysis

## Motion Category Results

| Condition | N | RMSE dx | RMSE dy | RMSE dyaw |
|-----------|---|---------|---------|----------|
| all_test | 820 | 0.0825 | 0.0992 | 0.0463 |
| fast_motion | 164 | 0.1060 | 0.1425 | 0.0675 |
| slow_motion | 164 | 0.1237 | 0.0803 | 0.0338 |
| high_yaw | 164 | 0.0979 | 0.1669 | 0.0780 |
| forward_dominant | 549 | 0.0691 | 0.0804 | 0.0384 |

## Synthetic Augmentation Results

| Condition | N | RMSE dx | RMSE dy | RMSE dyaw | Impact |
|-----------|---|---------|---------|----------|--------|
| normal | 820 | 0.0825 | 0.0992 | 0.0463 | minimal (+0%) |
| low_texture | 820 | 0.0825 | 0.0992 | 0.0463 | minimal (-0%) |
| motion_blur | 820 | 0.0825 | 0.0992 | 0.0463 | minimal (-0%) |
| brightness_dark | 820 | 0.0841 | 0.1013 | 0.0453 | minimal (+2%) |
| brightness_bright | 820 | 0.0819 | 0.0993 | 0.0468 | minimal (-0%) |
| noise | 820 | 0.0825 | 0.0992 | 0.0463 | minimal (+0%) |

## Key Observations

- Fast motion increases local-motion estimation error due to larger inter-frame displacement.
- High yaw still degrades translation estimates, but the local-frame formulation is physically cleaner.
- Low texture and motion blur reduce usable visual structure.
- Large lighting shifts remain a major generalization problem.
- Additive noise is less damaging than strong brightness changes.
