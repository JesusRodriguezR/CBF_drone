# Config

Configuration files for parameters, controller gains, hardware settings, and environment definitions.

## File Naming Convention

```
<scope>_<descriptor>.yaml   (or .json / .toml)
```

**Examples:**
- `crazyflie_default_params.yaml`
- `pid_gains_hover.yaml`
- `lqr_gains_trajectory.yaml`
- `radio_config.yaml`
- `arena_dimensions.yaml`

## Conventions

- Use **YAML** as the default config format for readability; JSON or TOML are acceptable alternatives.
- Never hardcode tunable parameters in scripts — always load from config files.
- Version-control all config files. Use descriptive commit messages when changing gains.
- Group related parameters logically within each file.
- Include comments explaining each parameter and its units.
