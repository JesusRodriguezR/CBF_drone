# Tools

Utility scripts and helper tools that support the project workflow.

## File Naming Convention

```
<action>_<descriptor>.py
```

**Examples:**
- `convert_usd_to_csv.py`
- `plot_quick_flight_summary.py`
- `validate_config_schema.py`
- `batch_run_simulations.py`

## Conventions

- Tools should be CLI-friendly (use `argparse` or `click` for argument parsing).
- Each tool should have a `--help` flag documenting its usage.
- Keep tools generic and reusable — avoid experiment-specific logic.
