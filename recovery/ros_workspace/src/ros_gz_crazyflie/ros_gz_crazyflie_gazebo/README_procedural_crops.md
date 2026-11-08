# Procedural crop visuals

The greenhouse keeps one simple collision box per row. Visual fidelity is
independent of collision fidelity:

- `low`: the existing row box.
- `medium`: repeated low-poly stems and leaf clusters.
- `high`: denser repeated clusters with varied height and leaf orientation.

The visual layers are intended to be selected by the forthcoming Gazebo GUI
plugin; they must never be used as collision geometry.
