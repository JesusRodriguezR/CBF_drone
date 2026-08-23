# Implementation Log

## Session 1 - Initial Setup

- Created project structure under `micro_uav_perception/`
- Set up Python venv with CUDA PyTorch (RTX 4070 laptop)
- Target dataset: UZH-FPV indoor_forward public Snapdragon sequences with clean sequence-level split
- First model: Tiny CNN estimating `[dx, dy, dyaw]` from `96x96` grayscale pairs
- Loss: MSE with yaw weighting (`lambda=0.1`)
- Training: Adam, `lr=1e-3`, `50` epochs, `batch_size=32`

## Session 2 - Generalization Refactor

- Switched from world-frame translation labels to local/body-frame relative motion
- Rebuilt samples as frame triplets `[t-1, t, t+1]`
- Added shared photometric augmentation for all frames in a triplet
- Added early stopping for validation-driven training
- Updated evaluation, trajectory reconstruction, demo packaging, and the live dashboard for the new model interface
