"""
Simulated Perception Module
=============================

Simulates what the AI Deck CNN would output in a real deployment:
  - lateral_offset: how far the drone is from lane center [m]
  - end_of_row_prob: probability that the drone is at the end of a lane [0,1]

Uses ground-truth position + noise to mimic a real CNN running on GAP8.
This lets us test the FSM and CBF without needing real camera images.
"""

import numpy as np
from greenhouse_env import GreenhouseEnv


class SimulatedPerception:
    """
    Simulated CNN perception for lane navigation.

    Mimics the two-output DroNet-style network:
      Output 1: lateral_offset — signed distance from lane center [m]
                positive = drone is to the right of center
      Output 2: end_of_row_prob — sigmoid probability of being at end of row
    """

    def __init__(self, env: GreenhouseEnv, noise_std: float = 0.02,
                 end_detect_distance: float = 0.8, seed: int = 42):
        """
        Parameters
        ----------
        env : GreenhouseEnv
        noise_std : float
            Standard deviation of lateral offset noise [m].
            Simulates CNN prediction uncertainty.
        end_detect_distance : float
            Distance from lane end at which end_of_row_prob starts rising [m].
        seed : int
            Random seed for reproducibility.
        """
        self.env = env
        self.noise_std = noise_std
        self.end_detect_distance = end_detect_distance
        self.rng = np.random.default_rng(seed)

        # Log
        self.log = {
            'time': [],
            'lateral_offset': [],
            'lateral_offset_noisy': [],
            'end_of_row_prob': [],
            'lane_idx': [],
        }

    def reset(self, seed: int = None):
        """Reset the perception module."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        for key in self.log:
            self.log[key] = []

    def perceive(self, x: float, y: float, current_lane: int,
                 lane_direction: int, time: float = 0.0) -> dict:
        """
        Compute perception outputs given drone position and lane context.

        Parameters
        ----------
        x, y : float
            Drone position [m].
        current_lane : int
            Index of the lane the drone is targeting (0-based).
        lane_direction : int
            +1 if flying top→bottom (decreasing y), -1 if bottom→top.
        time : float
            Current simulation time [s].

        Returns
        -------
        perception : dict with keys:
            'lateral_offset' : float — noisy signed offset from lane center [m]
            'end_of_row_prob' : float — probability of end-of-row [0, 1]
            'lateral_offset_true' : float — ground-truth offset (for logging)
        """
        env = self.env

        # ── Lateral offset ──
        if 0 <= current_lane < len(env.lane_centers_x):
            lane_center_x = env.lane_centers_x[current_lane]
            lateral_offset_true = x - lane_center_x  # positive = right of center
        else:
            lateral_offset_true = 0.0

        # Add noise (simulating CNN uncertainty)
        noise = self.rng.normal(0, self.noise_std)
        lateral_offset_noisy = lateral_offset_true + noise

        # ── End-of-row probability ──
        # Compute distance to the end of the lane in the direction of travel
        if lane_direction > 0:
            # Flying top → bottom (decreasing y)
            dist_to_end = y - env.lane_y_min
        else:
            # Flying bottom → top (increasing y)
            dist_to_end = env.lane_y_max - y

        # Sigmoid activation based on distance
        # When dist_to_end < end_detect_distance, probability rises
        if dist_to_end <= 0:
            end_prob = 1.0
        elif dist_to_end > self.end_detect_distance * 2:
            end_prob = 0.0
        else:
            # Smooth sigmoid
            k = 8.0 / self.end_detect_distance  # steepness
            end_prob = 1.0 / (1.0 + np.exp(k * (dist_to_end - self.end_detect_distance * 0.5)))

        # Add small noise to end probability
        end_prob = np.clip(end_prob + self.rng.normal(0, 0.02), 0.0, 1.0)

        # ── Log ──
        self.log['time'].append(time)
        self.log['lateral_offset'].append(lateral_offset_true)
        self.log['lateral_offset_noisy'].append(lateral_offset_noisy)
        self.log['end_of_row_prob'].append(end_prob)
        self.log['lane_idx'].append(current_lane)

        return {
            'lateral_offset': lateral_offset_noisy,
            'lateral_offset_true': lateral_offset_true,
            'end_of_row_prob': end_prob,
        }

    def get_log_arrays(self) -> dict:
        """Convert logs to numpy arrays."""
        return {key: np.array(val) for key, val in self.log.items()}
