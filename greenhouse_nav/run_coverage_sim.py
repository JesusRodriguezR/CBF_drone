"""
Greenhouse Lane Navigation Simulation — Main Runner
=====================================================

Ties together:
  - GreenhouseEnv (2D environment)
  - DroneModel (kinematic drone)
  - CoveragePlanner (boustrophedon waypoints)

Runs a simple waypoint-following controller and produces:
  - Top-down trajectory plot
  - Drone state over time

This is Block 1 of the sprint — just the environment + planner +
basic waypoint following. The FSM and CBF will be added in Block 2-4.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from greenhouse_env import GreenhouseEnv, GreenhouseConfig
from drone_model import DroneModel, DroneConfig
from coverage_planner import CoveragePlanner


def run_simulation(
    n_lanes: int = 6,
    dt: float = 0.02,          # 50 Hz
    cruise_speed: float = 0.4,  # m/s
    wp_reach_dist: float = 0.15,  # waypoint reached threshold [m]
    max_time: float = 300.0,   # max sim time [s]
):
    """Run the greenhouse lane navigation simulation."""

    # ── Setup ────────────────────────────────────
    env_config = GreenhouseConfig(
        n_lanes=n_lanes,
        lane_width=1.2,
        lane_length=8.0,
        headland_depth=1.5,
        crop_row_width=0.3,
    )
    env = GreenhouseEnv(env_config)

    drone_config = DroneConfig(max_speed=cruise_speed)
    drone = DroneModel(drone_config)

    planner = CoveragePlanner(env)
    waypoints = planner.plan_coverage(start_from='top')

    # Start position: top headland, above lane 1
    start_x = env.lane_centers_x[0]
    start_y = env.lane_y_max + env.config.headland_depth * 0.5
    drone.reset(start_x, start_y, theta=-np.pi/2)  # facing down

    print(f"Greenhouse: {n_lanes} lanes, {env.total_width:.1f}m x {env.total_height:.1f}m")
    print(f"Waypoints: {len(waypoints)}")
    print(f"Starting at ({start_x:.2f}, {start_y:.2f})")
    print()

    # ── Simulation loop ──────────────────────────
    t = 0.0
    step_count = 0

    while not planner.is_complete() and t < max_time:
        wp = planner.get_current_waypoint()
        if wp is None:
            break

        # Simple proportional waypoint-following controller
        dx = wp.x - drone.x
        dy = wp.y - drone.y
        dist = np.hypot(dx, dy)

        if dist < wp_reach_dist:
            # Reached waypoint
            old_wp = wp
            planner.advance_waypoint()
            wp = planner.get_current_waypoint()

            if old_wp.is_exit and old_wp.lane_idx >= 0:
                visited, total = planner.get_progress()
                print(f"  [t={t:6.1f}s] Completed lane {old_wp.lane_idx+1} "
                      f"({visited}/{total} lanes)")

            if wp is None:
                drone.log_state('DONE')
                break

            dx = wp.x - drone.x
            dy = wp.y - drone.y
            dist = np.hypot(dx, dy)

        # Velocity command: go toward waypoint
        if dist > 0.01:
            vx_cmd = cruise_speed * dx / dist
            vy_cmd = cruise_speed * dy / dist
        else:
            vx_cmd, vy_cmd = 0.0, 0.0

        # Determine state label for logging
        region = env.get_region(drone.x, drone.y)
        if 'lane' in region:
            state_label = 'IN_LANE'
        elif 'headland' in region:
            state_label = 'HEADLAND'
        else:
            state_label = 'TRANSIT'

        drone.step_velocity(vx_cmd, vy_cmd, dt)
        drone.log_state(state_label)

        t += dt
        step_count += 1

    # ── Results ──────────────────────────────────
    visited, total = planner.get_progress()
    print(f"\nSimulation complete:")
    print(f"  Time: {t:.1f}s")
    print(f"  Steps: {step_count}")
    print(f"  Lanes covered: {visited}/{total}")
    print(f"  Coverage: {100*visited/total:.0f}%")

    log = drone.get_log_arrays()
    return env, drone, planner, log


def plot_results(env, drone, planner, log, save_path=None):
    """Generate the top-down trajectory plot."""

    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                             gridspec_kw={'width_ratios': [2, 1]})

    # ── Left: Top-down trajectory ────────────────
    ax = axes[0]
    env.render(ax=ax)

    # Color trajectory by state
    state_colors = {
        'IN_LANE': '#1e64c8',    # blue
        'HEADLAND': '#e08c14',   # orange
        'TRANSIT': '#888888',    # gray
        'DONE': '#2d8a2d',       # green
    }

    x = log['x']
    y = log['y']
    labels = log['state_label']

    # Draw trajectory segments colored by state
    for i in range(1, len(x)):
        color = state_colors.get(labels[i], '#888')
        ax.plot([x[i-1], x[i]], [y[i-1], y[i]],
                color=color, linewidth=1.5, alpha=0.8)

    # Mark waypoints
    wp_pos = planner.get_waypoint_positions()
    ax.plot(wp_pos[:, 0], wp_pos[:, 1], 'k^', markersize=5, alpha=0.3,
            label='Waypoints')

    # Start and end markers
    ax.plot(x[0], y[0], 'go', markersize=10, zorder=5, label='Start')
    ax.plot(x[-1], y[-1], 'rs', markersize=10, zorder=5, label='End')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#1e64c8', lw=2, label='In Lane'),
        Line2D([0], [0], color='#e08c14', lw=2, label='Headland'),
        Line2D([0], [0], marker='o', color='g', lw=0, markersize=8, label='Start'),
        Line2D([0], [0], marker='s', color='r', lw=0, markersize=8, label='End'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    ax.set_title('Greenhouse Lane Coverage — Trajectory', fontweight='bold')

    # ── Right: Coverage progress ─────────────────
    ax2 = axes[1]
    t = log['time']

    # Compute cumulative coverage
    n_lanes = env.config.n_lanes
    coverage_pct = np.zeros(len(t))
    lanes_done = set()

    for i, label in enumerate(labels):
        # Check if we just exited a lane by looking at region
        region = env.get_region(x[i], y[i])
        if region.startswith('lane_'):
            lane_idx = int(region.split('_')[1])
            lanes_done.add(lane_idx)
        coverage_pct[i] = 100.0 * len(lanes_done) / n_lanes

    ax2.plot(t, coverage_pct, color='#1e64c8', linewidth=2)
    ax2.fill_between(t, 0, coverage_pct, alpha=0.15, color='#1e64c8')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Coverage [%]')
    ax2.set_title('Lane Coverage Over Time', fontweight='bold')
    ax2.set_ylim(-5, 105)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(100, color='#2d8a2d', linestyle='--', alpha=0.5, label='100%')
    ax2.legend(fontsize=8)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved plot to: {save_path}")

    plt.show()


if __name__ == '__main__':
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)

    env, drone, planner, log = run_simulation(n_lanes=6)
    plot_results(env, drone, planner, log,
                 save_path=os.path.join(results_dir, 'trajectory_coverage.png'))
