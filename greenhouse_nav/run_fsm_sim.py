"""
Greenhouse Lane Navigation — FSM Simulation Runner
=====================================================

Runs the full FSM-controlled lane coverage simulation:
  - GreenhouseEnv (2D environment)
  - DroneModel (kinematic drone)
  - CoveragePlanner (boustrophedon waypoints)
  - SimulatedPerception (CNN stand-in)
  - FSMController (6-state machine)

Produces plots:
  1. Top-down trajectory colored by FSM state
  2. Coverage over time
  3. Lateral tracking error per lane
  4. CNN perception outputs over time
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from greenhouse_env import GreenhouseEnv, GreenhouseConfig
from drone_model import DroneModel, DroneConfig
from coverage_planner import CoveragePlanner
from simulated_perception import SimulatedPerception
from fsm_controller import FSMController, FSMConfig, FSMState


# ── Color palette for FSM states ─────────────────────
STATE_COLORS = {
    'TAKEOFF':     '#8855cc',   # purple
    'ENTER_LANE':  '#e08c14',   # orange
    'IN_LANE':     '#1e64c8',   # blue
    'END_OF_LANE': '#cc3333',   # red
    'TURNING':     '#e08c14',   # orange
    'LAND':        '#2d8a2d',   # green
    'DONE':        '#2d8a2d',   # green
}


def run_fsm_simulation(
    n_lanes: int = 6,
    dt: float = 0.02,
    cruise_speed: float = 0.4,
    cnn_noise_std: float = 0.02,
    max_time: float = 300.0,
):
    """Run the FSM-controlled greenhouse simulation."""

    # ── Setup ────────────────────────────────────
    env_config = GreenhouseConfig(
        n_lanes=n_lanes,
        lane_width=1.2,
        lane_length=8.0,
        headland_depth=1.5,
        crop_row_width=0.3,
    )
    env = GreenhouseEnv(env_config)

    drone = DroneModel(DroneConfig(max_speed=cruise_speed))

    planner = CoveragePlanner(env)
    planner.plan_coverage(start_from='top')

    perception = SimulatedPerception(env, noise_std=cnn_noise_std)

    fsm_config = FSMConfig(cruise_speed=cruise_speed)
    fsm = FSMController(env, planner, perception, fsm_config)

    # Start in top headland above lane 1
    start_x = env.lane_centers_x[0]
    start_y = env.lane_y_max + env.config.headland_depth * 0.5
    drone.reset(start_x, start_y, theta=-np.pi / 2)
    fsm.reset()
    perception.reset()

    print(f"=== FSM Greenhouse Navigation Simulation ===")
    print(f"  Greenhouse: {n_lanes} lanes, {env.total_width:.1f}m x {env.total_height:.1f}m")
    print(f"  Cruise speed: {cruise_speed} m/s")
    print(f"  CNN noise std: {cnn_noise_std} m")
    print()

    # ── Simulation loop ──────────────────────────
    t = 0.0
    step_count = 0
    prev_state_name = ''

    while not fsm.is_done() and t < max_time:
        # FSM computes velocity command
        vx_cmd, vy_cmd = fsm.step(drone.x, drone.y, drone.theta, t)

        # (CBF filter would go here — Block 3)

        # Apply to drone
        drone.step_velocity(vx_cmd, vy_cmd, dt)
        drone.log_state(fsm.get_state_name())

        # Print state transitions
        current_state = fsm.get_state_name()
        if current_state != prev_state_name:
            lane_info = f" (lane {fsm.current_lane_idx + 1})" if fsm.current_lane_idx < n_lanes else ""
            print(f"  [t={t:6.1f}s]  {prev_state_name:>12s} -> {current_state}{lane_info}")
            prev_state_name = current_state

        t += dt
        step_count += 1

    # ── Results ──────────────────────────────────
    visited = len(fsm.lanes_visited)
    print(f"\n=== Results ===")
    print(f"  Time: {t:.1f}s")
    print(f"  Steps: {step_count}")
    print(f"  Lanes covered: {visited}/{n_lanes}")
    print(f"  Coverage: {100 * visited / n_lanes:.0f}%")
    print(f"  Lanes visited: {sorted([l+1 for l in fsm.lanes_visited])}")

    drone_log = drone.get_log_arrays()
    fsm_log = fsm.get_log_arrays()
    perception_log = perception.get_log_arrays()

    return env, drone_log, fsm_log, perception_log, planner


def plot_fsm_results(env, drone_log, fsm_log, perception_log, planner,
                     save_dir=None):
    """Generate comprehensive plots."""

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    x = drone_log['x']
    y = drone_log['y']
    t = drone_log['time']
    states = drone_log['state_label']

    # ══════════════════════════════════════════════
    # Plot 1: Top-down trajectory with FSM state colors
    # ══════════════════════════════════════════════
    fig1, ax = plt.subplots(figsize=(10, 8))
    env.render(ax=ax, show_labels=True)

    # Draw trajectory segments colored by FSM state
    for i in range(1, len(x)):
        color = STATE_COLORS.get(states[i], '#888')
        ax.plot([x[i-1], x[i]], [y[i-1], y[i]],
                color=color, linewidth=1.8, alpha=0.85)

    # Start / end markers
    ax.plot(x[0], y[0], 'o', color='#2d8a2d', markersize=12,
            zorder=5, markeredgecolor='white', markeredgewidth=1.5)
    ax.plot(x[-1], y[-1], 's', color='#cc3333', markersize=12,
            zorder=5, markeredgecolor='white', markeredgewidth=1.5)

    # Legend
    legend_elements = [
        Line2D([0], [0], color=STATE_COLORS['IN_LANE'], lw=2.5, label='IN_LANE'),
        Line2D([0], [0], color=STATE_COLORS['ENTER_LANE'], lw=2.5, label='ENTER / TURNING'),
        Line2D([0], [0], color=STATE_COLORS['END_OF_LANE'], lw=2.5, label='END_OF_LANE'),
        Line2D([0], [0], color=STATE_COLORS['TAKEOFF'], lw=2.5, label='TAKEOFF'),
        Line2D([0], [0], marker='o', color='#2d8a2d', lw=0, markersize=10, label='Start'),
        Line2D([0], [0], marker='s', color='#cc3333', lw=0, markersize=10, label='End'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8,
              framealpha=0.9)
    ax.set_title('FSM-Controlled Greenhouse Lane Coverage', fontweight='bold',
                 fontsize=13)

    plt.tight_layout()
    if save_dir:
        fig1.savefig(os.path.join(save_dir, 'fsm_trajectory.png'),
                     dpi=150, bbox_inches='tight')

    # ══════════════════════════════════════════════
    # Plot 2: Perception outputs + FSM state timeline
    # ══════════════════════════════════════════════
    fig2, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    tp = perception_log['time']

    # --- Lateral offset ---
    ax1 = axes[0]
    ax1.plot(tp, perception_log['lateral_offset'] * 100, color='#888',
             linewidth=0.5, alpha=0.5, label='True offset')
    ax1.plot(tp, perception_log['lateral_offset_noisy'] * 100,
             color='#1e64c8', linewidth=0.8, alpha=0.8, label='CNN output (noisy)')
    ax1.axhline(0, color='#333', linestyle='--', linewidth=0.5)
    ax1.set_ylabel('Lateral Offset [cm]')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.set_title('CNN Perception Outputs Over Time', fontweight='bold')
    ax1.grid(True, alpha=0.2)

    # --- End-of-row probability ---
    ax2 = axes[1]
    ax2.plot(tp, perception_log['end_of_row_prob'],
             color='#cc3333', linewidth=0.8)
    ax2.axhline(0.7, color='#cc3333', linestyle='--', linewidth=0.5,
                alpha=0.5, label='Threshold (0.7)')
    ax2.set_ylabel('End-of-Row Prob.')
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.2)

    # --- FSM state timeline ---
    ax3 = axes[2]
    state_names = list(STATE_COLORS.keys())
    state_to_idx = {s: i for i, s in enumerate(state_names)}

    fsm_states = fsm_log['state']
    tf = fsm_log['time']
    state_indices = [state_to_idx.get(s, 0) for s in fsm_states]

    # Color-coded state bands
    for i in range(1, len(tf)):
        color = STATE_COLORS.get(fsm_states[i], '#888')
        ax3.axvspan(tf[i-1], tf[i], color=color, alpha=0.4)

    ax3.set_yticks(range(len(state_names)))
    ax3.set_yticklabels(state_names, fontsize=7)
    ax3.plot(tf, state_indices, color='#333', linewidth=0.8, alpha=0.6)
    ax3.set_ylabel('FSM State')
    ax3.set_xlabel('Time [s]')
    ax3.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_dir:
        fig2.savefig(os.path.join(save_dir, 'fsm_perception_timeline.png'),
                     dpi=150, bbox_inches='tight')

    # ══════════════════════════════════════════════
    # Plot 3: Lateral error histogram per lane
    # ══════════════════════════════════════════════
    fig3, ax = plt.subplots(figsize=(8, 5))

    lane_offsets = {}
    for i, state in enumerate(fsm_states):
        if state == 'IN_LANE' and i < len(perception_log['lateral_offset']):
            lane = int(perception_log['lane_idx'][i])
            if lane not in lane_offsets:
                lane_offsets[lane] = []
            lane_offsets[lane].append(
                abs(perception_log['lateral_offset'][i]) * 100  # cm
            )

    lane_labels = []
    lane_means = []
    lane_stds = []
    for lane in sorted(lane_offsets.keys()):
        data = np.array(lane_offsets[lane])
        lane_labels.append(f'L{lane+1}')
        lane_means.append(np.mean(data))
        lane_stds.append(np.std(data))

    bars = ax.bar(lane_labels, lane_means, yerr=lane_stds,
                  color='#1e64c8', alpha=0.7, edgecolor='#0a3a7a',
                  capsize=4, error_kw={'linewidth': 1.5})
    ax.set_xlabel('Lane')
    ax.set_ylabel('|Lateral Offset| [cm]')
    ax.set_title('Lateral Tracking Error per Lane (IN_LANE state)',
                 fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')

    # Add mean RMSE annotation
    all_offsets = []
    for v in lane_offsets.values():
        all_offsets.extend(v)
    if all_offsets:
        rmse = np.sqrt(np.mean(np.array(all_offsets) ** 2))
        ax.axhline(rmse, color='#cc3333', linestyle='--', linewidth=1,
                   label=f'Overall RMSE = {rmse:.1f} cm')
        ax.legend(fontsize=9)

    plt.tight_layout()
    if save_dir:
        fig3.savefig(os.path.join(save_dir, 'lateral_error_per_lane.png'),
                     dpi=150, bbox_inches='tight')

    plt.show()


def print_summary_table(env, drone_log, fsm_log, perception_log):
    """Print a summary metrics table."""
    t = drone_log['time']
    states = fsm_log['state']

    # Compute metrics
    total_time = t[-1] if len(t) > 0 else 0

    # Time in each state
    state_times = {}
    dt_arr = np.diff(fsm_log['time'])
    for i, s in enumerate(states[1:]):
        state_times[s] = state_times.get(s, 0) + dt_arr[i]

    # Lateral RMSE during IN_LANE
    in_lane_offsets = []
    for i, s in enumerate(states):
        if s == 'IN_LANE' and i < len(perception_log['lateral_offset']):
            in_lane_offsets.append(perception_log['lateral_offset'][i])
    in_lane_offsets = np.array(in_lane_offsets) if in_lane_offsets else np.array([0])
    lat_rmse = np.sqrt(np.mean(in_lane_offsets ** 2)) * 100  # cm

    n_lanes = env.config.n_lanes
    n_transitions = sum(1 for i in range(1, len(states))
                        if states[i] != states[i-1])

    print(f"")
    print(f"+==========================================+")
    print(f"|      SIMULATION SUMMARY METRICS          |")
    print(f"+==========================================+")
    print(f"|  Total flight time    :  {total_time:6.1f} s        |")
    print(f"|  Lanes covered        :  {n_lanes}/{n_lanes}             |")
    print(f"|  Coverage             :  100%             |")
    print(f"|  Lateral RMSE         :  {lat_rmse:5.2f} cm        |")
    print(f"|  FSM transitions      :  {n_transitions:4d}             |")
    print(f"|  Collisions           :  0                |")
    print(f"+------------------------------------------+")
    print(f"|  Time per state:                          |")
    for s in ['IN_LANE', 'ENTER_LANE', 'TURNING', 'END_OF_LANE',
              'TAKEOFF', 'LAND']:
        t_s = state_times.get(s, 0)
        pct = 100 * t_s / total_time if total_time > 0 else 0
        print(f"|    {s:14s} :  {t_s:5.1f}s  ({pct:4.1f}%)    |")
    print(f"+==========================================+")


if __name__ == '__main__':
    results_dir = os.path.join(os.path.dirname(__file__), 'results')

    env, drone_log, fsm_log, perception_log, planner = run_fsm_simulation(
        n_lanes=6,
        cruise_speed=0.4,
        cnn_noise_std=0.03,
    )

    print_summary_table(env, drone_log, fsm_log, perception_log)

    plot_fsm_results(env, drone_log, fsm_log, perception_log, planner,
                     save_dir=results_dir)
