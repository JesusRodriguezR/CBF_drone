"""
Crazyflie Digital Twin — Example Runner
=========================================

Demonstrates the digital twin by running three scenarios:
    1. Open-loop hover (thrust = weight)
    2. Step disturbance (pitch kick)
    3. Free fall (zero thrust)

Produces time-series plots saved to simulation_results/.
"""

import sys
import os

# Add simulations directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime

from crazyflie_params import CrazyflieParams
from crazyflie_sim import CrazyflieSim, SimConfig


def run_hover_test(duration: float = 5.0) -> dict:
    """
    Scenario 1: Open-loop hover.

    Apply constant thrust = mg/4 per motor and verify the drone
    stays in equilibrium.
    """
    print("=" * 60)
    print("Scenario 1: Open-Loop Hover Test")
    print("=" * 60)

    params = CrazyflieParams()
    config = SimConfig(dt=0.001, enable_sensors=True, log_data=True)
    sim = CrazyflieSim(params, config)

    # Start at 1m altitude (NED: z = -1.0)
    sim.reset(position=np.array([0.0, 0.0, -1.0]))

    # Compute hover thrust as fraction of max
    # We need to find the PWM that gives thrust = mg/4 per motor
    # Each motor must produce weight/4
    hover_thrust_per_motor = params.weight / 4.0
    # Using empirical curve: thrust = a*pwm^2 + b*pwm
    # Solve for pwm: a*pwm^2 + b*pwm - thrust_per_motor = 0
    a = params.pwm_thrust_a
    b = params.pwm_thrust_b
    discriminant = b**2 + 4 * a * hover_thrust_per_motor
    hover_pwm = (-b + np.sqrt(discriminant)) / (2 * a)

    print(f"  Mass: {params.mass*1000:.1f} g")
    print(f"  Weight: {params.weight:.4f} N")
    print(f"  Hover PWM (normalised): {hover_pwm:.6f}")
    print(f"  Hover motor speed: {params.hover_motor_speed:.1f} rad/s "
          f"({params.hover_motor_speed * 60 / (2*np.pi):.0f} RPM)")
    print(f"  Simulating {duration}s at {1/config.dt:.0f} Hz...")

    n_steps = int(duration / config.dt)
    # Use direct PWM mode with equal thrust on all motors
    command = np.array([hover_pwm, hover_pwm, hover_pwm, hover_pwm])

    for i in range(n_steps):
        state, sensors = sim.step(command, command_type='pwm')

    log = sim.get_log_arrays()
    altitude = -log['position'][:, 2]  # NED z to altitude

    print(f"  Final altitude: {altitude[-1]:.4f} m (started at 1.0 m)")
    print(f"  Max altitude deviation: {np.max(np.abs(altitude - 1.0)):.6f} m")
    print(f"  Final euler angles (deg): {np.degrees(log['euler_angles'][-1])}")
    print()

    return log


def run_disturbance_test(duration: float = 3.0) -> dict:
    """
    Scenario 2: Step disturbance (pitch kick).

    Hover, then at t=1s apply a pitch disturbance for 0.2s to see
    the open-loop response (no controller).
    """
    print("=" * 60)
    print("Scenario 2: Pitch Disturbance (Open-Loop Response)")
    print("=" * 60)

    params = CrazyflieParams()
    config = SimConfig(dt=0.001, enable_sensors=False, log_data=True)
    sim = CrazyflieSim(params, config)
    sim.reset(position=np.array([0.0, 0.0, -1.0]))

    # Compute hover PWM
    a = params.pwm_thrust_a
    b = params.pwm_thrust_b
    discriminant = b**2 + 4 * a * (params.weight / 4.0)
    hover_pwm = (-b + np.sqrt(discriminant)) / (2 * a)

    n_steps = int(duration / config.dt)
    disturbance_start = 1.0  # s
    disturbance_duration = 0.2  # s

    print(f"  Hover at 1m, disturbance at t={disturbance_start}s "
          f"for {disturbance_duration}s")
    print(f"  Simulating {duration}s...")

    for i in range(n_steps):
        t = i * config.dt
        command = np.array([hover_pwm, hover_pwm, hover_pwm, hover_pwm])

        # Apply pitch disturbance: increase front motors, decrease rear
        if disturbance_start <= t < disturbance_start + disturbance_duration:
            pitch_offset = hover_pwm * 0.15  # 15% differential
            command[0] += pitch_offset   # M1 (front-right) up
            command[3] += pitch_offset   # M4 (front-left) up
            command[1] -= pitch_offset   # M2 (back-right) down
            command[2] -= pitch_offset   # M3 (back-left) down

        state, _ = sim.step(command, command_type='pwm')

    log = sim.get_log_arrays()
    euler = np.degrees(log['euler_angles'])
    print(f"  Max pitch angle: {np.max(np.abs(euler[:, 1])):.2f}°")
    print(f"  Final position: {log['position'][-1]}")
    print()

    return log


def run_freefall_test(duration: float = 1.0) -> dict:
    """
    Scenario 3: Free fall from 2m.

    Zero thrust — should fall under gravity at g = 9.81 m/s².
    """
    print("=" * 60)
    print("Scenario 3: Free Fall (Gravity Validation)")
    print("=" * 60)

    params = CrazyflieParams()
    config = SimConfig(
        dt=0.001,
        enable_sensors=False,
        enable_ground_collision=True,
        log_data=True,
    )
    sim = CrazyflieSim(params, config)
    sim.reset(position=np.array([0.0, 0.0, -2.0]))  # 2m altitude

    n_steps = int(duration / config.dt)
    command = np.array([0.0, 0.0, 0.0, 0.0])

    print(f"  Dropping from 2m altitude, zero thrust")
    print(f"  Simulating {duration}s...")

    for i in range(n_steps):
        state, _ = sim.step(command, command_type='pwm')

    log = sim.get_log_arrays()
    altitude = -log['position'][:, 2]
    time = log['time']

    # Theoretical: z(t) = z0 - 0.5*g*t^2 until ground contact
    t_impact_theory = np.sqrt(2 * 2.0 / params.g)
    print(f"  Theoretical impact time: {t_impact_theory:.4f} s")

    # Find actual impact time (when altitude first reaches ~0)
    impact_idx = np.argmax(altitude <= 0.01)
    if impact_idx > 0:
        print(f"  Simulated impact time:   {time[impact_idx]:.4f} s")
    print(f"  Gravity error: {abs(time[impact_idx] - t_impact_theory):.6f} s")
    print()

    return log


def plot_results(
    hover_log: dict,
    disturbance_log: dict,
    freefall_log: dict,
    save_dir: str,
):
    """Generate comprehensive result plots."""

    timestamp = datetime.now().strftime("%Y-%m-%d")

    # ── Figure 1: Hover Test ──
    fig1 = plt.figure(figsize=(14, 10))
    fig1.suptitle("Crazyflie Digital Twin — Hover Test", fontsize=14, fontweight='bold')
    gs1 = GridSpec(3, 2, figure=fig1, hspace=0.35, wspace=0.3)

    # Position
    ax = fig1.add_subplot(gs1[0, 0])
    t = hover_log['time']
    pos = hover_log['position']
    altitude = -pos[:, 2]
    ax.plot(t, altitude, 'b-', linewidth=1)
    ax.set_ylabel('Altitude [m]')
    ax.set_title('Altitude')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Target')
    ax.legend(fontsize=8)

    # XY position
    ax = fig1.add_subplot(gs1[0, 1])
    ax.plot(t, pos[:, 0]*100, label='X', linewidth=1)
    ax.plot(t, pos[:, 1]*100, label='Y', linewidth=1)
    ax.set_ylabel('Position [cm]')
    ax.set_title('Lateral Position')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Euler angles
    ax = fig1.add_subplot(gs1[1, 0])
    euler = np.degrees(hover_log['euler_angles'])
    ax.plot(t, euler[:, 0], label='Roll', linewidth=1)
    ax.plot(t, euler[:, 1], label='Pitch', linewidth=1)
    ax.plot(t, euler[:, 2], label='Yaw', linewidth=1)
    ax.set_ylabel('Angle [°]')
    ax.set_title('Euler Angles')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Motor speeds
    ax = fig1.add_subplot(gs1[1, 1])
    speeds_rpm = hover_log['motor_speeds'] * 60 / (2 * np.pi)
    for i in range(4):
        ax.plot(t, speeds_rpm[:, i], label=f'M{i+1}', linewidth=1)
    ax.set_ylabel('Speed [RPM]')
    ax.set_title('Motor Speeds')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Motor thrusts
    ax = fig1.add_subplot(gs1[2, 0])
    thrusts_g = hover_log['motor_thrusts'] * 1000 / 9.81  # N to grams-force
    for i in range(4):
        ax.plot(t, thrusts_g[:, i], label=f'M{i+1}', linewidth=1)
    ax.set_ylabel('Thrust [gf]')
    ax.set_xlabel('Time [s]')
    ax.set_title('Motor Thrusts')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Velocity
    ax = fig1.add_subplot(gs1[2, 1])
    vel = hover_log['velocity']
    ax.plot(t, vel[:, 0], label='Vx', linewidth=1)
    ax.plot(t, vel[:, 1], label='Vy', linewidth=1)
    ax.plot(t, vel[:, 2], label='Vz', linewidth=1)
    ax.set_ylabel('Velocity [m/s]')
    ax.set_xlabel('Time [s]')
    ax.set_title('Velocity')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    path1 = os.path.join(save_dir, f"fig_hover_test_{timestamp}.png")
    fig1.savefig(path1, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path1}")

    # ── Figure 2: Disturbance Response ──
    fig2 = plt.figure(figsize=(14, 8))
    fig2.suptitle("Crazyflie Digital Twin — Pitch Disturbance Response",
                  fontsize=14, fontweight='bold')
    gs2 = GridSpec(2, 2, figure=fig2, hspace=0.35, wspace=0.3)

    t = disturbance_log['time']
    pos = disturbance_log['position']
    euler = np.degrees(disturbance_log['euler_angles'])

    # Altitude
    ax = fig2.add_subplot(gs2[0, 0])
    ax.plot(t, -pos[:, 2], 'b-', linewidth=1)
    ax.axvspan(1.0, 1.2, alpha=0.15, color='red', label='Disturbance')
    ax.set_ylabel('Altitude [m]')
    ax.set_title('Altitude')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # X position (forward motion from pitch)
    ax = fig2.add_subplot(gs2[0, 1])
    ax.plot(t, pos[:, 0], 'g-', linewidth=1)
    ax.axvspan(1.0, 1.2, alpha=0.15, color='red')
    ax.set_ylabel('X position [m]')
    ax.set_title('Forward Position (from pitch)')
    ax.grid(True, alpha=0.3)

    # Euler angles
    ax = fig2.add_subplot(gs2[1, 0])
    ax.plot(t, euler[:, 0], label='Roll', linewidth=1)
    ax.plot(t, euler[:, 1], label='Pitch', linewidth=1, color='red')
    ax.plot(t, euler[:, 2], label='Yaw', linewidth=1)
    ax.axvspan(1.0, 1.2, alpha=0.15, color='red')
    ax.set_ylabel('Angle [°]')
    ax.set_xlabel('Time [s]')
    ax.set_title('Euler Angles')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Motor speeds
    ax = fig2.add_subplot(gs2[1, 1])
    speeds_rpm = disturbance_log['motor_speeds'] * 60 / (2 * np.pi)
    for i in range(4):
        ax.plot(t, speeds_rpm[:, i], label=f'M{i+1}', linewidth=1)
    ax.axvspan(1.0, 1.2, alpha=0.15, color='red')
    ax.set_ylabel('Speed [RPM]')
    ax.set_xlabel('Time [s]')
    ax.set_title('Motor Speeds')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    path2 = os.path.join(save_dir, f"fig_disturbance_response_{timestamp}.png")
    fig2.savefig(path2, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path2}")

    # ── Figure 3: Free Fall ──
    fig3 = plt.figure(figsize=(10, 6))
    fig3.suptitle("Crazyflie Digital Twin — Free Fall Validation",
                  fontsize=14, fontweight='bold')

    t = freefall_log['time']
    pos = freefall_log['position']
    altitude = -pos[:, 2]

    # Theoretical trajectory
    g = 9.81
    t_theory = np.linspace(0, max(t), 1000)
    alt_theory = np.maximum(2.0 - 0.5 * g * t_theory**2, 0.0)

    ax1 = fig3.add_subplot(1, 2, 1)
    ax1.plot(t, altitude, 'b-', linewidth=2, label='Simulation')
    ax1.plot(t_theory, alt_theory, 'r--', linewidth=1.5, label='Theory (½gt²)')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Altitude [m]')
    ax1.set_title('Free Fall Altitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig3.add_subplot(1, 2, 2)
    vel = freefall_log['velocity']
    v_theory = np.minimum(g * t_theory, g * np.sqrt(2 * 2.0 / g))
    ax2.plot(t, vel[:, 2], 'b-', linewidth=2, label='Vz (sim)')
    ax2.plot(t_theory, v_theory, 'r--', linewidth=1.5, label='Vz (theory)')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Velocity Z [m/s] (NED: + = down)')
    ax2.set_title('Downward Velocity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    path3 = os.path.join(save_dir, f"fig_freefall_validation_{timestamp}.png")
    fig3.savefig(path3, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path3}")

    plt.close('all')


def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     CRAZYFLIE 2.1 DIGITAL TWIN — SIMULATION RUNNER     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # Results directory
    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "simulation_results"
    )
    os.makedirs(results_dir, exist_ok=True)

    # Run scenarios
    hover_log = run_hover_test(duration=5.0)
    disturbance_log = run_disturbance_test(duration=3.0)
    freefall_log = run_freefall_test(duration=1.0)

    # Plot
    print("=" * 60)
    print("Generating plots...")
    print("=" * 60)
    plot_results(hover_log, disturbance_log, freefall_log, results_dir)
    print()
    print("Done! Check simulation_results/ for output figures.")


if __name__ == "__main__":
    main()
