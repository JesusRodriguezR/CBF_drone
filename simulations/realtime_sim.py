"""
Crazyflie Real-Time Interactive Simulation
============================================

Fly the Crazyflie digital twin in real-time using keyboard controls
with a 3D OpenGL visualisation.

Controls:
    W / S       Pitch forward / backward
    A / D       Roll left / right
    Q / E       Yaw left / right
    UP / DOWN   Increase / decrease altitude
    Space       Emergency stop (cut motors)
    R           Reset to start position
    C           Toggle camera mode (follow / free)
    Esc         Quit

Requirements:
    pip install pygame PyOpenGL numpy
"""

import sys
import os
import time
import math

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', 'control_arq_scripts'))

import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

from crazyflie_params import CrazyflieParams
from crazyflie_sim import CrazyflieSim, SimConfig
from rigid_body_dynamics import quat_to_euler, quat_to_rotation_matrix, POS, VEL, QUAT, OMEGA
from pid_controller import CrazyflieController


# ── Configuration ─────────────────────────────────────────────────────

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
FPS = 60
SIM_DT = 0.001          # 1 kHz simulation
SIM_STEPS_PER_FRAME = int((1.0 / FPS) / SIM_DT)  # steps per render frame

# Control sensitivities
MAX_ROLL_ANGLE = 12.0 * math.pi / 180.0      # 12° max roll
MAX_PITCH_ANGLE = 12.0 * math.pi / 180.0     # 12° max pitch
MAX_YAW_RATE = 60.0 * math.pi / 180.0        # 60°/s max yaw rate
ALTITUDE_RATE = 0.5                            # m/s altitude change rate
INPUT_SMOOTHING = 0.12                         # exponential smoothing factor (0–1, lower = smoother)

# Colours (R, G, B, A) normalised
COL_SKY = (0.05, 0.05, 0.12, 1.0)
COL_GRID = (0.15, 0.25, 0.35, 0.6)
COL_GRID_AXES = (0.3, 0.4, 0.5, 0.8)
COL_ARM_FRONT = (0.95, 0.25, 0.2, 1.0)     # Red front arms
COL_ARM_REAR = (0.2, 0.4, 0.95, 1.0)       # Blue rear arms
COL_BODY = (0.2, 0.2, 0.2, 1.0)            # Dark body
COL_MOTOR_CW = (1.0, 0.85, 0.0, 0.9)       # Yellow CW
COL_MOTOR_CCW = (0.0, 0.95, 0.6, 0.9)      # Cyan CCW
COL_TRAIL = (0.3, 0.8, 1.0, 0.5)           # Light blue trail
COL_THRUST = (1.0, 0.5, 0.0, 0.6)          # Orange thrust indicator


# ── 3D Drawing Functions ──────────────────────────────────────────────

def draw_ground_grid(size=10, spacing=0.5):
    """Draw a ground-plane grid at z=0."""
    glBegin(GL_LINES)
    half = size / 2.0
    n = int(size / spacing)
    for i in range(n + 1):
        x = -half + i * spacing
        # Grid lines along Y
        if abs(x) < 0.01:
            glColor4f(*COL_GRID_AXES)
        else:
            glColor4f(*COL_GRID)
        glVertex3f(x, -half, 0)
        glVertex3f(x, half, 0)
        # Grid lines along X
        y = -half + i * spacing
        if abs(y) < 0.01:
            glColor4f(*COL_GRID_AXES)
        else:
            glColor4f(*COL_GRID)
        glVertex3f(-half, y, 0)
        glVertex3f(half, y, 0)
    glEnd()


def draw_circle(radius, segments=16):
    """Draw a circle in the XY plane at origin."""
    glBegin(GL_LINE_LOOP)
    for i in range(segments):
        theta = 2.0 * math.pi * i / segments
        glVertex3f(radius * math.cos(theta), radius * math.sin(theta), 0)
    glEnd()


def draw_filled_circle(radius, segments=16):
    """Draw a filled circle in the XY plane at origin."""
    glBegin(GL_TRIANGLE_FAN)
    glVertex3f(0, 0, 0)
    for i in range(segments + 1):
        theta = 2.0 * math.pi * i / segments
        glVertex3f(radius * math.cos(theta), radius * math.sin(theta), 0)
    glEnd()


def draw_crazyflie(arm_length, motor_speeds, max_speed):
    """
    Draw the Crazyflie quadrotor body.

    Coordinate convention for drawing: OpenGL Y-up
    (we convert from NED in the caller)
    """
    d = arm_length * math.sqrt(2) / 2  # arm projection on X/Y

    # Motor positions (top view): FR, BR, BL, FL
    motors = [
        (+d, +d),   # M1 front-right
        (-d, +d),   # M2 back-right
        (-d, -d),   # M3 back-left
        (+d, -d),   # M4 front-left
    ]

    is_front = [True, False, False, True]
    spin_dirs = [+1, -1, +1, -1]  # CW, CCW, CW, CCW
    motor_colors = [COL_MOTOR_CW, COL_MOTOR_CCW, COL_MOTOR_CW, COL_MOTOR_CCW]

    # ── Central body (small square) ──
    body_size = arm_length * 0.25
    glColor4f(*COL_BODY)
    glLineWidth(3.0)
    glBegin(GL_LINE_LOOP)
    glVertex3f(-body_size, -body_size, 0)
    glVertex3f(+body_size, -body_size, 0)
    glVertex3f(+body_size, +body_size, 0)
    glVertex3f(-body_size, +body_size, 0)
    glEnd()

    # ── Arms ──
    glLineWidth(3.0)
    for i, (mx, my) in enumerate(motors):
        color = COL_ARM_FRONT if is_front[i] else COL_ARM_REAR
        glColor4f(*color)
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(mx, my, 0)
        glEnd()

    # ── Motor discs ──
    prop_radius = arm_length * 0.35
    for i, (mx, my) in enumerate(motors):
        glPushMatrix()
        glTranslatef(mx, my, 0)

        # Motor disc outline
        glColor4f(*motor_colors[i])
        glLineWidth(2.0)
        draw_circle(prop_radius)

        # Thrust indicator (filled proportional to speed)
        speed_frac = min(motor_speeds[i] / max_speed, 1.0) if max_speed > 0 else 0
        if speed_frac > 0.01:
            glColor4f(COL_THRUST[0], COL_THRUST[1], COL_THRUST[2],
                      COL_THRUST[3] * speed_frac)
            draw_filled_circle(prop_radius * speed_frac)

        # Spin direction indicator (small arrow)
        arrow_r = prop_radius * 0.6
        glColor4f(*motor_colors[i])
        glLineWidth(1.5)
        angle = 0.3 * spin_dirs[i]
        glBegin(GL_LINES)
        glVertex3f(arrow_r * math.cos(0), arrow_r * math.sin(0), 0)
        glVertex3f(arrow_r * math.cos(angle), arrow_r * math.sin(angle), 0)
        glEnd()

        glPopMatrix()

    # ── Heading indicator (small triangle on nose) ──
    glColor4f(1.0, 1.0, 1.0, 0.9)
    nose = arm_length * 0.15
    glBegin(GL_TRIANGLES)
    glVertex3f(arm_length * 0.55, 0, 0)
    glVertex3f(arm_length * 0.35, -nose * 0.4, 0)
    glVertex3f(arm_length * 0.35, +nose * 0.4, 0)
    glEnd()

    glLineWidth(1.0)


def draw_trail(trail_points):
    """Draw a fading trail line from recent positions."""
    n = len(trail_points)
    if n < 2:
        return
    glLineWidth(1.5)
    glBegin(GL_LINE_STRIP)
    for i, pt in enumerate(trail_points):
        alpha = (i / n) * COL_TRAIL[3]
        glColor4f(COL_TRAIL[0], COL_TRAIL[1], COL_TRAIL[2], alpha)
        glVertex3f(pt[0], pt[1], pt[2])
    glEnd()
    glLineWidth(1.0)


def draw_altitude_line(x, y, altitude):
    """Draw a vertical dotted line from ground to drone."""
    glColor4f(0.5, 0.5, 0.5, 0.3)
    glLineWidth(1.0)
    glLineStipple(1, 0xAAAA)
    glEnable(GL_LINE_STIPPLE)
    glBegin(GL_LINES)
    glVertex3f(x, y, 0)
    glVertex3f(x, y, altitude)
    glEnd()
    glDisable(GL_LINE_STIPPLE)


# ── HUD Drawing ──────────────────────────────────────────────────────

def draw_hud(surface, font, font_small, telemetry):
    """Draw 2D HUD overlay on the pygame surface."""
    # Semi-transparent background panels
    panel_color = (10, 15, 30, 180)
    accent = (60, 180, 255)
    warn = (255, 180, 40)
    good = (40, 255, 120)
    white = (220, 225, 240)

    # ── Left panel: Flight Data ──
    panel_w, panel_h = 250, 200
    panel_surface = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    panel_surface.fill(panel_color)

    y_offset = 10
    lines = [
        (f"ALT   {telemetry['altitude']:7.2f} m", accent),
        (f"VEL   {telemetry['speed']:7.2f} m/s", white),
        (f"VZ    {telemetry['vz']:+7.2f} m/s", white),
        ("", white),
        (f"ROLL  {telemetry['roll']:+7.1f}°", white),
        (f"PITCH {telemetry['pitch']:+7.1f}°", white),
        (f"YAW   {telemetry['yaw']:+7.1f}°", white),
    ]
    for text, color in lines:
        if text:
            rendered = font.render(text, True, color)
            panel_surface.blit(rendered, (12, y_offset))
        y_offset += 22

    surface.blit(panel_surface, (10, 10))

    # ── Right panel: Motors ──
    panel_w2, panel_h2 = 220, 150
    panel_surface2 = pygame.Surface((panel_w2, panel_h2), pygame.SRCALPHA)
    panel_surface2.fill(panel_color)

    y_offset = 10
    motor_names = ["M1 (CW) ", "M2 (CCW)", "M3 (CW) ", "M4 (CCW)"]
    motor_colors_hud = [COL_MOTOR_CW, COL_MOTOR_CCW, COL_MOTOR_CW, COL_MOTOR_CCW]
    for i in range(4):
        rpm = telemetry['motor_rpms'][i]
        pct = min(rpm / 22000 * 100, 100)
        color_tuple = tuple(int(c * 255) for c in motor_colors_hud[i][:3])

        text = f"{motor_names[i]} {rpm:6.0f} RPM"
        rendered = font.render(text, True, color_tuple)
        panel_surface2.blit(rendered, (12, y_offset))

        # Bar graph
        bar_x, bar_y = 12, y_offset + 18
        bar_w = int((panel_w2 - 24) * pct / 100)
        pygame.draw.rect(panel_surface2, (*color_tuple, 80), (bar_x, bar_y, panel_w2 - 24, 4))
        pygame.draw.rect(panel_surface2, color_tuple, (bar_x, bar_y, bar_w, 4))

        y_offset += 30

    surface.blit(panel_surface2, (WINDOW_WIDTH - panel_w2 - 10, 10))

    # ── Bottom bar: Controls & FPS ──
    bar_h = 30
    bar_surface = pygame.Surface((WINDOW_WIDTH, bar_h), pygame.SRCALPHA)
    bar_surface.fill((10, 15, 30, 160))

    controls_text = "W/S: Pitch  A/D: Roll  Q/E: Yaw  ↑/↓: Alt  Space: Stop  R: Reset  C: Camera  Esc: Quit"
    rendered = font_small.render(controls_text, True, (150, 160, 180))
    bar_surface.blit(rendered, (12, 6))

    fps_text = f"FPS: {telemetry['fps']:.0f}  |  SIM: {telemetry['sim_time']:.1f}s"
    rendered = font_small.render(fps_text, True, accent)
    bar_surface.blit(rendered, (WINDOW_WIDTH - 200, 6))

    surface.blit(bar_surface, (0, WINDOW_HEIGHT - bar_h))

    # ── Emergency stop indicator ──
    if telemetry.get('motors_killed', False):
        warn_surface = pygame.Surface((300, 50), pygame.SRCALPHA)
        warn_surface.fill((180, 30, 30, 200))
        rendered = font.render("⚠ MOTORS KILLED ⚠", True, (255, 255, 255))
        w = rendered.get_width()
        warn_surface.blit(rendered, ((300 - w) // 2, 12))
        surface.blit(warn_surface, ((WINDOW_WIDTH - 300) // 2, 60))


# ── Main Application ────────────────────────────────────────────────

class RealtimeSimulation:
    """Real-time interactive Crazyflie simulation."""

    def __init__(self):
        # ── Simulation ──
        self.params = CrazyflieParams()
        self.config = SimConfig(
            dt=SIM_DT,
            enable_sensors=False,
            enable_ground_collision=True,
            enable_ground_effect=True,
            enable_aerodynamics=True,
            log_data=False,
        )
        self.sim = CrazyflieSim(self.params, self.config)
        self.controller = CrazyflieController(self.params)

        # ── Desired state ──
        self.desired_altitude = 0.5
        self.desired_roll = 0.0
        self.desired_pitch = 0.0
        self.desired_yaw_rate = 0.0
        self.motors_killed = False

        # ── Trail ──
        self.trail = []
        self.trail_max = 500

        # ── Camera ──
        self.camera_follow = True
        self.camera_angle = 45.0    # degrees around Z
        self.camera_elevation = 30.0  # degrees above horizontal
        self.camera_distance = 3.0

        # ── Timing ──
        self.sim_time = 0.0

    def reset(self):
        """Reset simulation to initial state."""
        self.sim.reset(position=np.array([0.0, 0.0, -0.5]))
        # Pre-spin motors
        a = self.params.pwm_thrust_a
        b = self.params.pwm_thrust_b
        hover_pwm = (-b + np.sqrt(b**2 + 4*a*(self.params.weight/4.0))) / (2*a)
        cmd = np.array([hover_pwm]*4)
        for _ in range(200):
            self.sim.step(cmd, command_type='pwm')
        self.sim.state[VEL] = np.zeros(3)
        self.sim.state[OMEGA] = np.zeros(3)

        self.controller.reset()
        self.desired_altitude = 0.5
        self.desired_roll = 0.0
        self.desired_pitch = 0.0
        self.desired_yaw_rate = 0.0
        self.motors_killed = False
        self.trail.clear()
        self.sim_time = 0.0

    def process_input(self, keys, dt):
        """Read keyboard state and update desired setpoints with smoothing."""
        alpha = INPUT_SMOOTHING

        # Compute raw targets from key state
        # Pitch (W/S)
        if keys[K_w]:
            target_pitch = MAX_PITCH_ANGLE
        elif keys[K_s]:
            target_pitch = -MAX_PITCH_ANGLE
        else:
            target_pitch = 0.0

        # Roll (A/D)
        if keys[K_a]:
            target_roll = -MAX_ROLL_ANGLE
        elif keys[K_d]:
            target_roll = MAX_ROLL_ANGLE
        else:
            target_roll = 0.0

        # Yaw (Q/E)
        if keys[K_q]:
            target_yaw_rate = -MAX_YAW_RATE
        elif keys[K_e]:
            target_yaw_rate = MAX_YAW_RATE
        else:
            target_yaw_rate = 0.0

        # Exponential smoothing (prevents step inputs to PID)
        self.desired_roll += alpha * (target_roll - self.desired_roll)
        self.desired_pitch += alpha * (target_pitch - self.desired_pitch)
        self.desired_yaw_rate += alpha * (target_yaw_rate - self.desired_yaw_rate)

        # Altitude (UP/DOWN arrows) — no smoothing needed
        if keys[K_UP]:
            self.desired_altitude += ALTITUDE_RATE * dt
        if keys[K_DOWN]:
            self.desired_altitude -= ALTITUDE_RATE * dt
        self.desired_altitude = max(0.0, min(self.desired_altitude, 5.0))

    def step_simulation(self):
        """
        Advance simulation by one frame worth of steps.

        PID runs once per frame (~60 Hz) and holds the motor commands
        constant across all sim sub-steps (1 kHz physics).
        """
        # Compute control ONCE per frame (not per sub-step)
        if self.motors_killed:
            motor_pwms = np.array([0.0, 0.0, 0.0, 0.0])
        else:
            frame_dt = SIM_STEPS_PER_FRAME * SIM_DT  # ~16.7 ms
            motor_pwms = self.controller.compute(
                self.sim.state,
                desired_roll=self.desired_roll,
                desired_pitch=self.desired_pitch,
                desired_yaw_rate=self.desired_yaw_rate,
                desired_altitude=self.desired_altitude,
                dt=frame_dt,
            )

        # Run physics sub-steps with held motor commands
        for _ in range(SIM_STEPS_PER_FRAME):
            self.sim.step(motor_pwms, command_type='pwm')
            self.sim_time += SIM_DT

        # Update trail (in OpenGL coordinates: X=forward, Y=right, Z=up)
        pos = self.sim.state[POS]
        gl_pos = (pos[0], pos[1], -pos[2])  # NED to OpenGL (flip Z)
        self.trail.append(gl_pos)
        if len(self.trail) > self.trail_max:
            self.trail.pop(0)

    def get_telemetry(self, fps):
        """Gather current telemetry for HUD."""
        state = self.sim.state
        euler = quat_to_euler(state[QUAT])
        speeds = self.sim.motors.get_speeds()

        return {
            'altitude': -state[2],
            'speed': np.linalg.norm(state[VEL]),
            'vz': -state[5],  # NED vz to climb rate
            'roll': math.degrees(euler[0]),
            'pitch': math.degrees(euler[1]),
            'yaw': math.degrees(euler[2]),
            'motor_rpms': speeds * 60 / (2 * math.pi),
            'fps': fps,
            'sim_time': self.sim_time,
            'motors_killed': self.motors_killed,
        }

    def setup_camera(self, drone_pos_gl):
        """Set up the OpenGL camera looking at the drone."""
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        if self.camera_follow:
            target = drone_pos_gl
        else:
            target = (0, 0, 0.5)

        # Orbit camera
        elev_rad = math.radians(self.camera_elevation)
        angle_rad = math.radians(self.camera_angle)
        cam_x = target[0] + self.camera_distance * math.cos(elev_rad) * math.cos(angle_rad)
        cam_y = target[1] + self.camera_distance * math.cos(elev_rad) * math.sin(angle_rad)
        cam_z = target[2] + self.camera_distance * math.sin(elev_rad)

        gluLookAt(cam_x, cam_y, cam_z,
                  target[0], target[1], target[2],
                  0, 0, 1)  # Z-up in OpenGL

    def render_3d(self):
        """Render the 3D scene."""
        state = self.sim.state
        pos = state[POS]
        q = state[QUAT]
        R = quat_to_rotation_matrix(q)

        # NED to OpenGL: X_gl = X_ned, Y_gl = Y_ned, Z_gl = -Z_ned
        gl_pos = (pos[0], pos[1], -pos[2])

        # ── Clear ──
        glClearColor(*COL_SKY)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # ── Camera ──
        self.setup_camera(gl_pos)

        # ── Enable features ──
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)

        # ── Ground grid ──
        draw_ground_grid()

        # ── Altitude line ──
        draw_altitude_line(gl_pos[0], gl_pos[1], gl_pos[2])

        # ── Trail ──
        draw_trail(self.trail)

        # ── Drone ──
        glPushMatrix()
        glTranslatef(*gl_pos)

        # Convert NED rotation matrix to OpenGL
        # NED→GL: flip Z axis in rotation
        R_gl = R.copy()
        R_gl[2, :] = -R_gl[2, :]   # flip Z row
        R_gl[:, 2] = -R_gl[:, 2]   # flip Z column

        # Build 4x4 matrix for OpenGL (column-major)
        m = np.eye(4)
        m[:3, :3] = R_gl
        glMultMatrixf(m.T.flatten())

        motor_speeds = self.sim.motors.get_speeds()
        draw_crazyflie(self.params.arm_length, motor_speeds,
                       self.params.motor_max_speed * 2 * math.pi / 60)

        glPopMatrix()

    def run(self):
        """Main application loop."""
        pygame.init()
        pygame.display.set_caption("Crazyflie 2.1 Digital Twin — Real-Time Simulation")

        # Set up OpenGL display
        screen = pygame.display.set_mode(
            (WINDOW_WIDTH, WINDOW_HEIGHT),
            DOUBLEBUF | OPENGL
        )

        # Projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(50, WINDOW_WIDTH / WINDOW_HEIGHT, 0.1, 100.0)

        glEnable(GL_DEPTH_TEST)

        # Fonts (need a separate 2D surface for HUD)
        hud_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)

        # We cannot directly use pygame fonts on an OpenGL surface,
        # so we render the HUD to a texture and overlay it
        pygame.font.init()
        font = pygame.font.SysFont("Consolas", 16)
        font_small = pygame.font.SysFont("Consolas", 13)

        clock = pygame.time.Clock()

        # Initial reset
        self.reset()

        print("\n" + "=" * 60)
        print("  CRAZYFLIE REAL-TIME SIMULATION")
        print("  Controls: WASD + QE + ↑↓ + Space + R")
        print("=" * 60 + "\n")

        running = True
        while running:
            dt_frame = clock.tick(FPS) / 1000.0

            # ── Events ──
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False
                    elif event.key == K_SPACE:
                        self.motors_killed = not self.motors_killed
                    elif event.key == K_r:
                        self.reset()
                    elif event.key == K_c:
                        self.camera_follow = not self.camera_follow
                # Mouse wheel for camera zoom
                elif event.type == MOUSEWHEEL:
                    self.camera_distance = max(1.0, min(15.0,
                        self.camera_distance - event.y * 0.3))

            # Mouse drag for camera orbit
            mouse_buttons = pygame.mouse.get_pressed()
            if mouse_buttons[2]:  # Right mouse button
                rel = pygame.mouse.get_rel()
                self.camera_angle -= rel[0] * 0.3
                self.camera_elevation = max(-10, min(89,
                    self.camera_elevation + rel[1] * 0.3))
            else:
                pygame.mouse.get_rel()  # consume to prevent jump

            # ── Input ──
            keys = pygame.key.get_pressed()
            self.process_input(keys, dt_frame)

            # ── Simulate ──
            self.step_simulation()

            # ── Render 3D ──
            self.render_3d()

            # ── HUD (2D overlay) ──
            telemetry = self.get_telemetry(clock.get_fps())

            # Read OpenGL framebuffer, draw HUD, write back
            # This is the simplest approach for mixing pygame 2D + OpenGL
            hud_surface.fill((0, 0, 0, 0))
            draw_hud(hud_surface, font, font_small, telemetry)

            # Convert HUD surface to OpenGL texture and overlay
            hud_data = pygame.image.tostring(hud_surface, "RGBA", True)

            # Switch to 2D mode for HUD
            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            glOrtho(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT, -1, 1)
            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glLoadIdentity()

            glDisable(GL_DEPTH_TEST)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            # Draw HUD as raster
            glRasterPos2i(0, 0)
            glDrawPixels(WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, hud_data)

            glEnable(GL_DEPTH_TEST)
            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            glPopMatrix()

            # ── Swap ──
            pygame.display.flip()

        pygame.quit()


# ── Entry Point ──────────────────────────────────────────────────────

def main():
    app = RealtimeSimulation()
    app.run()


if __name__ == "__main__":
    main()
