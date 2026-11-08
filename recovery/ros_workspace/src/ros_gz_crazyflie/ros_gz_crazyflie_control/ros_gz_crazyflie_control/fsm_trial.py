"""First ROS/Gazebo trial of the greenhouse lane FSM.

This is intentionally a controlled integration trial: odometry supplies the
state and perfect row/end-of-row information is derived from the known world
geometry. Camera perception and CBF filtering are future replacements for
those pieces.
"""

from enum import Enum, auto
import json
import math
from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import Image
from std_msgs.msg import Float64, String


class State(Enum):
    TAKEOFF = auto()
    ENTER_LANE = auto()
    IN_LANE = auto()
    APPROACH_MONITOR_POINT = auto()
    OBSERVE = auto()
    END_OF_LANE = auto()
    TURNING = auto()
    LAND = auto()
    DONE = auto()


class GazeboFSMTrial(Node):
    """Drive the Crazyflie through all 20 Gazebo aisles."""

    def __init__(self):
        super().__init__('greenhouse_fsm_trial')
        self.declare_parameter('robot_prefix', '/crazyflie')
        # Twenty crop rows create nineteen internal aisles. The old twentieth
        # path landed at y=15.0, on the north wall, so it is intentionally
        # excluded from coverage.
        self.declare_parameter('n_lanes', 4)
        self.declare_parameter('lane_y_first', -2.25)
        self.declare_parameter('lane_pitch', 1.5)
        self.declare_parameter('x_min', -18.0)
        self.declare_parameter('x_max', 18.0)
        self.declare_parameter('cruise_speed', 0.35)
        self.declare_parameter('position_tolerance', 0.35)
        self.declare_parameter('takeoff_height', 0.5)
        self.declare_parameter('takeoff_settle_time', 0.75)
        self.declare_parameter('yaw_tolerance', 0.12)
        self.declare_parameter('horizontal_acceleration_limit', 0.8)
        default_monitoring_config = str(Path(get_package_share_directory(
            'ros_gz_crazyflie_control')) / 'config' / 'monitoring_points.json')
        self.declare_parameter('monitoring_config', default_monitoring_config)
        self.declare_parameter('observation_duration', 1.5)
        self.declare_parameter('observation_position_tolerance', 0.30)
        self.declare_parameter('observation_yaw_tolerance', 0.15)
        self.declare_parameter('observation_min_frames', 3)

        prefix = self.get_parameter('robot_prefix').value
        self.n_lanes = int(self.get_parameter('n_lanes').value)
        self.first_y = float(self.get_parameter('lane_y_first').value)
        self.pitch = float(self.get_parameter('lane_pitch').value)
        self.x_min = float(self.get_parameter('x_min').value)
        self.x_max = float(self.get_parameter('x_max').value)
        self.speed = float(self.get_parameter('cruise_speed').value)
        self.tolerance = float(self.get_parameter('position_tolerance').value)
        self.takeoff_height = float(self.get_parameter('takeoff_height').value)
        self.takeoff_settle_time = float(
            self.get_parameter('takeoff_settle_time').value)
        self.yaw_tolerance = float(self.get_parameter('yaw_tolerance').value)
        self.horizontal_acceleration_limit = float(
            self.get_parameter('horizontal_acceleration_limit').value)
        self.observation_duration = float(self.get_parameter('observation_duration').value)
        self.observation_position_tolerance = float(
            self.get_parameter('observation_position_tolerance').value)
        self.observation_yaw_tolerance = float(
            self.get_parameter('observation_yaw_tolerance').value)
        self.monitoring_points = self.load_monitoring_points(
            str(self.get_parameter('monitoring_config').value))

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.target_pub = self.create_publisher(PointStamped, '/greenhouse/fsm_target', 10)
        self.state_pub = self.create_publisher(String, '/greenhouse/fsm_state', 10)
        self.yaw_pub = self.create_publisher(Float64, '/greenhouse/fsm_desired_yaw', 10)
        self.odom_sub = self.create_subscription(
            Odometry, prefix + '/odom', self.odom_callback, 10)
        self.camera_sub = self.create_subscription(
            Image, prefix + '/camera/image', self.camera_callback, 10)
        self.timer = self.create_timer(0.05, self.control_loop)

        self.state = State.TAKEOFF
        self.lane = 0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.vz = 0.0
        self.yaw = 0.0
        self.have_odom = False
        self.state_started = self.get_clock().now()
        self.last_reported_state = None
        self.last_cmd_x = 0.0
        self.last_cmd_y = 0.0
        self.monitor_index = 0
        self.monitor_event_id = 0
        self.active_monitor = None
        self.monitor_event_started = None
        self.observation_frame_count = 0

        self.get_logger().info(
            f'FSM trial ready: {self.n_lanes} aisles, '
            f'{len(self.monitoring_points)} monitoring points')

    def load_monitoring_points(self, config_path):
        if not config_path:
            return []
        try:
            with Path(config_path).open() as handle:
                points = json.load(handle)
            required = {'aisle', 'longitudinal_position', 'crop_side',
                        'observation_yaw_offset', 'crop_section_id'}
            result = []
            for point in points:
                if not required.issubset(point):
                    raise ValueError(f'monitoring point missing {required - set(point)}')
                result.append(point)
            return sorted(result, key=lambda p: (int(p['aisle']), float(p['longitudinal_position'])))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.get_logger().error(f'Could not load monitoring_config={config_path}: {exc}')
            return []

    def odom_callback(self, msg):
        pose = msg.pose.pose
        self.x = pose.position.x
        self.y = pose.position.y
        self.z = pose.position.z
        self.vz = msg.twist.twist.linear.z
        q = pose.orientation
        self.yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.have_odom = True

    def camera_callback(self, _msg):
        if self.state == State.OBSERVE:
            self.observation_frame_count += 1

    def set_state(self, state):
        if state != self.state:
            self.state = state
            self.state_started = self.get_clock().now()
            self.get_logger().info(
                f'FSM -> {state.name} (lane {self.lane + 1}/{self.n_lanes})')

    def elapsed(self):
        return (self.get_clock().now() - self.state_started).nanoseconds * 1e-9

    def lane_y(self):
        return self.first_y + self.pitch * self.lane

    def lane_yaw(self):
        # Face the travel direction: east on odd-numbered lanes and west on
        # even-numbered lanes. The turn between lanes is therefore 180 deg.
        return 0.0 if self.lane % 2 == 0 else 3.141592653589793

    def next_monitor_for_lane(self):
        aisle = self.lane + 1
        candidates = [(i, p) for i, p in enumerate(self.monitoring_points)
                      if int(p['aisle']) == aisle and i >= self.monitor_index]
        if not candidates:
            return None
        return candidates[0]

    def monitor_target_x(self):
        return float(self.active_monitor['longitudinal_position'])

    def monitor_yaw(self):
        return (self.lane_yaw() + math.radians(
            float(self.active_monitor['observation_yaw_offset']))) % (2.0 * math.pi)

    def yaw_error(self, desired):
        return (desired - self.yaw + math.pi) % (2.0 * math.pi) - math.pi

    def ramp_horizontal_command(self, cmd):
        # The FSM runs at 20 Hz. Limiting the command step prevents the
        # multicopter controller from demanding an instantaneous tilt when
        # takeoff, lane entry, or a headland turn changes state.
        max_step = self.horizontal_acceleration_limit * 0.05
        dx = max(-max_step, min(max_step, cmd.linear.x - self.last_cmd_x))
        dy = max(-max_step, min(max_step, cmd.linear.y - self.last_cmd_y))
        self.last_cmd_x += dx
        self.last_cmd_y += dy
        cmd.linear.x = self.last_cmd_x
        cmd.linear.y = self.last_cmd_y

    def control_loop(self):
        cmd = Twist()
        if not self.have_odom:
            self.cmd_pub.publish(cmd)
            return

        # control_services uses a positive vertical command both to initiate
        # takeoff and to keep its altitude PID in HOVER mode.
        if self.state not in (State.TAKEOFF, State.LAND, State.DONE):
            cmd.linear.z = 0.1

        if self.state == State.TAKEOFF:
            if self.z < self.takeoff_height or abs(self.vz) > 0.12:
                cmd.linear.z = 0.25
            elif (self.elapsed() < self.takeoff_settle_time
                  or abs(self.vz) > 0.06):
                # Keep the altitude controller active while the vehicle
                # settles before allowing horizontal motion to begin.
                cmd.linear.z = 0.1
            else:
                self.set_state(State.ENTER_LANE)

        elif self.state == State.ENTER_LANE:
            dy = self.lane_y() - self.y
            if abs(self.x - self.x_min) < self.tolerance and abs(dy) < self.tolerance:
                self.set_state(State.IN_LANE)
            else:
                cmd.linear.x = max(-self.speed, min(self.speed,
                    self.speed * (self.x_min - self.x)))
                cmd.linear.y = max(-self.speed, min(self.speed,
                    self.speed * dy))

        elif self.state == State.IN_LANE:
            direction = 1.0 if self.lane % 2 == 0 else -1.0
            target_x = self.x_max if direction > 0 else self.x_min
            next_monitor = self.next_monitor_for_lane()
            monitor_ahead = (next_monitor is not None and
                             ((direction > 0 and self.x >= float(next_monitor[1]['longitudinal_position'])) or
                              (direction < 0 and self.x <= float(next_monitor[1]['longitudinal_position']))))
            if monitor_ahead:
                self.monitor_index, self.active_monitor = next_monitor
                self.monitor_event_id += 1
                self.monitor_event_started = None
                self.set_state(State.APPROACH_MONITOR_POINT)
            elif abs(target_x - self.x) < self.tolerance:
                self.set_state(State.END_OF_LANE)
            else:
                cmd.linear.x = direction * self.speed
                cmd.linear.y = max(-0.15, min(0.15,
                    0.8 * (self.lane_y() - self.y)))

        elif self.state == State.APPROACH_MONITOR_POINT:
            direction = 1.0 if self.lane % 2 == 0 else -1.0
            dx = self.monitor_target_x() - self.x
            dy = self.lane_y() - self.y
            cmd.linear.x = max(-0.18, min(0.18, 0.8 * dx))
            cmd.linear.y = max(-0.15, min(0.15, 0.8 * dy))
            if (abs(dx) < self.observation_position_tolerance and
                    abs(dy) < self.observation_position_tolerance):
                self.set_state(State.OBSERVE)
                self.monitor_event_started = self.get_clock().now()
                self.observation_frame_count = 0

        elif self.state == State.OBSERVE:
            # Stop translating while the existing yaw controller turns the camera.
            if (abs(self.yaw_error(self.monitor_yaw())) <= self.observation_yaw_tolerance and
                    self.elapsed() >= self.observation_duration and
                    self.observation_frame_count >= int(self.get_parameter(
                        'observation_min_frames').value)):
                self.monitor_index += 1
                self.active_monitor = None
                self.monitor_event_started = None
                self.set_state(State.IN_LANE)

        elif self.state == State.END_OF_LANE:
            if self.elapsed() > 0.8:
                if self.lane == self.n_lanes - 1:
                    self.set_state(State.LAND)
                else:
                    self.lane += 1
                    self.set_state(State.TURNING)

        elif self.state == State.TURNING:
            dy = self.lane_y() - self.y
            yaw_error = (self.lane_yaw() - self.yaw + math.pi) % (
                2.0 * math.pi) - math.pi
            if abs(yaw_error) > self.yaw_tolerance:
                # Rotate in place at the headland.  Translating while the
                # body/world-frame command basis is rotating causes a jump.
                pass
            elif abs(dy) < self.tolerance:
                self.set_state(State.ENTER_LANE)
            else:
                cmd.linear.y = max(-self.speed, min(self.speed, 0.8 * dy))

        elif self.state == State.LAND:
            if self.z > 0.08:
                cmd.linear.z = -0.2
            else:
                self.set_state(State.DONE)

        if self.state != self.last_reported_state:
            self.last_reported_state = self.state
        target = PointStamped()
        target.header.stamp = self.get_clock().now().to_msg()
        target.header.frame_id = 'greenhouse'
        target.point.x = self.x_max if self.state == State.IN_LANE and self.lane % 2 == 0 else self.x_min
        if self.state == State.IN_LANE and self.lane % 2 == 1:
            target.point.x = self.x_min
        if self.state in (State.APPROACH_MONITOR_POINT, State.OBSERVE) and self.active_monitor:
            target.point.x = self.monitor_target_x()
        if self.state in (State.ENTER_LANE, State.TURNING):
            target.point.x = self.x_min if self.state == State.ENTER_LANE else self.x
        target.point.y = self.lane_y()
        target.point.z = self.takeoff_height
        self.target_pub.publish(target)
        state_msg = String()
        desired_yaw = self.monitor_yaw() if self.active_monitor else self.lane_yaw()
        event_id = self.monitor_event_id if self.active_monitor or self.state == State.OBSERVE else 0
        point_id = self.active_monitor.get('monitoring_point_id', '') if self.active_monitor else ''
        section = self.active_monitor.get('crop_section_id', '') if self.active_monitor else ''
        side = self.active_monitor.get('crop_side', '') if self.active_monitor else ''
        state_msg.data = (f'{self.state.name}|lane={self.lane + 1}|error_y={self.lane_y() - self.y:.3f}'
                          f'|monitor_event_id={event_id}|monitoring_point_id={point_id}'
                          f'|crop_section_id={section}|crop_side={side}'
                          f'|desired_yaw={desired_yaw:.6f}|observation_active={int(self.state == State.OBSERVE)}'
                          f'|monitor_x={self.monitor_target_x() if self.active_monitor else ""}'
                          f'|monitor_y={self.lane_y() if self.active_monitor else ""}')
        self.state_pub.publish(state_msg)
        yaw_msg = Float64()
        yaw_msg.data = desired_yaw
        self.yaw_pub.publish(yaw_msg)
        self.ramp_horizontal_command(cmd)
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = GazeboFSMTrial()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.cmd_pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
