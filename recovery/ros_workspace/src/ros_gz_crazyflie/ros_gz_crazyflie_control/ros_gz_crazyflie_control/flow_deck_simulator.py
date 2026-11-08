"""Flow Deck v2 optical-flow channel emulator.

The real PMW3901 reports accumulated image motion (pixel deltas) and the
firmware consumes it together with the measurement interval and ToF height.
Gazebo Fortress has no PMW3901 sensor, so this node derives the equivalent
measurement from simulated odometry and the downward range sensor.
"""

import math

import rclpy
from geometry_msgs.msg import Vector3Stamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Range
from rclpy.node import Node


class FlowDeckSimulator(Node):
    def __init__(self):
        super().__init__('flow_deck_simulator')
        self.declare_parameter('robot_prefix', '/crazyflie')
        self.declare_parameter('focal_length_px', 160.0)
        prefix = self.get_parameter('robot_prefix').value
        self.focal_length_px = float(self.get_parameter('focal_length_px').value)
        self.height = 0.5
        self.last_time = None
        self.last_x = None
        self.last_y = None
        self.odom_sub = self.create_subscription(Odometry, prefix + '/odom', self.odom_cb, 20)
        self.range_sub = self.create_subscription(Range, prefix + '/flow/range', self.range_cb, 10)
        self.flow_pub = self.create_publisher(Vector3Stamped, prefix + '/flow', 20)
        self.velocity_pub = self.create_publisher(Vector3Stamped, prefix + '/flow/velocity', 20)

    def range_cb(self, msg):
        if math.isfinite(msg.range) and msg.range > 0.0:
            self.height = msg.range

    def odom_cb(self, msg):
        now = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        if self.last_time is None:
            self.last_time, self.last_x, self.last_y = now, x, y
            return
        dt = now - self.last_time
        if dt <= 1e-6 or dt > 0.2:
            self.last_time, self.last_x, self.last_y = now, x, y
            return

        # Camera image motion is opposite ground motion.  This is the
        # PMW3901-style accumulated pixel convention used by the firmware.
        dx = x - self.last_x
        dy = y - self.last_y
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        # PMW3901 reports motion in its body/image axes, not world axes.
        body_dx = math.cos(yaw) * dx + math.sin(yaw) * dy
        body_dy = -math.sin(yaw) * dx + math.cos(yaw) * dy
        body_vx = (math.cos(yaw) * msg.twist.twist.linear.x
                   + math.sin(yaw) * msg.twist.twist.linear.y)
        body_vy = (-math.sin(yaw) * msg.twist.twist.linear.x
                   + math.cos(yaw) * msg.twist.twist.linear.y)
        scale = self.focal_length_px / max(self.height, 0.08)
        flow = Vector3Stamped()
        flow.header = msg.header
        flow.vector.x = -scale * body_dx
        flow.vector.y = -scale * body_dy
        flow.vector.z = dt
        self.flow_pub.publish(flow)

        velocity = Vector3Stamped()
        velocity.header = msg.header
        velocity.vector.x = -scale * body_vx
        velocity.vector.y = -scale * body_vy
        velocity.vector.z = 0.0
        self.velocity_pub.publish(velocity)
        self.last_time, self.last_x, self.last_y = now, x, y


def main(args=None):
    rclpy.init(args=args)
    node = FlowDeckSimulator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
