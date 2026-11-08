"""Passive single-frame tutor-wire perception characterization node."""

import csv
import math
from pathlib import Path
import threading

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


class BaselinePerception(Node):
    """Detect vertical tutor-wire contrast independently in each frame."""

    ROW_Y = (-3.0, -1.5, 0.0, 1.5, 3.0)
    WIRE_X = tuple(round(-16.8 + 0.3 * i, 2) for i in range(113))
    WIRE_BOTTOM_Z = 0.65
    WIRE_TOP_Z = 2.25
    WIRE_LENGTH_M = WIRE_TOP_Z - WIRE_BOTTOM_Z

    def __init__(self):
        super().__init__('baseline_perception')
        self.declare_parameter('input_topic', '/crazyflie/camera/image_delayed')
        self.declare_parameter('log_path', '/tmp/crazyflie_perception_baseline.csv')
        self.declare_parameter('image_width', 320)
        self.declare_parameter('image_height', 320)
        self.declare_parameter('horizontal_fov_rad', 1.518436)
        self.declare_parameter('wire_row_margin_m', 0.9)
        self.declare_parameter('min_confidence', 0.02)
        self.input_topic = str(self.get_parameter('input_topic').value)
        self.log_path = Path(str(self.get_parameter('log_path').value))
        self.width = int(self.get_parameter('image_width').value)
        self.height = int(self.get_parameter('image_height').value)
        self.fx = (self.width / 2.0) / math.tan(
            float(self.get_parameter('horizontal_fov_rad').value) / 2.0)
        self.fy = self.fx
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0
        self.min_confidence = float(self.get_parameter('min_confidence').value)
        self.lock = threading.Lock()
        self.odom = None
        self.fsm_state = 'UNKNOWN'
        self.aisle = ''
        self.monitor_event_id = '0'
        self.monitoring_point_id = ''
        self.crop_section_id = ''
        self.crop_side = ''
        self.desired_yaw = ''
        self.observation_active = '0'
        self.sequence_id = 0
        self.csv_file = None
        self.writer = None
        self._open_log()
        self.create_subscription(Image, self.input_topic, self.image_callback, 10)
        self.create_subscription(Odometry, '/crazyflie/odom', self.odom_callback, 20)
        self.create_subscription(String, '/greenhouse/fsm_state', self.fsm_callback, 20)
        self.get_logger().info(f'Passive perception: {self.input_topic} -> {self.log_path}')

    def _open_log(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_file = self.log_path.open('w', newline='', buffering=1)
        fields = [
            'frame_sequence_id', 'original_capture_timestamp_s', 'arrival_timestamp_s',
            'processing_timestamp_s', 'age_of_information_ms', 'frame_accepted',
            'rejection_reason', 'detection_valid', 'estimated_wire_u_px',
            'estimated_wire_v_px', 'estimated_wire_distance_m', 'confidence',
            'wire_gt_x_m', 'wire_gt_y_m', 'wire_gt_z_m', 'wire_gt_forward_m',
            'wire_gt_lateral_m', 'wire_gt_distance_m', 'wire_gt_u_px', 'wire_gt_v_px',
            'distance_error_m', 'image_position_error_px', 'gt_pose_x_m', 'gt_pose_y_m',
            'gt_pose_z_m', 'gt_pose_yaw_rad', 'odom_pose_x_m', 'odom_pose_y_m',
            'odom_pose_z_m', 'odom_pose_yaw_rad', 'velocity_x_mps', 'velocity_y_mps',
            'velocity_z_mps', 'fsm_state', 'aisle', 'input_topic',
            'monitoring_event_id', 'monitoring_point_id', 'aisle_id', 'crop_row_id',
            'crop_section_id', 'crop_side', 'desired_yaw_rad', 'observation_active',
            'nominal_monitor_x_m', 'position_error_to_monitor_point_m', 'yaw_error_rad',
        ]
        self.writer = csv.DictWriter(self.csv_file, fieldnames=fields)
        self.writer.writeheader()

    @staticmethod
    def stamp_ns(stamp):
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def odom_callback(self, msg):
        pose = msg.pose.pose
        q = pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        with self.lock:
            self.odom = {
                'x': pose.position.x, 'y': pose.position.y, 'z': pose.position.z,
                'yaw': yaw, 'vx': msg.twist.twist.linear.x,
                'vy': msg.twist.twist.linear.y, 'vz': msg.twist.twist.linear.z,
            }

    def fsm_callback(self, msg):
        fields = msg.data.split('|')
        with self.lock:
            self.fsm_state = fields[0] if fields else 'UNKNOWN'
            values = {f.split('=', 1)[0]: f.split('=', 1)[1]
                      for f in fields[1:] if '=' in f}
            self.aisle = values.get('lane', '')
            self.monitor_event_id = values.get('monitor_event_id', '0')
            self.monitoring_point_id = values.get('monitoring_point_id', '')
            self.crop_section_id = values.get('crop_section_id', '')
            self.crop_side = values.get('crop_side', '')
            self.desired_yaw = values.get('desired_yaw', '')
            self.observation_active = values.get('observation_active', '0')
            self.monitor_x = values.get('monitor_x', '')
            self.monitor_y = values.get('monitor_y', '')

    @staticmethod
    def image_array(msg):
        channels = 3 if msg.encoding in ('rgb8', 'bgr8') else 1
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        row_bytes = int(msg.step) if msg.step else msg.width * channels
        rows = raw[:msg.height * row_bytes].reshape(msg.height, row_bytes)
        if channels == 1:
            return rows[:, :msg.width].astype(np.float32)
        pixels = rows[:, :msg.width * channels].reshape(msg.height, msg.width, channels)
        return pixels.mean(axis=2).astype(np.float32)

    def choose_ground_truth_wire(self, odom):
        if odom is None:
            return None
        row_y = min(self.ROW_Y, key=lambda y: abs(y - odom['y']))
        candidates = []
        cyaw, syaw = math.cos(odom['yaw']), math.sin(odom['yaw'])
        for x in self.WIRE_X:
            dx, dy = x - odom['x'], row_y - odom['y']
            forward = cyaw * dx + syaw * dy
            lateral = -syaw * dx + cyaw * dy
            if forward > 0.25 and abs(lateral) < 12.0:
                candidates.append((forward, abs(lateral), x, row_y, forward, lateral))
        if not candidates:
            return None
        _, _, x, y, forward, lateral = min(candidates)
        distance = math.sqrt(forward * forward + lateral * lateral +
                             (1.45 - odom['z']) ** 2)
        return {'x': x, 'y': y, 'z': 1.45, 'forward': forward,
                'lateral': lateral, 'distance': distance}

    def project_wire(self, wire, odom):
        if wire is None or odom is None or wire['forward'] <= 0.0:
            return None
        return (self.cx + self.fx * wire['lateral'] / wire['forward'],
                self.cy - self.fy * (wire['z'] - odom['z']) / wire['forward'])

    def detect_wire(self, image):
        """Transparent single-frame vertical-contrast baseline detector."""
        h, w = image.shape
        if h < 8 or w < 8:
            return None
        left, right = 4, w - 4
        top, bottom = max(4, h // 20), min(h - 4, h - h // 20)
        center = image[top:bottom, left:right]
        neighbors = 0.5 * (image[top:bottom, left - 2:right - 2] +
                            image[top:bottom, left + 2:right + 2])
        contrast = np.abs(center - neighbors)
        profile = contrast.mean(axis=0)
        if profile.size < 3:
            return None
        peak = int(np.argmax(profile))
        value = float(profile[peak])
        baseline = float(np.median(profile)) + 1e-6
        confidence = max(0.0, min(1.0, (value / baseline - 1.0) / 20.0))
        if confidence < self.min_confidence:
            return None
        x = left + peak
        row_profile = contrast[:, peak]
        threshold = max(float(row_profile.max()) * 0.35, 1e-3)
        ys = np.flatnonzero(row_profile >= threshold) + top
        if ys.size < 2:
            return None
        weights = row_profile[ys - top]
        v = float(np.average(ys, weights=weights))
        span = max(2.0, float(ys.max() - ys.min()))
        estimated_distance = self.fy * self.WIRE_LENGTH_M / span
        return float(x), v, estimated_distance, confidence

    @staticmethod
    def fmt(value):
        return '' if value is None or not math.isfinite(float(value)) else f'{float(value):.6f}'

    def image_callback(self, msg):
        arrival_ns = self.get_clock().now().nanoseconds
        capture_ns = self.stamp_ns(msg.header.stamp)
        with self.lock:
            odom = dict(self.odom) if self.odom else None
            fsm_state, aisle = self.fsm_state, self.aisle
            event_id, point_id = self.monitor_event_id, self.monitoring_point_id
            crop_section, crop_side = self.crop_section_id, self.crop_side
            desired_yaw, observation_active = self.desired_yaw, self.observation_active
            monitor_x = getattr(self, 'monitor_x', '')
            monitor_y = getattr(self, 'monitor_y', '')
            self.sequence_id += 1
            sequence_id = self.sequence_id
        capture_s = capture_ns / 1e9 if capture_ns else None
        aoi_ms = (arrival_ns - capture_ns) / 1e6 if capture_ns else None
        wire = self.choose_ground_truth_wire(odom)
        gt_projection = self.project_wire(wire, odom)
        detection, reason = None, ''
        try:
            detection = self.detect_wire(self.image_array(msg))
        except (ValueError, TypeError) as exc:
            reason = f'image_decode:{type(exc).__name__}'
        if detection is None and not reason:
            reason = 'no_vertical_wire'
        processing_ns = self.get_clock().now().nanoseconds
        est_u = est_v = est_distance = confidence = None
        if detection:
            est_u, est_v, est_distance, confidence = detection
        gt_u, gt_v = gt_projection if gt_projection else (None, None)
        distance_error = est_distance - wire['distance'] if detection and wire else None
        position_error = (math.hypot(est_u - gt_u, est_v - gt_v)
                          if detection and gt_u is not None else None)
        row = {
            'frame_sequence_id': sequence_id,
            'original_capture_timestamp_s': self.fmt(capture_s),
            'arrival_timestamp_s': self.fmt(arrival_ns / 1e9),
            'processing_timestamp_s': self.fmt(processing_ns / 1e9),
            'age_of_information_ms': self.fmt(aoi_ms), 'frame_accepted': 1,
            'rejection_reason': reason, 'detection_valid': int(detection is not None),
            'estimated_wire_u_px': self.fmt(est_u), 'estimated_wire_v_px': self.fmt(est_v),
            'estimated_wire_distance_m': self.fmt(est_distance),
            'confidence': self.fmt(confidence),
            'wire_gt_x_m': self.fmt(wire['x'] if wire else None),
            'wire_gt_y_m': self.fmt(wire['y'] if wire else None),
            'wire_gt_z_m': self.fmt(wire['z'] if wire else None),
            'wire_gt_forward_m': self.fmt(wire['forward'] if wire else None),
            'wire_gt_lateral_m': self.fmt(wire['lateral'] if wire else None),
            'wire_gt_distance_m': self.fmt(wire['distance'] if wire else None),
            'wire_gt_u_px': self.fmt(gt_u), 'wire_gt_v_px': self.fmt(gt_v),
            'distance_error_m': self.fmt(distance_error),
            'image_position_error_px': self.fmt(position_error),
            'gt_pose_x_m': self.fmt(odom['x'] if odom else None),
            'gt_pose_y_m': self.fmt(odom['y'] if odom else None),
            'gt_pose_z_m': self.fmt(odom['z'] if odom else None),
            'gt_pose_yaw_rad': self.fmt(odom['yaw'] if odom else None),
            'odom_pose_x_m': self.fmt(odom['x'] if odom else None),
            'odom_pose_y_m': self.fmt(odom['y'] if odom else None),
            'odom_pose_z_m': self.fmt(odom['z'] if odom else None),
            'odom_pose_yaw_rad': self.fmt(odom['yaw'] if odom else None),
            'velocity_x_mps': self.fmt(odom['vx'] if odom else None),
            'velocity_y_mps': self.fmt(odom['vy'] if odom else None),
            'velocity_z_mps': self.fmt(odom['vz'] if odom else None),
            'fsm_state': fsm_state, 'aisle': aisle, 'input_topic': self.input_topic,
            'monitoring_event_id': event_id, 'monitoring_point_id': point_id,
            'aisle_id': aisle, 'crop_row_id': f'row_{int(aisle):02d}' if aisle.isdigit() else '',
            'crop_section_id': crop_section, 'crop_side': crop_side,
            'desired_yaw_rad': desired_yaw, 'observation_active': observation_active,
            'nominal_monitor_x_m': monitor_x,
            'position_error_to_monitor_point_m': self.fmt(
                math.hypot(float(monitor_x) - odom['x'], float(monitor_y) - odom['y'])
                if monitor_x and monitor_y and odom else None),
            'yaw_error_rad': self.fmt(
                ((float(desired_yaw) - odom['yaw'] + math.pi) % (2 * math.pi) - math.pi)
                if desired_yaw and odom else None),
        }
        self.writer.writerow(row)

    def destroy_node(self):
        if self.csv_file:
            self.csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BaselinePerception()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
