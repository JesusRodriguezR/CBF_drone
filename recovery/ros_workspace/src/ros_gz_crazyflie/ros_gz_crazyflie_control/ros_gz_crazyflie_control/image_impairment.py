"""Configurable image-delivery impairment layer.

The raw camera image and its header are never modified.  Frames are queued
after capture, optionally dropped, and published on a separate delayed topic.
The node also publishes JSON status and writes a CSV experiment log.
"""

import csv
import heapq
import json
import math
import random
import threading

from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger

import rclpy
from rclpy.node import Node


class ImageImpairment(Node):
    def __init__(self):
        super().__init__('image_impairment')
        self.declare_parameter('input_topic', '/crazyflie/camera/image')
        self.declare_parameter('output_topic', '/crazyflie/camera/image_delayed')
        self.declare_parameter('status_topic', '/crazyflie/camera/impairment_status')
        self.declare_parameter('log_path', '/tmp/crazyflie_image_impairment.csv')
        self.declare_parameter('delay_mode', 'disabled')
        self.declare_parameter('base_delay_ms', 0.0)
        self.declare_parameter('jitter_ms', 0.0)
        self.declare_parameter('jitter_distribution', 'uniform')
        self.declare_parameter('packet_loss_enabled', False)
        self.declare_parameter('packet_loss_probability', 0.0)
        self.declare_parameter('reject_out_of_order', True)

        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.status_topic = self.get_parameter('status_topic').value
        self.log_path = self.get_parameter('log_path').value
        self._queue = []
        self._order = 0
        self._next_frame_id = 0
        self._latest_capture_ns = None
        self._last_frame_id = None
        self._last_delay_ms = 0.0
        self._received = 0
        self._dropped = 0
        self._frames_seen = 0
        self._packet_loss_dropped = 0
        self._out_of_order = 0
        self._lock = threading.Lock()

        self.publisher = self.create_publisher(Image, self.output_topic, 10)
        self.status_publisher = self.create_publisher(String, self.status_topic, 10)
        self.subscription = self.create_subscription(
            Image, self.input_topic, self.image_callback, 20)
        self.create_timer(0.005, self.delivery_callback)
        self.create_timer(0.1, self.status_callback)
        self.create_service(Trigger, '~/reset_statistics', self.reset_statistics)
        self.add_on_set_parameters_callback(self.parameter_callback)

        self._open_log()
        self.get_logger().info(
            f'Image impairment: {self.input_topic} -> {self.output_topic}')

    def _open_log(self):
        self.log_file = open(self.log_path, 'w', newline='', buffering=1)
        self.csv_writer = csv.DictWriter(self.log_file, fieldnames=[
            'ros_time_s', 'frame_id', 'header_frame_id', 'capture_time_s',
            'delivery_time_s', 'configured_delay_ms', 'measured_delay_ms',
            'aoi_ms', 'base_delay_ms', 'jitter_ms', 'jitter_distribution',
            'packet_loss_probability', 'delay_mode', 'delivered', 'status'])
        self.csv_writer.writeheader()

    @staticmethod
    def _stamp_ns(msg):
        return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)

    def _now_ns(self):
        return self.get_clock().now().nanoseconds

    def _delay_ms(self):
        mode = self.get_parameter('delay_mode').value
        base = max(0.0, float(self.get_parameter('base_delay_ms').value))
        jitter = max(0.0, float(self.get_parameter('jitter_ms').value))
        if mode == 'disabled':
            return 0.0
        if mode == 'fixed' or jitter == 0.0:
            return base
        distribution = self.get_parameter('jitter_distribution').value
        if distribution == 'gaussian':
            return max(0.0, base + random.gauss(0.0, jitter))
        return max(0.0, base + random.uniform(-jitter, jitter))

    def _config(self):
        return {
            'base_delay_ms': float(self.get_parameter('base_delay_ms').value),
            'jitter_ms': float(self.get_parameter('jitter_ms').value),
            'jitter_distribution': self.get_parameter('jitter_distribution').value,
            'packet_loss_probability': float(
                self.get_parameter('packet_loss_probability').value
                if self.get_parameter('packet_loss_enabled').value else 0.0),
            'delay_mode': self.get_parameter('delay_mode').value,
        }

    def _aoi_ms(self, now_ns):
        if self._latest_capture_ns is None:
            return None
        return max(0.0, (now_ns - self._latest_capture_ns) / 1e6)

    def _log(self, frame_id, header_frame_id, capture_ns, delivery_ns,
             configured_delay_ms, delivered, status):
        config = self._config()
        measured = (delivery_ns - capture_ns) / 1e6
        self.csv_writer.writerow({
            'ros_time_s': f'{delivery_ns / 1e9:.9f}',
            'frame_id': frame_id,
            'header_frame_id': header_frame_id,
            'capture_time_s': f'{capture_ns / 1e9:.9f}',
            'delivery_time_s': f'{delivery_ns / 1e9:.9f}',
            'configured_delay_ms': f'{configured_delay_ms:.3f}',
            'measured_delay_ms': f'{measured:.3f}',
            'aoi_ms': '' if self._aoi_ms(delivery_ns) is None else f'{self._aoi_ms(delivery_ns):.3f}',
            'base_delay_ms': f'{config["base_delay_ms"]:.3f}',
            'jitter_ms': f'{config["jitter_ms"]:.3f}',
            'jitter_distribution': config['jitter_distribution'],
            'packet_loss_probability': f'{config["packet_loss_probability"]:.3f}',
            'delay_mode': config['delay_mode'],
            'delivered': delivered,
            'status': status,
        })

    def image_callback(self, msg):
        now_ns = self._now_ns()
        capture_ns = self._stamp_ns(msg)
        timestamp_valid = capture_ns > 0
        if not timestamp_valid:
            # Keep the raw header untouched.  Internally use receipt time only
            # as a traceable fallback and report the invalid source timestamp.
            capture_ns = now_ns
            self.get_logger().warning('Received camera frame with zero timestamp')

        with self._lock:
            frame_id = self._next_frame_id
            self._next_frame_id += 1
            self._frames_seen += 1

        configured_delay_ms = self._delay_ms()
        loss_enabled = bool(self.get_parameter('packet_loss_enabled').value)
        loss_probability = float(self.get_parameter('packet_loss_probability').value)
        if loss_enabled and random.random() < max(0.0, min(100.0, loss_probability)) / 100.0:
            with self._lock:
                self._dropped += 1
                self._packet_loss_dropped += 1
            self._log(frame_id, msg.header.frame_id, capture_ns, now_ns,
                      configured_delay_ms, False, 'packet_loss')
            return

        due_ns = now_ns + int(configured_delay_ms * 1e6)
        with self._lock:
            self._order += 1
            heapq.heappush(self._queue, (
                due_ns, self._order, frame_id, capture_ns,
                timestamp_valid, msg, configured_delay_ms))

    def delivery_callback(self):
        now_ns = self._now_ns()
        while True:
            with self._lock:
                if not self._queue or self._queue[0][0] > now_ns:
                    return
                _, _, frame_id, capture_ns, timestamp_valid, msg, configured_delay_ms = heapq.heappop(self._queue)

            reject_old = bool(self.get_parameter('reject_out_of_order').value)
            if reject_old and self._latest_capture_ns is not None and capture_ns <= self._latest_capture_ns:
                with self._lock:
                    self._dropped += 1
                    self._out_of_order += 1
                self._log(frame_id, msg.header.frame_id, capture_ns, now_ns,
                          configured_delay_ms, False, 'out_of_order')
                continue

            self.publisher.publish(msg)
            self._latest_capture_ns = capture_ns
            self._last_frame_id = frame_id
            self._last_delay_ms = max(0.0, (now_ns - capture_ns) / 1e6)
            with self._lock:
                self._received += 1
            self._log(frame_id, msg.header.frame_id, capture_ns, now_ns,
                      configured_delay_ms, True,
                      'delivered' if timestamp_valid else 'delivered_invalid_timestamp')

    def status_callback(self):
        now_ns = self._now_ns()
        with self._lock:
            received = self._received
            dropped = self._dropped
            frames_seen = self._frames_seen
            packet_loss_dropped = self._packet_loss_dropped
            queue_depth = len(self._queue)
        payload = {
            'ros_time_s': now_ns / 1e9,
            'last_frame_id': self._last_frame_id,
            'aoi_ms': self._aoi_ms(now_ns),
            'last_delay_ms': self._last_delay_ms,
            'frames_received': received,
            'frames_dropped': dropped,
            'frames_seen': frames_seen,
            'packet_loss_dropped': packet_loss_dropped,
            'out_of_order': self._out_of_order,
            'loss_percent': 100.0 * packet_loss_dropped / frames_seen if frames_seen else 0.0,
            'delivery_drop_percent': 100.0 * dropped / frames_seen if frames_seen else 0.0,
            'queue_depth': queue_depth,
            **self._config(),
        }
        msg = String()
        msg.data = json.dumps(payload, separators=(',', ':'))
        self.status_publisher.publish(msg)

    def parameter_callback(self, params):
        for param in params:
            if param.name in ('base_delay_ms', 'jitter_ms', 'packet_loss_probability') and float(param.value) < 0:
                return SetParametersResult(successful=False, reason=f'{param.name} cannot be negative')
            if param.name == 'delay_mode' and param.value not in ('disabled', 'fixed', 'jitter'):
                return SetParametersResult(successful=False, reason='delay_mode must be disabled, fixed, or jitter')
            if param.name == 'jitter_distribution' and param.value not in ('uniform', 'gaussian'):
                return SetParametersResult(successful=False, reason='unsupported jitter distribution')
        return SetParametersResult(successful=True)

    def reset_statistics(self, _, response):
        with self._lock:
            self._received = 0
            self._dropped = 0
            self._frames_seen = 0
            self._packet_loss_dropped = 0
            self._out_of_order = 0
        response.success = True
        response.message = 'Statistics reset'
        return response

    def destroy_node(self):
        if hasattr(self, 'log_file'):
            self.log_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ImageImpairment()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
