"""Simple real-time top-down monitor for the Gazebo FSM trial."""

import tkinter as tk
import json
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

import rclpy
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String


class Monitor(Node):
    def __init__(self):
        super().__init__('greenhouse_fsm_visualizer')
        self.actual = []
        self.target = (0.0, 0.0)
        self.state = 'WAITING'
        self.error_y = 0.0
        self.monitoring_points = []
        config = Path(get_package_share_directory('ros_gz_crazyflie_control')) / 'config' / 'monitoring_points.json'
        try:
            self.monitoring_points = json.loads(config.read_text())
        except (OSError, json.JSONDecodeError):
            pass
        self.active_point = ''
        self.observation_active = False
        self.crop_side = ''
        self.command = Twist()
        self.create_subscription(Odometry, '/crazyflie/odom', self.odom_cb, 20)
        self.create_subscription(PointStamped, '/greenhouse/fsm_target', self.target_cb, 20)
        self.create_subscription(String, '/greenhouse/fsm_state', self.state_cb, 20)
        self.create_subscription(Twist, '/cmd_vel', self.cmd_cb, 20)

    def odom_cb(self, msg):
        p = msg.pose.pose.position
        self.actual.append((p.x, p.y))
        self.actual = self.actual[-4000:]

    def target_cb(self, msg):
        self.target = (msg.point.x, msg.point.y)

    def state_cb(self, msg):
        fields = msg.data.split('|')
        self.state = fields[0]
        values = {f.split('=', 1)[0]: f.split('=', 1)[1]
                  for f in fields[1:] if '=' in f}
        self.active_point = values.get('monitoring_point_id', '')
        self.observation_active = values.get('observation_active', '0') == '1'
        self.crop_side = values.get('crop_side', '')
        for field in fields[1:]:
            if field.startswith('error_y='):
                self.error_y = float(field.split('=', 1)[1])

    def cmd_cb(self, msg):
        self.command = msg


class Window:
    def __init__(self, node):
        self.node = node
        self.root = tk.Tk()
        self.root.title('Greenhouse FSM — intended vs actual trajectory')
        self.canvas = tk.Canvas(self.root, width=900, height=720, bg='#f7f7f7')
        self.canvas.pack()
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.draw()

    def xy(self, x, y):
        return 50 + (x + 20.0) * 800.0 / 40.0, 650 - (y + 15.0) * 600.0 / 30.0

    def draw(self):
        try:
            rclpy.spin_once(self.node, timeout_sec=0.0)
            self.canvas.delete('all')
            # Crop rows and world boundary.
            self.canvas.create_rectangle(50, 50, 850, 650, outline='#777')
            for i in range(5):
                y = -3.0 + 1.5 * i
                x1, yy = self.xy(-17, y)
                x2, _ = self.xy(17, y)
                self.canvas.create_line(x1, yy, x2, yy, fill='#b9b9b9', width=8)

            for point in self.node.monitoring_points:
                px, py = self.xy(float(point['longitudinal_position']),
                                 -2.25 + 1.5 * (int(point['aisle']) - 1))
                active = point.get('monitoring_point_id') == self.node.active_point
                self.canvas.create_oval(px - (8 if active else 5), py - (8 if active else 5),
                                        px + (8 if active else 5), py + (8 if active else 5),
                                        fill='#ff7f0e' if active else '#9b59b6',
                                        outline='#111', width=2 if active else 1)

            # Intended boustrophedon path in blue.
            planned = []
            for i in range(4):
                y = -2.25 + 1.5 * i
                if i % 2 == 0:
                    planned += [(-18, y), (18, y)]
                else:
                    planned += [(18, y), (-18, y)]
            for a, b in zip(planned, planned[1:]):
                self.canvas.create_line(*self.xy(*a), *self.xy(*b), fill='#145cff', width=2)

            # Actual odometry trace in red.
            if len(self.node.actual) > 1:
                coords = []
                for p in self.node.actual:
                    coords.extend(self.xy(*p))
                self.canvas.create_line(*coords, fill='#e32626', width=3)

            tx, ty = self.xy(*self.node.target)
            self.canvas.create_oval(tx - 7, ty - 7, tx + 7, ty + 7,
                                    fill='#ffd21f', outline='#8a7000', width=2)
            ax, ay = self.xy(*(self.node.actual[-1] if self.node.actual else (0, 0)))
            self.canvas.create_oval(ax - 5, ay - 5, ax + 5, ay + 5,
                                    fill='#e32626', outline='white')

            drift = abs(self.node.error_y)
            status = 'OK — FSM sees lane centered' if drift < 0.20 else 'DRIFT — FSM sees lateral error'
            self.canvas.create_text(60, 18, anchor='w', text=
                f'State: {self.node.state}    {status}    lane error: {self.node.error_y:+.2f} m'
                f'    OBSERVE={self.node.observation_active} side={self.node.crop_side or "-"}',
                fill='#111', font=('TkDefaultFont', 12, 'bold'))
            self.canvas.create_text(60, 685, anchor='w', text=
                f'Blue: intended path   Red: odometry   Yellow: FSM target   Purple/Orange: monitoring points   '
                f'cmd vx={self.node.command.linear.x:+.2f}  vy={self.node.command.linear.y:+.2f}  vz={self.node.command.linear.z:+.2f}',
                fill='#222')
            self.root.after(100, self.draw)
        except tk.TclError:
            pass

    def close(self):
        self.root.destroy()
        self.node.destroy_node()
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = Monitor()
    Window(node).root.mainloop()


if __name__ == '__main__':
    main()
