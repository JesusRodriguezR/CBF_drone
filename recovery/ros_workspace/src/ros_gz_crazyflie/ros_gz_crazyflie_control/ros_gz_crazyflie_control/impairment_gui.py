"""Small independent Tk GUI for the image impairment node."""

import json
import threading
import tkinter as tk
from tkinter import ttk

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.srv import SetParameters
from std_msgs.msg import String
from std_srvs.srv import Trigger


class ImpairmentGuiNode(Node):
    def __init__(self):
        super().__init__('image_impairment_gui')
        self.status = {}
        self.lock = threading.Lock()
        self.create_subscription(
            String, '/crazyflie/camera/impairment_status', self.status_callback, 10)
        self.parameter_client = self.create_client(
            SetParameters, '/image_impairment/set_parameters')
        self.reset_client = self.create_client(Trigger, '/image_impairment/reset_statistics')

    def status_callback(self, msg):
        try:
            with self.lock:
                self.status = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warning('Invalid impairment status JSON')

    def set_values(self, values):
        if not self.parameter_client.wait_for_service(timeout_sec=0.2):
            self.get_logger().warning('Image impairment parameter service is unavailable')
            return
        request = SetParameters.Request()
        request.parameters = [
            Parameter(name, value=value).to_parameter_msg()
            for name, value in values.items()
        ]
        self.parameter_client.call_async(request)

    def reset(self):
        if self.reset_client.wait_for_service(timeout_sec=0.2):
            self.reset_client.call_async(Trigger.Request())


class Window:
    def __init__(self, node):
        self.node = node
        self.root = tk.Tk()
        self.root.title('Camera network / perception impairment')
        self.root.resizable(False, False)
        self.vars = {}
        self._build()
        self.root.after(100, self.refresh)
        self.root.protocol('WM_DELETE_WINDOW', self.close)

    def _entry(self, parent, label, key, row, default):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', padx=5, pady=3)
        value = tk.StringVar(value=default)
        self.vars[key] = value
        ttk.Entry(parent, textvariable=value, width=12).grid(row=row, column=1, padx=5, pady=3)

    def _build(self):
        delay = ttk.LabelFrame(self.root, text='Delay')
        delay.pack(fill='x', padx=8, pady=6)
        ttk.Label(delay, text='Mode').grid(row=0, column=0, sticky='w', padx=5, pady=3)
        self.vars['delay_mode'] = tk.StringVar(value='disabled')
        ttk.Combobox(delay, textvariable=self.vars['delay_mode'], state='readonly', width=15,
                     values=('disabled', 'fixed', 'jitter')).grid(row=0, column=1, padx=5, pady=3)
        self._entry(delay, 'Base delay [ms]', 'base_delay_ms', 1, '0')
        self._entry(delay, 'Jitter magnitude [ms]', 'jitter_ms', 2, '0')
        ttk.Label(delay, text='Distribution').grid(row=3, column=0, sticky='w', padx=5, pady=3)
        self.vars['jitter_distribution'] = tk.StringVar(value='uniform')
        ttk.Combobox(delay, textvariable=self.vars['jitter_distribution'], state='readonly', width=15,
                     values=('uniform', 'gaussian')).grid(row=3, column=1, padx=5, pady=3)

        loss = ttk.LabelFrame(self.root, text='Packet loss')
        loss.pack(fill='x', padx=8, pady=6)
        self.vars['packet_loss_enabled'] = tk.BooleanVar(value=False)
        ttk.Checkbutton(loss, text='Enable packet loss', variable=self.vars['packet_loss_enabled']).grid(
            row=0, column=0, columnspan=2, sticky='w', padx=5, pady=3)
        self._entry(loss, 'Probability [%]', 'packet_loss_probability', 1, '0')

        options = ttk.LabelFrame(self.root, text='Frame ordering')
        options.pack(fill='x', padx=8, pady=6)
        self.vars['reject_out_of_order'] = tk.BooleanVar(value=True)
        ttk.Checkbutton(options, text='Reject stale/out-of-order frames',
                        variable=self.vars['reject_out_of_order']).pack(anchor='w', padx=5, pady=3)

        buttons = ttk.Frame(self.root)
        buttons.pack(fill='x', padx=8, pady=4)
        ttk.Button(buttons, text='Apply / Update', command=self.apply).pack(side='left', padx=3)
        ttk.Button(buttons, text='Reset Statistics', command=self.node.reset).pack(side='left', padx=3)

        status = ttk.LabelFrame(self.root, text='Live status')
        status.pack(fill='x', padx=8, pady=6)
        self.status_label = ttk.Label(status, text='Waiting for /image_impairment')
        self.status_label.pack(anchor='w', padx=5, pady=5)

    def apply(self):
        try:
            values = {
                'delay_mode': self.vars['delay_mode'].get(),
                'base_delay_ms': float(self.vars['base_delay_ms'].get()),
                'jitter_ms': float(self.vars['jitter_ms'].get()),
                'jitter_distribution': self.vars['jitter_distribution'].get(),
                'packet_loss_enabled': bool(self.vars['packet_loss_enabled'].get()),
                'packet_loss_probability': float(self.vars['packet_loss_probability'].get()),
                'reject_out_of_order': bool(self.vars['reject_out_of_order'].get()),
            }
            self.node.set_values(values)
        except ValueError:
            self.status_label.configure(text='Invalid numeric parameter')

    def refresh(self):
        with self.node.lock:
            status = dict(self.node.status)
        if status:
            aoi = status.get('aoi_ms')
            aoi_text = 'n/a' if aoi is None else f'{aoi:.1f} ms'
            self.status_label.configure(text=(
                f"Last frame: {status.get('last_frame_id', 'n/a')}    "
                f"AoI: {aoi_text}    Delay: {status.get('last_delay_ms', 0.0):.1f} ms\n"
                f"Received: {status.get('frames_received', 0)}    "
                f"Dropped: {status.get('frames_dropped', 0)}    "
                f"Loss: {status.get('loss_percent', 0.0):.1f}%    "
                f"Out-of-order: {status.get('out_of_order', 0)}"))
        self.root.after(100, self.refresh)

    def close(self):
        self.root.destroy()
        self.node.destroy_node()
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = ImpairmentGuiNode()
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()
    Window(node).root.mainloop()
