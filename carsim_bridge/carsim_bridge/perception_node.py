#!/usr/bin/env python3
"""CNN camera -> /lane_state.

Cote VM. Le SEUL noeud qui connait le modele de perception : pour tout le
reste du graphe, /lane_state est un contrat (docs/lane-state-contract.md),
pas une implementation -- le jour ou l'INT8/ONNX ou une pipeline CV
classique (ADR-9) remplace ce modele, on remplace ce noeud, pas les autres.

    /carsim/image_raw  sensor_msgs/Image      (in,  ~30 Hz, rgb8)
    /lane_state        carsim_msgs/LaneState  (out)

Thin rclpy wrapper: all the model/preprocessing logic lives in
perception_inference.py, which has no rclpy dependency and is what's unit
tested directly. This file only exists to plug that logic into the ROS2
graph -- see tests/test_perception_node.py for why that split matters on a
project where ROS2 only runs in the VM (ADR-1).
"""
import os
import sys
import time

import rclpy
import torch
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from rclpy.time import Time as RclpyTime
from sensor_msgs.msg import Image

# One level up, not two: this only needs to resolve its own sibling
# module (carsim_bridge.perception_inference below), and this file's
# parent directory IS the outer carsim_bridge/ package root regardless of
# nesting depth -- unlike perception_inference.py, which reaches all the
# way to the repo root for `perception.*` and does need "../..".
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from carsim_msgs.msg import LaneState  # noqa: E402
from carsim_bridge.perception_inference import (  # noqa: E402
    DEFAULT_CHECKPOINT, distribution_stats, load_model, ros_image_to_pil,
    run_inference,
)


class PerceptionNode(Node):

    def __init__(self, **kwargs):
        # **kwargs forwarded to rclpy.node.Node -- lets tests construct this
        # node with parameter_overrides (e.g. a temp checkpoint_path) without
        # a live graph or a real trained checkpoint on disk.
        super().__init__('perception_node', **kwargs)

        self.declare_parameter('checkpoint_path', DEFAULT_CHECKPOINT)
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('device', 'cpu')
        # Frame count, not a time window: at whatever rate is actually
        # achieved (which is the thing being measured), a fixed time window
        # would silently under-sample if the rate is worse than expected.
        self.declare_parameter('stats_window_frames', 500)
        # '' disables the file dump -- get_logger().info() below always
        # fires regardless, so disabling this only drops the copy-pasteable
        # markdown snippet, not the measurement itself.
        self.declare_parameter('stats_output_path', '/tmp/perception_node_vm_stats.md')

        checkpoint_path = self.get_parameter('checkpoint_path').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.device = torch.device(self.get_parameter('device').value)
        self.model = load_model(checkpoint_path, self.device)
        self.stats_window_frames = self.get_parameter('stats_window_frames').value
        self.stats_output_path = self.get_parameter('stats_output_path').value

        self.pub = self.create_publisher(LaneState, 'lane_state', 10)
        self.create_subscription(Image, 'carsim/image_raw', self.on_image,
                                  QoSPresetProfiles.SENSOR_DATA.value)

        self.n_msgs = 0
        self.t_preprocess_sum = 0.0
        self.t_forward_sum = 0.0
        self.t_publish_sum = 0.0
        self.create_timer(2.0, self.report)

        # Raw sample buffers for the fuller distribution report (task:
        # "M2 closure -- perception_node inference frequency"). Kept
        # separate from the *_sum accumulators above, which only ever need
        # a mean for the lightweight 2 s heartbeat.
        self._t_pre_samples = []
        self._t_fwd_samples = []
        self._t_pub_samples = []
        self._interval_samples = []
        self._age_samples = []
        self._last_publish_perf = None

        self.get_logger().info(
            f'perception_node actif  checkpoint={checkpoint_path}  device={self.device}')

    def on_image(self, msg: Image):
        pil_img = ros_image_to_pil(msg.width, msg.height, msg.encoding, msg.data)
        e_y, e_psi, confidence, t_pre, t_fwd = run_inference(self.model, pil_img, self.device)

        t2 = time.perf_counter()
        out = LaneState()
        # Propagated render time, NOT re-stamped here -- the M1 contract's
        # timestamp semantics (docs/lane-state-contract.md section 3)
        # require the age computation downstream to be exact, which only
        # holds if every hop copies this forward instead of restamping.
        out.header.stamp = msg.header.stamp
        out.header.frame_id = msg.header.frame_id
        out.lateral_error = float(e_y)
        out.heading_error = float(e_psi)
        # ADR-12: kappa's loss weight is 0 -- the head is kept in the
        # architecture but was never trained, so its raw output is
        # initialization drift that could vary unpredictably between
        # checkpoints. Publishing it would look like a real curvature signal
        # to a downstream MPC feedforward term; publish the one value that
        # cannot be mistaken for one.
        out.curvature = 0.0
        out.confidence = float(confidence)
        out.valid = confidence >= self.confidence_threshold

        self.pub.publish(out)
        t3 = time.perf_counter()

        self.n_msgs += 1
        self.t_preprocess_sum += t_pre
        self.t_forward_sum += t_fwd
        self.t_publish_sum += (t3 - t2)

        # Wall-clock interval between consecutive publications, on a single
        # monotonic clock (perf_counter) -- the actual achievable rate, not
        # the sum of the three stage timings above (which ignores whatever
        # gap the ROS2 executor/subscription queue adds between callbacks).
        if self._last_publish_perf is not None:
            self._interval_samples.append(t3 - self._last_publish_perf)
        self._last_publish_perf = t3

        # End-to-end age at publication: this node's publish time minus
        # msg.header.stamp, both read from the VM's own ROS clock. NOT a
        # Mac/VM cross-clock measurement and does NOT span the ZeroMQ hop --
        # bridge_node.py stamps every image with self.get_clock().now() at
        # the moment IT receives the frame (carsim_bridge/bridge_node.py
        # poll()), not with the Mac's render time. The Mac->VM hop itself is
        # already measured, clock-skew-corrected (ADR-4's round-trip-sum
        # method), and published separately on /carsim/latency_ms -- add
        # that to this number for the full Mac-render-to-lane_state age.
        # .nanoseconds (plain int), not Time.__sub__ -- get_clock().now() and
        # Time.from_msg() don't default to the same rclpy clock_type, and
        # subtracting two Time objects directly raises unless they match.
        now_ns = self.get_clock().now().nanoseconds
        stamp_ns = RclpyTime.from_msg(msg.header.stamp).nanoseconds
        self._age_samples.append((now_ns - stamp_ns) * 1e-9)

        self._t_pre_samples.append(t_pre)
        self._t_fwd_samples.append(t_fwd)
        self._t_pub_samples.append(t3 - t2)

        if len(self._t_pre_samples) >= self.stats_window_frames:
            self._report_distribution()

    def report(self):
        if self.n_msgs == 0:
            return
        n = self.n_msgs
        total = self.t_preprocess_sum + self.t_forward_sum + self.t_publish_sum
        self.get_logger().info(
            f'{n} msgs | preprocess {self.t_preprocess_sum/n*1000:.2f} ms | '
            f'forward {self.t_forward_sum/n*1000:.2f} ms | '
            f'publish {self.t_publish_sum/n*1000:.2f} ms | '
            f'total {total/n*1000:.2f} ms')
        self.n_msgs = 0
        self.t_preprocess_sum = self.t_forward_sum = self.t_publish_sum = 0.0

    def _report_distribution(self):
        """Full mean/p50/p95/max/std report over stats_window_frames frames
        -- the lightweight report() above only ever tracks a mean, which
        hides jitter. Logged always; also written to stats_output_path
        (markdown, results.md-ready) when that parameter is non-empty."""
        pre = distribution_stats(self._t_pre_samples)
        fwd = distribution_stats(self._t_fwd_samples)
        pub = distribution_stats(self._t_pub_samples)
        interval = distribution_stats(self._interval_samples)
        age = distribution_stats(self._age_samples)

        def row(name, d):
            return (f'| {name} | {d["mean"]:.3f} | {d["p50"]:.3f} | '
                    f'{d["p95"]:.3f} | {d["max"]:.3f} | {d["std"]:.3f} | {d["n"]} |')

        lines = [
            f'# perception_node VM stats (n={pre["n"]} frames)',
            '',
            '| Stage | mean (ms) | p50 | p95 | max | std | n |',
            '|---|---|---|---|---|---|---|',
            row('preprocess', pre),
            row('forward', fwd),
            row('publish (msg build)', pub),
        ]
        if interval is not None:
            rate_hz = 1000.0 / interval['mean'] if interval['mean'] > 0 else float('nan')
            lines.append(row('publish interval', interval))
            lines.append('')
            lines.append(
                f'Achieved rate ~{rate_hz:.1f} Hz (mean interval, single VM clock '
                f'via perf_counter -- the actual achievable rate, not derived '
                f'from the stage timings above).')
        lines.append('')
        lines.append(row('age at publish (header.stamp -> publish)', age))
        lines.append('')
        lines.append(
            'age at publish is single-clock (VM ROS clock on both ends -- '
            'bridge_node stamps images at ITS OWN receipt time, not the '
            'Mac render time) and covers graph-internal latency only: it '
            'does NOT include the Mac->VM ZeroMQ hop. Add /carsim/latency_ms '
            '(bridge_node.py report(), already clock-skew-corrected per '
            'ADR-4) for the full Mac-render-to-lane_state age.')

        report_text = '\n'.join(lines)
        for line in lines:
            if line:
                self.get_logger().info(line)

        if self.stats_output_path:
            with open(self.stats_output_path, 'w') as f:
                f.write(report_text + '\n')

        self._t_pre_samples.clear()
        self._t_fwd_samples.clear()
        self._t_pub_samples.clear()
        self._interval_samples.clear()
        self._age_samples.clear()


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
