#!/usr/bin/env python3
"""CNN camera -> /lane_state.

Runs on the VM. The ONLY node that knows about the perception model --
for the rest of the graph, /lane_state is a contract
(docs/lane-state-contract.md), not an implementation: the day INT8/ONNX
or a classic CV pipeline (ADR-9) replaces this model, this node gets
replaced, not the others.

    /carsim/image_raw  sensor_msgs/Image      (in,  ~30 Hz, rgb8)
    /lane_state        carsim_msgs/LaneState  (out)

Thin rclpy wrapper: all the model/preprocessing logic lives in
perception_inference.py, which has no rclpy dependency and is what's
unit tested directly. This file only exists to plug that logic into
the ROS2 graph -- see tests/test_perception_node.py for why that split
matters on a project where ROS2 only runs in the VM (ADR-1).
"""
import os
import sys
import threading
import time

import rclpy
import torch
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import Image

# One level up, not two: resolves this file's own sibling module
# (carsim_bridge.perception_inference), unlike perception_inference.py's
# own "../.." reach to the repo root for `perception.*`.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from carsim_msgs.msg import LaneState  # noqa: E402
from carsim_bridge.clock_utils import AgeCalibrator  # noqa: E402
from carsim_bridge.perception_inference import (  # noqa: E402
    DEFAULT_CHECKPOINT, distribution_stats, load_model, ros_image_to_pil,
    run_inference,
)


class PerceptionNode(Node):

    def __init__(self, **kwargs):
        # **kwargs forwarded to Node -- lets tests construct this node
        # with parameter_overrides, without a live graph.
        super().__init__('perception_node', **kwargs)

        self.declare_parameter('checkpoint_path', DEFAULT_CHECKPOINT)
        # 0.5, reviewed but left unjustified (ADR-23's Consequence).
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('device', 'cpu')
        # Frame count, not a time window: stays correctly sized even if
        # the achieved rate differs from the nominal one.
        self.declare_parameter('stats_window_frames', 500)
        # '' disables only the markdown file dump, not the log report.
        self.declare_parameter(
            'stats_output_path', '/tmp/perception_node_vm_stats.md')

        checkpoint_path = self.get_parameter('checkpoint_path').value
        threshold_param = self.get_parameter('confidence_threshold')
        self.confidence_threshold = threshold_param.value
        self.device = torch.device(self.get_parameter('device').value)
        self.model = load_model(checkpoint_path, self.device)
        window_param = self.get_parameter('stats_window_frames')
        self.stats_window_frames = window_param.value
        self.stats_output_path = self.get_parameter(
            'stats_output_path').value

        self.pub = self.create_publisher(LaneState, 'lane_state', 10)
        self.create_subscription(
            Image, 'carsim/image_raw', self.on_image,
            QoSPresetProfiles.SENSOR_DATA.value)

        self.n_msgs = 0
        self.t_preprocess_sum = 0.0
        self.t_forward_sum = 0.0
        self.t_publish_sum = 0.0
        self.create_timer(2.0, self.report)

        # Full-distribution sample buffers, separate from the *_sum
        # accumulators above (which only need a mean for the 2s report).
        self._t_pre_samples = []
        self._t_fwd_samples = []
        self._t_pub_samples = []
        self._interval_samples = []
        self._age_samples = []
        self._last_publish_perf = None
        # Set by _spawn_distribution_report(); tests join() it.
        self._report_thread = None
        self._age_calibrator = AgeCalibrator()  # ADR-25

        self.get_logger().info(
            f'perception_node active  checkpoint={checkpoint_path}  '
            f'device={self.device}')

    def on_image(self, msg: Image):
        pil_img = ros_image_to_pil(
            msg.width, msg.height, msg.encoding, msg.data)
        e_y, e_psi, kappa, confidence, t_pre, t_fwd = run_inference(
            self.model, pil_img, self.device)

        t2 = time.perf_counter()
        out = LaneState()
        # Propagated render time, not re-stamped: the age computation
        # downstream needs every hop to forward the original stamp
        # (lane-state-contract.md section 3).
        out.header.stamp = msg.header.stamp
        out.header.frame_id = msg.header.frame_id
        out.lateral_error = float(e_y)
        out.heading_error = float(e_psi)
        out.curvature = float(kappa)  # real signal since ADR-18; ADR-23
        out.confidence = float(confidence)
        out.valid = confidence >= self.confidence_threshold

        self.pub.publish(out)
        t3 = time.perf_counter()

        self.n_msgs += 1
        self.t_preprocess_sum += t_pre
        self.t_forward_sum += t_fwd
        self.t_publish_sum += (t3 - t2)

        # Wall-clock interval between publications, not the sum of the
        # three stage timings (which ignores executor/queue gaps).
        if self._last_publish_perf is not None:
            self._interval_samples.append(t3 - self._last_publish_perf)
        self._last_publish_perf = t3

        # Graph-internal age only (ADR-25 AgeCalibrator corrects for
        # header.stamp being sim-clock, ADR-13); doesn't cover the
        # Mac->VM hop, which /carsim/latency_ms measures separately.
        now_ns = self.get_clock().now().nanoseconds
        self._age_samples.append(
            self._age_calibrator.age_s(now_ns, msg.header.stamp))

        self._t_pre_samples.append(t_pre)
        self._t_fwd_samples.append(t_fwd)
        self._t_pub_samples.append(t3 - t2)

        if len(self._t_pre_samples) >= self.stats_window_frames:
            self._spawn_distribution_report()

    def report(self):
        if self.n_msgs == 0:
            return
        n = self.n_msgs
        total = (
            self.t_preprocess_sum + self.t_forward_sum + self.t_publish_sum)
        self.get_logger().info(
            f'{n} msgs | preprocess {self.t_preprocess_sum/n*1000:.2f} ms | '
            f'forward {self.t_forward_sum/n*1000:.2f} ms | '
            f'publish {self.t_publish_sum/n*1000:.2f} ms | '
            f'total {total/n*1000:.2f} ms')
        self.n_msgs = 0
        self.t_preprocess_sum = self.t_forward_sum = self.t_publish_sum = 0.0

    def _spawn_distribution_report(self):
        """Hand the just-filled sample window off to a background thread
        and reset the buffers for the next window -- both O(1) reference
        reassignments, so on_image() itself never waits on the report.

        _report_distribution() below measured ~3 ms in isolation, but
        running inline here was directly implicated in a 198 ms
        forward-pass stall under real graph load during M3's full-chain
        integration test -- CPU contention an isolated timing never
        reproduces. The fix is structural: nothing this method does
        needs on_image()'s thread, so it no longer runs there.

        Swapping BEFORE spawning is what makes this safe without a lock:
        the thread only ever sees the lists captured here, and on_image()
        keeps appending to fresh ones -- no mutable state is shared."""
        samples = {
            "pre": self._t_pre_samples,
            "fwd": self._t_fwd_samples,
            "pub": self._t_pub_samples,
            "interval": self._interval_samples,
            "age": self._age_samples,
        }
        self._t_pre_samples = []
        self._t_fwd_samples = []
        self._t_pub_samples = []
        self._interval_samples = []
        self._age_samples = []
        self._report_thread = threading.Thread(
            target=self._report_distribution, args=(samples,), daemon=True)
        self._report_thread.start()

    def _report_distribution(self, samples):
        """Full mean/p50/p95/max/std report over stats_window_frames
        frames -- report() above only ever tracks a mean, which hides
        jitter. Logged always; also written to stats_output_path
        (markdown) when that parameter is non-empty.

        Runs on the background thread spawned by
        _spawn_distribution_report() above, operating only on the
        `samples` snapshot it was handed -- does not touch
        self._t_pre_samples etc, which on_image() may already be
        appending to on the main thread by the time this runs."""
        pre = distribution_stats(samples["pre"])
        fwd = distribution_stats(samples["fwd"])
        pub = distribution_stats(samples["pub"])
        interval = distribution_stats(samples["interval"])
        age = distribution_stats(samples["age"])

        def row(name, d):
            return (
                f'| {name} | {d["mean"]:.3f} | {d["p50"]:.3f} | '
                f'{d["p95"]:.3f} | {d["max"]:.3f} | {d["std"]:.3f} | '
                f'{d["n"]} |')

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
            rate_hz = (
                1000.0 / interval['mean'] if interval['mean'] > 0
                else float('nan'))
            lines.append(row('publish interval', interval))
            lines.append('')
            lines.append(
                f'Achieved rate ~{rate_hz:.1f} Hz (mean interval, single '
                f'VM clock via perf_counter).')
        lines.append('')
        lines.append(row('age at publish (header.stamp -> publish)', age))
        lines.append('')
        lines.append(
            'age at publish is clock-offset-calibrated (AgeCalibrator, '
            'clock_utils.py, ADR-25). Graph-internal latency only: add '
            '/carsim/latency_ms (bridge_node.py, ADR-4) for the full '
            'Mac-render-to-lane_state age.')

        report_text = '\n'.join(lines)
        for line in lines:
            if line:
                self.get_logger().info(line)

        if self.stats_output_path:
            with open(self.stats_output_path, 'w') as f:
                f.write(report_text + '\n')


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
