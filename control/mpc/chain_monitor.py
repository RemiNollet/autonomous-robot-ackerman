#!/usr/bin/env python3
"""Ground-truth monitor for the full ROS2 chain (sim_server ->
bridge_node -> perception_node -> mpc_node -> bridge_node -> sim) --
M3's first full-chain integration run. Sibling to
control/mpc/closed_loop_sim.py (perfect simulator-truth state, no
perception/ROS2 in the loop): this is the opposite end, real noisy
perception through the real ROS2 graph, nothing bypassed.

Read-only observer: subscribes to /carsim/odom (true pose+velocity)
and /lane_state (perception's own reported values, for side-by-side
comparison) -- publishes nothing, participates in no control decision.
Ground truth (c0,c1,c2)/kappa computed the SAME way
closed_loop_sim.py's perfect-state run did (compute_lane_state,
windowed_curvature_average, ADR-18) from the TRUE odometry pose, not
from /lane_state's own self-reported values, so perception's own error
is visible here rather than perception grading itself.

Logs at /lane_state's own rate, matching mpc_node's control cadence
and closed_loop_sim.py's logging rate, for a directly comparable .npz.
Solve status/solve time and health-fallback-trigger counts are not
observable from outside mpc_node without modifying it -- this script
does neither; those numbers come from grepping mpc_node's own log
after the run, reported separately from this script's .npz.

Run on the VM, full chain already up (sim_server on the Mac,
bridge_node/perception_node/mpc_node on the VM):
    source /opt/ros/kilted/setup.bash
    source ~/ros2_ws/install/setup.bash
    source ~/venvs/ackerman_ros2/bin/activate
    python3 control/mpc/chain_monitor.py --laps 3 --out-dir /tmp \
        --tag chain_run

Also writes a `{tag}_checkpoint_log.npz` snapshot every
checkpoint_interval_s (default 120s), overwritten each time, so a
sustained run's trend (or a reason to stop early) can be read without
waiting for the run's own final save.
"""
import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import QoSPresetProfiles  # noqa: E402

from carsim_bridge.clock_utils import AgeCalibrator  # noqa: E402
from carsim_msgs.msg import LaneState  # noqa: E402
from perception.dataset.geometry import compute_lane_state  # noqa: E402
from perception.dataset.track_definitions import (  # noqa: E402
    REFERENCE_TRACK,
)
from perception.dataset.windowed_relabel import (  # noqa: E402
    windowed_curvature_average,
)


def yaw_from_quat(q):
    return math.atan2(
        2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class ChainMonitor(Node):

    def __init__(self, laps, out_dir, tag, checkpoint_interval_s=120.0):
        super().__init__('chain_monitor')
        self.laps_target = laps
        self.out_dir = out_dir
        self.tag = tag

        self.create_subscription(
            Odometry, 'carsim/odom', self.on_odom,
            QoSPresetProfiles.SENSOR_DATA.value)
        self.create_subscription(
            LaneState, 'lane_state', self.on_lane_state, 10)
        if checkpoint_interval_s > 0:
            self.create_timer(checkpoint_interval_s, self._checkpoint)

        self.have_odom = False
        self.x = self.y = self.yaw = self.v = 0.0
        self._age_calibrator = AgeCalibrator()

        self.log = {k: [] for k in [
            "t", "x", "y", "yaw", "s", "v",
            "lateral_error", "heading_error", "kappa",
            "perception_lateral_error", "perception_heading_error",
            "perception_kappa", "perception_confidence", "perception_valid",
            "lane_state_age_s",
        ]}
        self.s_prev = None
        self.total_unwrapped_s = 0.0
        self.t0_wall = time.perf_counter()
        self.n_lane_state_msgs = 0
        self.done = False

    def on_odom(self, msg: Odometry):
        self.x = float(msg.pose.pose.position.x)
        self.y = float(msg.pose.pose.position.y)
        self.yaw = yaw_from_quat(msg.pose.pose.orientation)
        self.v = float(msg.twist.twist.linear.x)
        self.have_odom = True

    def on_lane_state(self, msg: LaneState):
        if not self.have_odom or self.done:
            return

        lane_state = compute_lane_state(
            REFERENCE_TRACK, self.x, self.y, self.yaw)
        kappa_true = windowed_curvature_average(
            REFERENCE_TRACK, self.x, self.y)

        now_ns = self.get_clock().now().nanoseconds
        age_s = self._age_calibrator.age_s(now_ns, msg.header.stamp)

        s = lane_state.s
        if self.s_prev is not None:
            ds = s - self.s_prev
            total_length = REFERENCE_TRACK.total_length
            if ds < -total_length * 0.5:
                ds += total_length
            elif ds > total_length * 0.5:
                ds -= total_length
            self.total_unwrapped_s += ds
        self.s_prev = s

        t = time.perf_counter() - self.t0_wall
        self.log["t"].append(t)
        self.log["x"].append(self.x)
        self.log["y"].append(self.y)
        self.log["yaw"].append(self.yaw)
        self.log["s"].append(s)
        self.log["v"].append(self.v)
        self.log["lateral_error"].append(lane_state.lateral_error)
        self.log["heading_error"].append(lane_state.heading_error)
        self.log["kappa"].append(kappa_true)
        self.log["perception_lateral_error"].append(float(msg.lateral_error))
        self.log["perception_heading_error"].append(
            float(msg.heading_error))
        self.log["perception_kappa"].append(float(msg.curvature))
        self.log["perception_confidence"].append(float(msg.confidence))
        self.log["perception_valid"].append(bool(msg.valid))
        self.log["lane_state_age_s"].append(age_s)

        self.n_lane_state_msgs += 1
        if self.n_lane_state_msgs % 100 == 0:
            laps_done = self.total_unwrapped_s / REFERENCE_TRACK.total_length
            self.get_logger().info(
                f'{self.n_lane_state_msgs} /lane_state msgs, '
                f'{laps_done:.2f} laps, t={t:.1f}s')

        target_s = self.laps_target * REFERENCE_TRACK.total_length
        if self.total_unwrapped_s >= target_s:
            self.done = True
            self._save()

    def _save(self):
        for k in self.log:
            self.log[k] = np.array(self.log[k])
        out_path = os.path.join(self.out_dir, f'{self.tag}_log.npz')
        np.savez(out_path, **self.log)
        laps_done = self.total_unwrapped_s / REFERENCE_TRACK.total_length
        self.get_logger().info(
            f'DONE: {self.n_lane_state_msgs} msgs, {laps_done:.2f} laps, '
            f'saved to {out_path}')

    def _checkpoint(self):
        """Periodic snapshot for a run long enough that waiting for
        _save() isn't practical -- builds arrays from copies of the
        current lists rather than converting self.log in place, so
        on_lane_state's own appends (and the eventual real _save())
        are unaffected by this running concurrently on the executor."""
        if self.done or self.n_lane_state_msgs == 0:
            return
        snapshot = {k: np.array(v) for k, v in self.log.items()}
        out_path = os.path.join(
            self.out_dir, f'{self.tag}_checkpoint_log.npz')
        np.savez(out_path, **snapshot)
        laps_done = self.total_unwrapped_s / REFERENCE_TRACK.total_length
        self.get_logger().info(
            f'checkpoint: {self.n_lane_state_msgs} msgs, '
            f'{laps_done:.2f} laps, saved to {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--laps', type=float, default=3.0)
    ap.add_argument('--out-dir', default='/tmp')
    ap.add_argument('--tag', default='chain_run')
    ap.add_argument(
        '--checkpoint-interval', type=float, default=120.0,
        help='seconds between checkpoint saves, 0 to disable')
    args = ap.parse_args()

    rclpy.init()
    node = ChainMonitor(
        args.laps, args.out_dir, args.tag, args.checkpoint_interval)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=1.0)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
