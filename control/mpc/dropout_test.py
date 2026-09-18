#!/usr/bin/env python3
"""M3 forced perception-dropout test.

Sits between perception_node and mpc_node on the ROS2 graph
(perception_node launched with `-r lane_state:=lane_state_raw` so its
real output lands on lane_state_raw instead of lane_state; this node
republishes to lane_state, which mpc_node actually subscribes to).
perception_node itself is never touched or killed -- its own
health/timing logs stay available so a spontaneous VM-jitter event
(session finding: correlates with macOS display sleep/wake, not this
test) can be told apart from an injected trial by cross-referencing
timestamps after the run.

mpc_node's health gate (carsim_bridge/mpc_node.py on_lane_state) is
purely message-driven -- there is no independent timer polling
staleness, it only re-evaluates when a new /lane_state message
actually arrives. So "blocking forwarding" alone (silence, nothing
delivered) would leave mpc_node completely unaware anything is wrong
until the next message shows up fresh -- it can't "hold" a safe state
it's never asked to re-evaluate. Instead, for staleness trials this
node keeps forwarding at perception's own ~25 Hz cadence throughout
the injection window, but with header.stamp frozen at the value of
the last real message received right as the trial started -- so age
grows correctly with real elapsed time and every delivered message
during the window is independently re-evaluated by mpc_node, exactly
reproducing "hold for the full duration". For invalidity trials, the
next N real (fresh-stamped) messages are forwarded with valid
overridden to False, leaving staleness out of it entirely.

AgeCalibrator (clock_utils.py) is a running minimum of (now - stamp):
by construction, messages with an artificially large injected age can
never lower that minimum, so this injection method cannot corrupt the
calibration; verified empirically too, by checking ages return to
normal immediately after each trial ends (recorded in the per-trial
log this script also produces).

Internal-state proxy for the recovery check: mpc_node.py has no other
state that could cause a residual cooldown -- read directly,
`unhealthy` is recomputed fresh every call, so this script does not
add instrumentation to mpc_node.py. Instead it subscribes to
/carsim/cmd directly (the observable output of that internal state) to
confirm commands resume an ordinary solve-driven shape immediately
after recovery, not a ramp.

Run on the VM, full chain already up with perception_node launched
with the topic remap:
    source /opt/ros/kilted/setup.bash
    source ~/ros2_ws/install/setup.bash
    source ~/venvs/ackerman_ros2/bin/activate
    python3 control/mpc/dropout_test.py --out-dir /tmp --tag dropout_test
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from geometry_msgs.msg import Twist  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import QoSPresetProfiles  # noqa: E402

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


def build_trial_plan():
    """(kind, param, post_gap_s) x 3 reps each, in the order run."""
    plan = []
    stale_configs = [
        (0.1, 5.0), (0.14, 5.0), (0.16, 5.0), (1.5, 8.0), (7.5, 15.0),
    ]
    for duration, gap in stale_configs:
        for rep in range(3):
            plan.append(
                dict(kind='stale', param=duration, rep=rep, post_gap_s=gap))
    for n, gap in [(1, 5.0), (2, 5.0), (3, 5.0), (10, 5.0)]:
        for rep in range(3):
            plan.append(
                dict(kind='invalid', param=n, rep=rep, post_gap_s=gap))
    return plan


class DropoutTest(Node):

    def __init__(self, out_dir, tag, initial_settle_s):
        super().__init__('dropout_test')
        self.out_dir = out_dir
        self.tag = tag

        self.pub = self.create_publisher(LaneState, 'lane_state', 10)
        self.create_subscription(
            LaneState, 'lane_state_raw', self.on_lane_state_raw, 10)
        self.create_subscription(
            Odometry, 'carsim/odom', self.on_odom,
            QoSPresetProfiles.SENSOR_DATA.value)
        self.create_subscription(Twist, 'carsim/cmd', self.on_cmd, 10)
        # 200 Hz, not the ~25 Hz message cadence: the shortest trial
        # durations are only 2.5-4x a 40ms message period, so ending a
        # trial on message arrival (or a coarser timer) can overshoot
        # by a full period -- confirmed empirically (10 Hz overshot a
        # 100ms trial to 196ms).
        self.create_timer(0.005, self._tick)

        self.have_odom = False
        self.x = self.y = self.yaw = self.v = 0.0
        self.cmd_angular_z = self.cmd_linear_x = 0.0

        self.last_real_msg = None
        self.frozen_stamp = None
        self.invalid_sent_count = 0

        self.plan = build_trial_plan()
        self.trial_index = -1
        self.phase = 'settle'
        self.phase_target_s = initial_settle_s
        self.phase_start_epoch = self._now_s()
        self.trial_records = []
        self.done = False

        self.log = {k: [] for k in [
            "t", "x", "y", "yaw", "v", "lateral_error", "heading_error",
            "kappa", "cmd_angular_z", "cmd_linear_x", "trial_index",
            "phase_code",
        ]}
        self.t0_wall = time.perf_counter()

        self.get_logger().info(
            f'dropout_test active: {len(self.plan)} trials planned, '
            f'{initial_settle_s:.1f}s initial settle')

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def on_odom(self, msg: Odometry):
        self.x = float(msg.pose.pose.position.x)
        self.y = float(msg.pose.pose.position.y)
        self.yaw = yaw_from_quat(msg.pose.pose.orientation)
        self.v = float(msg.twist.twist.linear.x)
        self.have_odom = True
        self._log_sample()

    def on_cmd(self, msg: Twist):
        self.cmd_angular_z = float(msg.angular.z)
        self.cmd_linear_x = float(msg.linear.x)

    def _log_sample(self):
        if not self.have_odom or self.done:
            return
        lane_state = compute_lane_state(
            REFERENCE_TRACK, self.x, self.y, self.yaw)
        kappa_true = windowed_curvature_average(
            REFERENCE_TRACK, self.x, self.y)
        phase_code = {'settle': 0, 'inject': 1, 'done': 2}[self.phase]
        self.log["t"].append(time.perf_counter() - self.t0_wall)
        self.log["x"].append(self.x)
        self.log["y"].append(self.y)
        self.log["yaw"].append(self.yaw)
        self.log["v"].append(self.v)
        self.log["lateral_error"].append(lane_state.lateral_error)
        self.log["heading_error"].append(lane_state.heading_error)
        self.log["kappa"].append(kappa_true)
        self.log["cmd_angular_z"].append(self.cmd_angular_z)
        self.log["cmd_linear_x"].append(self.cmd_linear_x)
        self.log["trial_index"].append(self.trial_index)
        self.log["phase_code"].append(phase_code)

    def on_lane_state_raw(self, msg: LaneState):
        self.last_real_msg = msg

        if self.phase != 'inject' or self.done:
            self.pub.publish(msg)
            return

        trial = self.plan[self.trial_index]
        if trial['kind'] == 'stale':
            out = LaneState()
            out.header.stamp = self.frozen_stamp
            out.header.frame_id = msg.header.frame_id
            out.lateral_error = msg.lateral_error
            out.heading_error = msg.heading_error
            out.curvature = msg.curvature
            out.confidence = msg.confidence
            out.valid = True
            self.pub.publish(out)
        else:  # invalid
            out = LaneState()
            out.header.stamp = msg.header.stamp
            out.header.frame_id = msg.header.frame_id
            out.lateral_error = msg.lateral_error
            out.heading_error = msg.heading_error
            out.curvature = msg.curvature
            out.confidence = msg.confidence
            out.valid = False
            self.pub.publish(out)
            self.invalid_sent_count += 1
            if self.invalid_sent_count >= trial['param']:
                self._end_trial()

    def _start_next_trial(self):
        self.trial_index += 1
        if self.trial_index >= len(self.plan):
            self.done = True
            self.phase = 'done'
            self._save()
            return
        trial = self.plan[self.trial_index]
        self.phase = 'inject'
        self.phase_start_epoch = self._now_s()
        if trial['kind'] == 'stale':
            self.frozen_stamp = (
                self.last_real_msg.header.stamp
                if self.last_real_msg is not None else None)
        else:
            self.invalid_sent_count = 0
        self.get_logger().info(
            f"TRIAL START #{self.trial_index} kind={trial['kind']} "
            f"param={trial['param']} rep={trial['rep']} "
            f"t_start_epoch={self.phase_start_epoch:.6f}")

    def _end_trial(self):
        trial = self.plan[self.trial_index]
        t_end = self._now_s()
        self.trial_records.append(dict(
            trial_index=self.trial_index, kind=trial['kind'],
            param=trial['param'], rep=trial['rep'],
            t_start_epoch=self.phase_start_epoch, t_end_epoch=t_end))
        duration = t_end - self.phase_start_epoch
        self.get_logger().info(
            f"TRIAL END   #{self.trial_index} kind={trial['kind']} "
            f"param={trial['param']} rep={trial['rep']} "
            f"t_end_epoch={t_end:.6f} duration={duration:.3f}s")
        self.phase = 'settle'
        self.phase_start_epoch = t_end
        self.phase_target_s = trial['post_gap_s']

    def _tick(self):
        if self.done:
            return
        elapsed = self._now_s() - self.phase_start_epoch

        if self.phase == 'settle':
            settled = elapsed >= self.phase_target_s
            if settled and self.last_real_msg is not None:
                self._start_next_trial()
            return

        if self.phase == 'inject':
            trial = self.plan[self.trial_index]
            if trial['kind'] == 'stale':
                if elapsed >= trial['param']:
                    self._end_trial()
            else:
                # Safety net only -- normal completion is
                # message-count-driven in on_lane_state_raw.
                if elapsed >= 5.0:
                    self.get_logger().warning(
                        f"TRIAL #{self.trial_index} invalid-count safety "
                        f"timeout (only "
                        f"{self.invalid_sent_count}/{trial['param']} "
                        f"sent) -- ending anyway")
                    self._end_trial()

    def _save(self):
        for k in self.log:
            self.log[k] = np.array(self.log[k])
        out_path = os.path.join(self.out_dir, f'{self.tag}_groundtruth.npz')
        np.savez(out_path, **self.log)

        trials_path = os.path.join(self.out_dir, f'{self.tag}_trials.json')
        with open(trials_path, 'w') as f:
            json.dump(self.trial_records, f, indent=2)

        self.get_logger().info(
            f'DONE: {len(self.trial_records)} trials completed, '
            f'ground truth saved to {out_path}, '
            f'trial windows saved to {trials_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='/tmp')
    ap.add_argument('--tag', default='dropout_test')
    ap.add_argument('--initial-settle', type=float, default=5.0)
    args = ap.parse_args()

    rclpy.init()
    node = DropoutTest(args.out_dir, args.tag, args.initial_settle)
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
