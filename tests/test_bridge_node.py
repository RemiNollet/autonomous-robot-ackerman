"""
Tests for carsim_bridge/bridge_node.py -- the rclpy Node wrapper for the
ZeroMQ <-> ROS2 bridge.

This machine has no rclpy (ROS2 runs only in the VM -- docs/decisions.md
ADR-1): the whole module is skipped here via pytest.importorskip and runs
in the VM instead, following the same pattern as tests/test_perception_node.py.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# Second entry: carsim_bridge is a nested ament_python package
# (carsim_bridge/carsim_bridge/), so `import carsim_bridge.X` needs the
# outer carsim_bridge/ directory on sys.path too.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "carsim_bridge"))

rclpy = pytest.importorskip("rclpy")

from rclpy.time import Time as RclpyTime

import carsim_bridge.bridge_node as bn


@pytest.fixture(autouse=True)
def ros_context():
    rclpy.init()
    yield
    rclpy.try_shutdown()


def test_sim_time_to_stamp_matches_t_sim_not_wall_clock():
    """Regression test for ADR-13: header.stamp must derive from the
    decoded t_sim (simulation render time, docs/lane-state-contract.md
    section 3), never from wall-clock time at receipt. A regression back
    to self.get_clock().now() would produce a stamp around the current
    epoch time (~1.7e9+ seconds for any date in this project's timeline);
    t_sim is seconds since the sim process started (small), so the two
    are trivially distinguishable -- this doesn't just check the value
    is plausible, it checks it isn't wall-clock-shaped."""
    t_sim = 42.25
    stamp = bn.sim_time_to_stamp(t_sim)

    expected = RclpyTime(seconds=t_sim).to_msg()
    assert stamp.sec == expected.sec
    assert stamp.nanosec == expected.nanosec
    assert stamp.sec < 10_000


def test_make_odom_stamp_derives_from_header_t_sim():
    node = bn.BridgeNode()
    try:
        header = {
            't_sim': 7.5,
            'pose': {'x': 1.0, 'y': 2.0, 'yaw': 0.3},
            'twist': {'vx': 0.5, 'vy': 0.0, 'yaw_rate': 0.1},
        }
        msg = node.make_odom(header)
        expected = RclpyTime(seconds=7.5).to_msg()
        assert msg.header.stamp.sec == expected.sec
        assert msg.header.stamp.nanosec == expected.nanosec
        assert msg.header.frame_id == 'odom'
    finally:
        node.destroy_node()


def test_make_image_stamp_derives_from_header_t_sim():
    node = bn.BridgeNode()
    try:
        header = {
            't_sim': 100.125,
            'img': {'h': 240, 'w': 320, 'c': 3, 'encoding': 'rgb8'},
        }
        payload = b'\x00' * (320 * 240 * 3)
        msg = node.make_image(header, payload)
        expected = RclpyTime(seconds=100.125).to_msg()
        assert msg.header.stamp.sec == expected.sec
        assert msg.header.stamp.nanosec == expected.nanosec
        assert msg.height == 240 and msg.width == 320
    finally:
        node.destroy_node()
