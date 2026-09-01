"""
Tests for carsim_bridge/perception_node.py -- the rclpy Node wrapper.

This machine has no rclpy (ROS2 runs only in the VM -- docs/decisions.md
ADR-1): the whole module is skipped here via pytest.importorskip and runs
in the VM instead. Node/preprocessing-independent tests (that don't need a
live ROS2 graph) are in tests/test_perception_inference.py and DO run here
-- importorskip at module level skips the entire file, so anything that
should run on this machine cannot share a file with these.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# Second entry: carsim_bridge is a nested ament_python package
# (carsim_bridge/carsim_bridge/), so `import carsim_bridge.X` needs the
# outer carsim_bridge/ directory on sys.path too, not just the repo root
# that resolves `perception.*` above.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "carsim_bridge"))

torch = pytest.importorskip("torch")
rclpy = pytest.importorskip("rclpy")
pytest.importorskip("carsim_msgs.msg")

from rclpy.parameter import Parameter

import carsim_bridge.perception_node as pn
from perception.model.lane_cnn import LaneCNN


@pytest.fixture
def tmp_checkpoint(tmp_path):
    """A random (untrained) checkpoint, matching train.py's save_checkpoint
    format -- node construction shouldn't require a real trained model."""
    model = LaneCNN(width_mult=1.0)
    path = tmp_path / "test_checkpoint.pt"
    torch.save({"state_dict": model.state_dict(), "width_mult": 1.0}, path)
    return str(path)


@pytest.fixture(autouse=True)
def ros_context():
    rclpy.init()
    yield
    rclpy.try_shutdown()


def test_node_constructs_without_a_live_graph(tmp_checkpoint):
    """No publisher/subscriber on the other end, no rclpy.spin() -- just
    construction, matching the task's 'constructs without a live graph'."""
    node = pn.PerceptionNode(parameter_overrides=[
        Parameter('checkpoint_path', value=tmp_checkpoint),
        Parameter('device', value='cpu'),
    ])
    try:
        assert node.get_name() == 'perception_node'
        assert node.model is not None
    finally:
        node.destroy_node()


def test_publishes_lane_state_with_propagated_stamp_from_synthetic_image(tmp_checkpoint):
    from sensor_msgs.msg import Image

    node = pn.PerceptionNode(parameter_overrides=[
        Parameter('checkpoint_path', value=tmp_checkpoint),
        Parameter('device', value='cpu'),
    ])
    try:
        published = []
        node.pub.publish = published.append  # capture instead of sending on a live graph

        rng = np.random.default_rng(1)
        img_arr = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
        msg = Image()
        msg.header.stamp.sec = 123
        msg.header.stamp.nanosec = 456789
        msg.header.frame_id = 'base_link'
        msg.height, msg.width = 240, 320
        msg.encoding = 'rgb8'
        msg.is_bigendian = 0
        msg.step = 320 * 3
        msg.data = img_arr.tobytes()

        node.on_image(msg)

        assert len(published) == 1
        out = published[0]
        # Propagated, not re-stamped: must equal the SOURCE image's stamp
        # exactly, not "close to now" -- the M1 contract's whole point.
        assert out.header.stamp.sec == 123
        assert out.header.stamp.nanosec == 456789
        assert out.header.frame_id == 'base_link'
        assert out.curvature == 0.0  # ADR-12: never the network's untrained output
        assert 0.0 <= out.confidence <= 1.0
        assert out.valid == (out.confidence >= node.confidence_threshold)
    finally:
        node.destroy_node()


def test_stats_report_fires_after_window_and_resets_buffers(tmp_checkpoint, tmp_path):
    """Regression test for a real pitfall: get_clock().now() and
    Time.from_msg(msg.header.stamp) don't share a clock_type by default, and
    subtracting two rclpy Time objects directly raises when they don't
    match. stamp is set via the node's own clock (matching what
    bridge_node.py actually does -- self.get_clock().now().to_msg() at
    receipt time), so this exercises the real age-at-publish code path, not
    an arbitrary fixed timestamp."""
    from sensor_msgs.msg import Image

    stats_path = tmp_path / "stats.md"
    node = pn.PerceptionNode(parameter_overrides=[
        Parameter('checkpoint_path', value=tmp_checkpoint),
        Parameter('device', value='cpu'),
        Parameter('stats_window_frames', value=5),
        Parameter('stats_output_path', value=str(stats_path)),
    ])
    try:
        node.pub.publish = lambda msg: None

        rng = np.random.default_rng(2)
        for _ in range(5):
            img_arr = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
            msg = Image()
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.header.frame_id = 'base_link'
            msg.height, msg.width = 240, 320
            msg.encoding = 'rgb8'
            msg.is_bigendian = 0
            msg.step = 320 * 3
            msg.data = img_arr.tobytes()
            node.on_image(msg)

        assert stats_path.exists()
        content = stats_path.read_text()
        assert 'preprocess' in content
        assert 'publish interval' in content
        assert 'age at publish' in content

        # Window reset after the report -- next frame starts a fresh window,
        # not an ever-growing list.
        assert node._t_pre_samples == []
        assert node._age_samples == []
    finally:
        node.destroy_node()
