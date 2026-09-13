"""
Tests for carsim_bridge/mpc_node.py -- the rclpy Node wrapper around
control/mpc/ocp.py's acados solver.

This machine has neither rclpy nor acados_template (ROS2 and acados both
run only in the VM -- docs/decisions.md ADR-1): the whole module is
skipped here via pytest.importorskip and runs in the VM instead. Same
skip pattern as tests/test_perception_node.py and tests/test_mpc_ocp.py.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "carsim_bridge"))

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("acados_template")
pytest.importorskip("carsim_msgs.msg")

from rclpy.parameter import Parameter

import carsim_bridge.mpc_node as mn
from carsim_msgs.msg import LaneState
from control.mpc.params import DELTA_DOT_MAX, DELTA_MAX, L


@pytest.fixture(autouse=True)
def ros_context():
    rclpy.init()
    yield
    rclpy.try_shutdown()


def _make_lane_state(lateral_error=0.0, heading_error=0.0, curvature=0.0,
                      confidence=1.0, valid=True, sec=0, nanosec=0):
    msg = LaneState()
    msg.header.stamp.sec = sec
    msg.header.stamp.nanosec = nanosec
    msg.header.frame_id = 'base_link'
    msg.lateral_error = float(lateral_error)
    msg.heading_error = float(heading_error)
    msg.curvature = float(curvature)
    msg.confidence = float(confidence)
    msg.valid = bool(valid)
    return msg


def test_node_constructs_without_a_live_graph():
    """Node construction builds a real acados solver (control/mpc/ocp.py's
    build_ocp + AcadosOcpSolver, same formulation test_mpc_ocp.py already
    validates) -- not mocked, matching this project's own testing
    philosophy (test_mpc_ocp.py builds a real solver too, no stub)."""
    node = mn.MpcNode()
    try:
        assert node.get_name() == 'mpc_node'
        assert node.solver is not None
    finally:
        node.destroy_node()


def test_publishes_plausible_cmd_from_fixed_lane_state():
    """Smoke test: a fixed /lane_state message in, a plausible /carsim/cmd
    out. No live graph -- publish() is monkeypatched to capture instead of
    sending, same technique test_perception_node.py uses."""
    node = mn.MpcNode()
    try:
        published = []
        node.pub_cmd.publish = published.append

        msg = _make_lane_state(lateral_error=0.05, heading_error=0.02, curvature=0.0)
        node.on_lane_state(msg)

        assert len(published) == 1
        cmd = published[0]
        assert math.isfinite(cmd.angular.z)
        assert math.isfinite(cmd.linear.x)
        assert -DELTA_MAX <= cmd.angular.z <= DELTA_MAX
        assert -1.0 <= cmd.linear.x <= 1.0
    finally:
        node.destroy_node()


def test_zero_error_zero_curvature_gives_zero_steering():
    """The straight/on-reference case (sanity_check.py's own case 1) --
    the OCP should command no steering at all, not just "small"."""
    node = mn.MpcNode()
    try:
        published = []
        node.pub_cmd.publish = published.append

        msg = _make_lane_state(lateral_error=0.0, heading_error=0.0, curvature=0.0)
        node.on_lane_state(msg)

        cmd = published[0]
        assert abs(cmd.angular.z) < 1e-6
    finally:
        node.destroy_node()


def test_curvature_feedforward_matches_atan_l_kappa():
    """Sustained nonzero curvature, x0 already at the equilibrium (no
    prior odom, so age_s calibration hasn't kicked in yet and delta_est
    starts at 0 -- feed it twice so the second call's x0 reflects the
    first call's converged-ish delta) -- loose sanity check that the
    published steering moves toward atan(L*kappa), the ADR-21 fix this
    node's whole reason for existing depends on, not toward 0."""
    node = mn.MpcNode()
    try:
        published = []
        node.pub_cmd.publish = published.append

        kappa = 1.0 / 3.0  # R=3m, this track's tightest turn
        msg = _make_lane_state(lateral_error=0.0, heading_error=0.0, curvature=kappa)
        for _ in range(5):
            node.on_lane_state(msg)

        cmd = published[-1]
        feedforward = math.atan(L * kappa)
        assert cmd.angular.z > 0.0
        assert abs(cmd.angular.z - feedforward) < 0.2 * feedforward
    finally:
        node.destroy_node()


def test_solve_failure_decays_delta_toward_zero_not_repeat_last():
    """Fallback logic (ADR-24, point 3), tested in isolation from acados
    itself by monkeypatching solve_fixed_reference to fail deterministically
    -- real solver failures aren't reliably reproducible from a hand-picked
    input, but the fallback BRANCH is exactly as testable as any other."""
    node = mn.MpcNode()
    try:
        published = []
        node.pub_cmd.publish = published.append
        node.delta_est = 0.3  # pretend a prior successful solve left this

        original = mn.solve_fixed_reference
        mn.solve_fixed_reference = lambda *a, **k: 1  # any nonzero status
        try:
            msg = _make_lane_state(lateral_error=0.05, heading_error=0.0, curvature=0.0)
            node.on_lane_state(msg)
        finally:
            mn.solve_fixed_reference = original

        cmd = published[0]
        assert cmd.angular.z == pytest.approx(0.3 * mn.FALLBACK_DECAY)
        assert abs(cmd.angular.z) < 0.3  # decayed, not repeated unchanged
        assert node.n_solve_failures == 1
    finally:
        node.destroy_node()


def test_age_calibration_handles_sim_clock_vs_wall_clock_offset():
    """header.stamp is t_sim (sim-clock-relative, small numbers -- ADR-13),
    NOT wall-clock epoch time; self.get_clock().now() IS wall-clock epoch
    (no node in this graph sets use_sim_time). Directly subtracting them
    would give an astronomically large "age". This is exactly the
    calibration _age_s exists to correct -- confirm it does, not just that
    it runs."""
    node = mn.MpcNode()
    try:
        from builtin_interfaces.msg import Time as TimeMsg
        stamp = TimeMsg(sec=5, nanosec=0)  # a plausible t_sim value

        age1 = node._age_s(stamp)
        # First observation calibrates the offset -- always reads as
        # age=0 by construction (the module's own documented cold-start
        # artifact), not "wrong", just the first sample defining the
        # baseline.
        assert age1 == 0.0

        # A LATER wall-clock call against the SAME stamp must not explode
        # into a multi-year number just because sec=5 is nowhere near
        # actual epoch time -- it should stay a small, sane age.
        age2 = node._age_s(stamp)
        assert 0.0 <= age2 < 1.0
    finally:
        node.destroy_node()
