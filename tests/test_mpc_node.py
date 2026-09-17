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


# --- ADR-27: perception-health fallback (stale / persistently-invalid) ----

def test_fresh_valid_message_unaffected_by_health_gate():
    """A normal, healthy message goes through the ordinary solve path --
    the health gate must not fire (or change behavior) on the common
    case. Explicit test for this, not just inferred from the other tests
    above passing."""
    node = mn.MpcNode()
    try:
        published = []
        node.pub_cmd.publish = published.append

        msg = _make_lane_state(lateral_error=0.05, heading_error=0.0, curvature=0.0)
        node.on_lane_state(msg)

        assert node.n_invalid_consecutive == 0
        assert node.n_solves == 1  # a real solve happened, not a skip
        assert len(published) == 1
    finally:
        node.destroy_node()


def test_stale_message_triggers_safe_state():
    """First message calibrates the age baseline (age=0 by construction,
    AgeCalibrator's own documented cold-start property); a SECOND message
    stamped a full second EARLIER than the first reads as ~1s old under
    that calibration -- comfortably over STALE_AGE_THRESHOLD_S (0.15s),
    without needing to actually wait in the test. Odometry set to a
    nonzero speed first so a v_target=0 safe state is visibly distinct
    from the default v_target=1.0 behavior (braking, not accelerating)."""
    node = mn.MpcNode()
    try:
        published = []
        node.pub_cmd.publish = published.append

        from nav_msgs.msg import Odometry
        odom = Odometry()
        odom.twist.twist.linear.x = 1.0
        node.on_odom(odom)

        msg1 = _make_lane_state(sec=10)  # calibrates baseline, age=0
        node.on_lane_state(msg1)
        n_solves_after_first = node.n_solves

        node.delta_est = 0.3  # pretend the healthy solve above left this
        msg2 = _make_lane_state(lateral_error=0.05, sec=9)  # 1s "earlier"
        node.on_lane_state(msg2)

        assert node.n_solves == n_solves_after_first  # NO solve attempted
        cmd = published[-1]
        assert cmd.angular.z == pytest.approx(0.3 * mn.FALLBACK_DECAY)
        # v_target=0 while v_meas=1.0 -> braking (negative), not the
        # default v_target=1.0's near-zero-error accel.
        assert cmd.linear.x < 0.0
    finally:
        node.destroy_node()


def test_single_invalid_message_does_not_trigger():
    """One low-confidence frame is expected sensor noise (ADR-27), below
    N_INVALID_CONSECUTIVE_THRESHOLD -- must NOT trigger the safe state,
    confirming the counter (not a bare boolean) is what gates this."""
    node = mn.MpcNode()
    try:
        published = []
        node.pub_cmd.publish = published.append

        msg = _make_lane_state(lateral_error=0.05, valid=False, confidence=0.1)
        node.on_lane_state(msg)

        assert node.n_invalid_consecutive == 1
        assert node.n_solves == 1  # still a real solve, gate didn't fire
    finally:
        node.destroy_node()


def test_n_consecutive_invalid_triggers():
    """N_INVALID_CONSECUTIVE_THRESHOLD (3) consecutive invalid messages
    -- the run, not a single frame -- trips the safe state. The first
    N-1 calls solve normally (the count hasn't REACHED the threshold
    yet, same as test_single_invalid_message_does_not_trigger's own
    point extended) -- only the Nth call, where the count reaches the
    threshold, skips the solve and decays. delta_est isn't pinned to a
    hand-picked value here (unlike the solve-failure test): the first
    N-1 calls DO solve for real and update it, so what's checked is the
    decay RELATIONSHIP (post-trigger == pre-trigger * FALLBACK_DECAY),
    not a hardcoded number."""
    node = mn.MpcNode()
    try:
        published = []
        node.pub_cmd.publish = published.append

        msg = _make_lane_state(lateral_error=0.05, valid=False, confidence=0.1)
        for _ in range(mn.N_INVALID_CONSECUTIVE_THRESHOLD - 1):
            node.on_lane_state(msg)  # count below threshold -- solves normally each time

        assert node.n_solves == mn.N_INVALID_CONSECUTIVE_THRESHOLD - 1
        delta_before_trigger = node.delta_est

        node.on_lane_state(msg)  # count reaches threshold -- triggers

        assert node.n_invalid_consecutive == mn.N_INVALID_CONSECUTIVE_THRESHOLD
        assert node.n_solves == mn.N_INVALID_CONSECUTIVE_THRESHOLD - 1  # unchanged: skipped
        cmd = published[-1]
        assert cmd.angular.z == pytest.approx(delta_before_trigger * mn.FALLBACK_DECAY)
        assert abs(cmd.angular.z) < abs(delta_before_trigger)  # decayed, not repeated
    finally:
        node.destroy_node()


def test_recovery_resumes_normal_immediately_no_ramp():
    """A fresh+valid message right after a triggered safe state goes
    straight back to a normal solve -- no special-cased ramp-up state,
    confirmed by a real solve happening on the very next message (not
    "eventually" or after some recovery window)."""
    node = mn.MpcNode()
    try:
        published = []
        node.pub_cmd.publish = published.append

        invalid_msg = _make_lane_state(lateral_error=0.05, valid=False, confidence=0.1)
        for _ in range(mn.N_INVALID_CONSECUTIVE_THRESHOLD):
            node.on_lane_state(invalid_msg)
        # N-1 of these solved normally (count below threshold each time),
        # the last one tripped the trigger and skipped -- see
        # test_n_consecutive_invalid_triggers for the detailed breakdown.
        n_solves_at_trigger = node.n_solves
        assert n_solves_at_trigger == mn.N_INVALID_CONSECUTIVE_THRESHOLD - 1

        fresh_msg = _make_lane_state(lateral_error=0.05, heading_error=0.0,
                                      curvature=0.0, valid=True, confidence=1.0)
        node.on_lane_state(fresh_msg)

        assert node.n_invalid_consecutive == 0
        # A real solve, immediately, not deferred or ramped.
        assert node.n_solves == n_solves_at_trigger + 1
        cmd = published[-1]
        assert math.isfinite(cmd.angular.z)
    finally:
        node.destroy_node()
