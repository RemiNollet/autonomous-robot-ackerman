"""
Unit tests for perception/dataset/geometry.py.

Hand-computed cases on a straight segment and a known-radius arc, plus the
acceptance test specified in docs/lane-state-contract.md. This is the test
suite that resolves the sign-convention risk flagged in the SRS risk
register (M1: "mislabeled or imbalanced dataset").
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from perception.dataset.geometry import (
    LineSegment, Arc, Track, compute_lane_state, reconstruct_local_path, wrap_to_pi,
)


def approx(a, b, tol=1e-6):
    assert abs(a - b) < tol, f"expected {b}, got {a} (diff {abs(a-b)})"


# ---------------------------------------------------------------------------
# Straight line: lateral error
# ---------------------------------------------------------------------------

def test_straight_line_vehicle_right_of_centerline():
    """Contract acceptance test, part 1: vehicle to the right of the
    centerline, heading aligned with the lane -> lateral_error > 0
    (positive means centerline is to the vehicle's left)."""
    track = Track([LineSegment(0, 0, heading=0.0, length=100)])
    # y = -0.3: to the right of the x-axis centerline (REP-103, y is left)
    ls = compute_lane_state(track, vehicle_x=10.0, vehicle_y=-0.3, vehicle_heading=0.0)
    approx(ls.lateral_error, 0.3)
    approx(ls.heading_error, 0.0)
    approx(ls.curvature, 0.0)


def test_straight_line_vehicle_left_of_centerline():
    track = Track([LineSegment(0, 0, heading=0.0, length=100)])
    ls = compute_lane_state(track, vehicle_x=10.0, vehicle_y=0.5, vehicle_heading=0.0)
    approx(ls.lateral_error, -0.5)


# ---------------------------------------------------------------------------
# Straight line: heading error
# ---------------------------------------------------------------------------

def test_straight_line_tangent_left_of_heading():
    """Contract acceptance test, part 2 (corrected wording): vehicle heading
    is rotated clockwise (right) relative to the lane tangent, so the tangent
    is to the vehicle's left -> heading_error > 0 (calls for a left correction).
    """
    track = Track([LineSegment(0, 0, heading=0.0, length=100)])
    ls = compute_lane_state(track, vehicle_x=10.0, vehicle_y=0.0, vehicle_heading=-0.2)
    approx(ls.heading_error, 0.2)


def test_straight_line_tangent_right_of_heading():
    track = Track([LineSegment(0, 0, heading=0.0, length=100)])
    ls = compute_lane_state(track, vehicle_x=10.0, vehicle_y=0.0, vehicle_heading=0.2)
    approx(ls.heading_error, -0.2)


# ---------------------------------------------------------------------------
# Arc: curvature sign and magnitude
# ---------------------------------------------------------------------------

def test_arc_right_turn_curvature():
    """Quarter circle, radius 10, turning right (CW, sweep < 0).
    Expect curvature = -1/radius exactly."""
    arc = Arc(cx=0, cy=0, radius=10.0, start_angle=math.pi / 2, sweep=-math.pi / 2)
    track = Track([arc])
    # vehicle exactly on the arc at its midpoint, aligned with the tangent
    mid_s = arc.length / 2
    mx, my = arc.point_at(mid_s)
    mh = arc.heading_at(mid_s)
    ls = compute_lane_state(track, vehicle_x=mx, vehicle_y=my, vehicle_heading=mh)
    approx(ls.lateral_error, 0.0, tol=1e-6)
    approx(ls.heading_error, 0.0, tol=1e-6)
    approx(ls.curvature, -1.0 / 10.0)


def test_arc_left_turn_curvature():
    """Same geometry, sweep > 0 (CCW / left turn). Expect curvature = +1/radius."""
    arc = Arc(cx=0, cy=0, radius=10.0, start_angle=-math.pi / 2, sweep=math.pi / 2)
    track = Track([arc])
    mid_s = arc.length / 2
    mx, my = arc.point_at(mid_s)
    mh = arc.heading_at(mid_s)
    ls = compute_lane_state(track, vehicle_x=mx, vehicle_y=my, vehicle_heading=mh)
    approx(ls.curvature, 1.0 / 10.0)


def test_arc_lateral_offset_sign():
    """Vehicle offset from a point on a left-turning arc, along the local
    left-normal direction -> centerline is now to the vehicle's right,
    so lateral_error should be negative."""
    arc = Arc(cx=0, cy=0, radius=10.0, start_angle=-math.pi / 2, sweep=math.pi / 2)
    track = Track([arc])
    mid_s = arc.length / 2
    mx, my = arc.point_at(mid_s)
    mh = arc.heading_at(mid_s)
    # step 0.5 m along the left normal (-sin(mh), cos(mh))
    offset = 0.5
    vx = mx - offset * math.sin(mh)
    vy = my + offset * math.cos(mh)
    ls = compute_lane_state(track, vehicle_x=vx, vehicle_y=vy, vehicle_heading=mh)
    approx(ls.lateral_error, -offset, tol=1e-3)


# ---------------------------------------------------------------------------
# Polynomial reconstruction consistency (c0, c1, c2 vs actual track shape)
# ---------------------------------------------------------------------------

def test_polynomial_reconstruction_matches_track_near_projection():
    """y(x) rebuilt from (lateral_error, heading_error, curvature) should
    approximate the true track shape in the vehicle frame for small x,
    on a curved track. This is the check that the MPC's preview path is
    actually representative of the lane."""
    arc = Arc(cx=0, cy=0, radius=15.0, start_angle=-math.pi / 2, sweep=math.pi / 2)
    track = Track([arc])
    # vehicle slightly off the arc, not perfectly aligned
    s0 = arc.length * 0.3
    px, py = arc.point_at(s0)
    heading = arc.heading_at(s0) + 0.05  # small heading error
    # nudge position slightly off centerline too
    vx = px - 0.1 * math.sin(heading)
    vy = py + 0.1 * math.cos(heading)

    ls = compute_lane_state(track, vx, vy, heading)

    for x_lookahead in (0.5, 1.0, 2.0):
        # true track point at arc length s0 + x_lookahead (approx, since arc
        # length != vehicle-frame x exactly once heading/lateral error is
        # nonzero, but close enough at these small offsets for a tolerance check)
        s_target = min(track.total_length, s0 + x_lookahead)
        tx, ty = track.point_at(s_target)
        # transform true track point into vehicle frame
        dx, dy = tx - vx, ty - vy
        local_x = dx * math.cos(heading) + dy * math.sin(heading)
        local_y = -dx * math.sin(heading) + dy * math.cos(heading)

        y_pred = reconstruct_local_path(ls, local_x)
        # loose tolerance: this is a local quadratic approximation of a
        # circle, error grows with x and with the true heading/lateral offset
        assert abs(y_pred - local_y) < 0.05, (
            f"x={local_x:.2f}: predicted y={y_pred:.4f}, true y={local_y:.4f}"
        )


# ---------------------------------------------------------------------------
# wrap_to_pi sanity
# ---------------------------------------------------------------------------

def test_wrap_to_pi():
    approx(wrap_to_pi(math.pi + 0.1), -math.pi + 0.1)
    approx(wrap_to_pi(-math.pi - 0.1), math.pi - 0.1)
    approx(wrap_to_pi(0.5), 0.5)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
