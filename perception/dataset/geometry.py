"""
Track geometry and vehicle-to-centerline projection for the /lane_state contract.

Implements the sign conventions from docs/lane-state-contract.md:
  - Frame: REP-103 (x forward, y left, angles CCW positive)
  - lateral_error > 0  <=> centerline is to the LEFT of the vehicle
  - heading_error > 0  <=> lane tangent points LEFT of vehicle heading
  - curvature > 0      <=> lane curves LEFT

Track is built from parametric primitives (line segments and circular arcs)
so that curvature ground truth is exact, not estimated. This is what FR-8's
label quality depends on: a mesh-based track would make curvature an
estimation problem and the CNN would inherit that error.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass
class LaneState:
    lateral_error: float   # m
    heading_error: float   # rad
    curvature: float       # 1/m
    s: float                # arc length at projection point, for debugging/plots


# ---------------------------------------------------------------------------
# Track primitives
# ---------------------------------------------------------------------------

class LineSegment:
    """Straight segment. Curvature is identically zero."""

    def __init__(self, x0: float, y0: float, heading: float, length: float):
        self.x0, self.y0 = x0, y0
        self.heading = heading
        self.length = length
        self._dir = (math.cos(heading), math.sin(heading))

    def point_at(self, s: float) -> Tuple[float, float]:
        s = max(0.0, min(self.length, s))
        return (self.x0 + self._dir[0] * s, self.y0 + self._dir[1] * s)

    def heading_at(self, s: float) -> float:
        return self.heading

    def curvature_at(self, s: float) -> float:
        return 0.0

    def project(self, px: float, py: float) -> Tuple[float, float]:
        """Closed-form perpendicular projection onto the segment.

        Returns (s_local, distance), s_local clamped to [0, length].
        """
        dx, dy = px - self.x0, py - self.y0
        s_local = dx * self._dir[0] + dy * self._dir[1]
        s_local = max(0.0, min(self.length, s_local))
        cx, cy = self.point_at(s_local)
        dist = math.hypot(px - cx, py - cy)
        return s_local, dist


class Arc:
    """Circular arc. curvature = sign(sweep) / radius (constant along the arc).

    sweep > 0 is a CCW (left) turn, sweep < 0 is CW (right), matching the
    curvature sign convention. start_angle is the angle from the center to
    the arc's start point, in world frame.
    """

    def __init__(self, cx: float, cy: float, radius: float, start_angle: float, sweep: float):
        self.cx, self.cy = cx, cy
        self.radius = radius
        self.start_angle = start_angle
        self.sweep = sweep
        self.direction = 1.0 if sweep >= 0 else -1.0
        self.length = radius * abs(sweep)
        self._curvature = self.direction / radius

    def _angle_at(self, s: float) -> float:
        s = max(0.0, min(self.length, s))
        return self.start_angle + self.direction * (s / self.radius)

    def point_at(self, s: float) -> Tuple[float, float]:
        theta = self._angle_at(s)
        return (self.cx + self.radius * math.cos(theta), self.cy + self.radius * math.sin(theta))

    def heading_at(self, s: float) -> float:
        theta = self._angle_at(s)
        # CCW travel: tangent leads the radius vector by +90deg.
        # CW travel: tangent trails by -90deg (direction = -1 handles this).
        return wrap_to_pi(theta + self.direction * math.pi / 2)

    def curvature_at(self, s: float) -> float:
        return self._curvature

    def project(self, px: float, py: float) -> Tuple[float, float]:
        """Nearest point on the arc (not the full circle) via angular clamping."""
        angle_to_point = math.atan2(py - self.cy, px - self.cx)
        # Shortest signed angular gap from start_angle to the point, then
        # scaled by direction*radius to get arc length travelled. Wrapping
        # BEFORE applying direction (not after) is what makes this correct
        # for CW arcs too — applying direction twice cancels itself out near
        # the branch cut and silently returns s_local=0 instead of the true
        # projection (caught by test_arc_right_turn_curvature).
        angle_gap = wrap_to_pi(angle_to_point - self.start_angle)
        s_local = self.direction * self.radius * angle_gap
        s_local = max(0.0, min(self.length, s_local))
        cx, cy = self.point_at(s_local)
        dist = math.hypot(px - cx, py - cy)
        return s_local, dist


class Track:
    """Ordered sequence of primitives forming a centerline."""

    def __init__(self, primitives: List):
        self.primitives = primitives
        self.starts = []
        s = 0.0
        for p in primitives:
            self.starts.append(s)
            s += p.length
        self.total_length = s

    def _locate(self, s_global: float):
        for i in range(len(self.primitives) - 1, -1, -1):
            if s_global >= self.starts[i] - 1e-9:
                return self.primitives[i], s_global - self.starts[i]
        return self.primitives[0], 0.0

    def point_at(self, s_global: float) -> Tuple[float, float]:
        p, s_local = self._locate(s_global)
        return p.point_at(s_local)

    def heading_at(self, s_global: float) -> float:
        p, s_local = self._locate(s_global)
        return p.heading_at(s_local)

    def curvature_at(self, s_global: float) -> float:
        p, s_local = self._locate(s_global)
        return p.curvature_at(s_local)

    def project(self, px: float, py: float) -> float:
        """Global nearest-point projection. Returns arc length s_global.

        Exact for our primitive set (segments + arcs each have closed-form
        local projection) — no dense sampling, no interpolation error.
        """
        best_s, best_dist = None, math.inf
        for start, p in zip(self.starts, self.primitives):
            s_local, dist = p.project(px, py)
            if dist < best_dist:
                best_dist = dist
                best_s = start + s_local
        return best_s


# ---------------------------------------------------------------------------
# Vehicle -> /lane_state
# ---------------------------------------------------------------------------

def compute_lane_state(track: Track, vehicle_x: float, vehicle_y: float, vehicle_heading: float) -> LaneState:
    """Project the vehicle onto the track centerline and compute the
    /lane_state scalars, per docs/lane-state-contract.md.

    Uses the Frenet frame at the projection point (track tangent), the
    standard cross-track/heading-error convention in path-tracking
    literature. This coincides with the vehicle-frame polynomial fit
    (c0, c1, c2) to first order, exactly so in the zero-heading-error
    limit, which is the normal operating regime for a tracking controller.
    """
    s = track.project(vehicle_x, vehicle_y)
    tx, ty = track.point_at(s)
    t_heading = track.heading_at(s)
    curvature = track.curvature_at(s)

    dx = vehicle_x - tx
    dy = vehicle_y - ty

    # Signed distance of the vehicle from the track's tangent line, using the
    # LEFT normal of the tangent direction (-sin, cos). Positive when the
    # vehicle sits to the left of the centerline.
    e_y_vehicle = -dx * math.sin(t_heading) + dy * math.cos(t_heading)

    # lateral_error is centerline-relative-to-vehicle (contract §2), the
    # opposite sign of "vehicle relative to centerline":
    #   vehicle to the vehicle's... i.e. vehicle right of centerline (e_y_vehicle<0)
    #   => centerline left of vehicle => lateral_error > 0.
    lateral_error = -e_y_vehicle

    # heading_error > 0 <=> tangent points left of vehicle heading (contract §2).
    heading_error = wrap_to_pi(t_heading - vehicle_heading)

    return LaneState(lateral_error=lateral_error, heading_error=heading_error,
                      curvature=curvature, s=s)


def reconstruct_local_path(lane_state: LaneState, x) -> float:
    """y(x) = c0 + c1*x + c2*x^2 in the vehicle frame, from the three
    /lane_state scalars. Used by the MPC preview and by the geometry tests
    to check consistency against the actual track shape.
    """
    c0 = lane_state.lateral_error
    c1 = math.tan(lane_state.heading_error)
    c2 = lane_state.curvature / 2.0
    return c0 + c1 * x + c2 * x * x
