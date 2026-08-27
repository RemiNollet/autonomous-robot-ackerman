"""
Reference track for the Ackerman robot project.

Built entirely from the primitives already validated in geometry.py — the
same Track object is the single source of truth used both to generate the
visual MJCF scene (track_mjcf.py) and the ground-truth /lane_state labels
(dataset generator, M1). This is deliberate: hand-authoring the MJCF
geometry separately from the label geometry would let the two drift apart
silently, exactly the failure mode ADR-7 caught in the projection code
itself.

Closure by construction, not by numeric fitting:
  The loop is two identical halves, each turning exactly 180 degrees
  (two 90-degree arcs). After the first half, vehicle heading has rotated
  180 degrees; repeating the identical primitive sequence from there
  retraces the same local motions but mirrored, so the second half's net
  displacement exactly cancels the first half's. This holds for ANY choice
  of straight lengths and radii, as long as each half turns exactly pi
  radians — no equation-solving required. Verified regardless in
  tests/test_track_definitions.py, since a proof on paper is not a
  substitute for checking the arithmetic.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from perception.dataset.geometry import LineSegment, Arc, Track
import math

# Two distinct radii and two distinct straight lengths -> curvature values
# {0, 1/R1, 1/R2} appear in the dataset, not just a single constant turn.
STRAIGHT_A = 4.0   # m
STRAIGHT_B = 6.0   # m
RADIUS_1 = 3.0     # m (tighter turn)
RADIUS_2 = 5.0     # m (wider turn)

LANE_HALF_WIDTH = 0.4  # m, distance from centerline to each lane boundary


def _half(x0, y0, heading0, straight_len, radius):
    """One quarter of the loop: a straight followed by a 90-degree left arc.
    Returns the primitives and the (x, y, heading) at its end, so the next
    quarter can be chained from there.
    """
    seg = LineSegment(x0, y0, heading0, straight_len)
    ex, ey = seg.point_at(straight_len)

    # Arc center is 90 degrees left of the heading at the segment's end.
    cx = ex - radius * math.sin(heading0)
    cy = ey + radius * math.cos(heading0)
    start_angle = math.atan2(ey - cy, ex - cx)
    arc = Arc(cx, cy, radius, start_angle=start_angle, sweep=math.pi / 2)
    ax, ay = arc.point_at(arc.length)
    aheading = arc.heading_at(arc.length)

    return [seg, arc], (ax, ay, aheading)


def build_reference_track() -> Track:
    x, y, heading = 0.0, 0.0, 0.0
    primitives = []

    for _ in range(2):  # two identical halves, per the closure argument above
        p1, (x, y, heading) = _half(x, y, heading, STRAIGHT_A, RADIUS_1)
        p2, (x, y, heading) = _half(x, y, heading, STRAIGHT_B, RADIUS_2)
        primitives.extend(p1)
        primitives.extend(p2)

    return Track(primitives)


REFERENCE_TRACK = build_reference_track()
