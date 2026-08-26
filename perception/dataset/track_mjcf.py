"""
Generates the MJCF track scene from a Track object (track_definitions.py).

This is the load-bearing design choice for FR-4: the visual scene and the
ground-truth labels come from the SAME parametric Track, not two separately
authored versions that could drift apart. If you ever change the track
layout, change track_definitions.py and regenerate — never hand-edit the
output XML's marking geometry directly.

Lane boundaries are rendered as a sequence of connected thin box segments
following the centerline offset by +/- LANE_HALF_WIDTH along the local
normal, sampled at fixed arc-length intervals. This approximates a curve
with straight sub-segments, which is standard practice in MJCF since there
is no native curved-strip primitive.
"""

import math
from perception.dataset.geometry import Track, wrap_to_pi

SAMPLE_SPACING = 0.2   # m, spacing between polyline samples along each boundary
MARK_HEIGHT = 0.02      # m, thickness of the marking strip
MARK_WIDTH = 0.05       # m, width of the marking strip
ROAD_MARGIN = 3.0       # m, ground plane padding beyond the track's bounding box


def _left_normal(heading: float):
    return (-math.sin(heading), math.cos(heading))


def _sample_boundary(track: Track, side: float, half_width: float, spacing: float):
    """side = +1.0 for left boundary, -1.0 for right boundary."""
    n = max(2, int(track.total_length / spacing))
    points = []
    for i in range(n + 1):
        s = min(track.total_length, track.total_length * i / n)
        cx, cy = track.point_at(s)
        h = track.heading_at(min(s, track.total_length - 1e-6))
        nx, ny = _left_normal(h)
        points.append((cx + side * half_width * nx, cy + side * half_width * ny, h))
    return points


def _segment_geoms(points, name_prefix: str) -> str:
    """One thin box geom per consecutive pair of boundary points."""
    geoms = []
    for i in range(len(points) - 1):
        x0, y0, h0 = points[i]
        x1, y1, _ = points[i + 1]
        mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        length = math.hypot(x1 - x0, y1 - y0)
        if length < 1e-6:
            continue
        heading = math.atan2(y1 - y0, x1 - x0)
        # MuJoCo box geoms take half-extents as size, and orientation as a quaternion
        # about z for a heading rotation.
        qw = math.cos(heading / 2.0)
        qz = math.sin(heading / 2.0)
        geoms.append(
            f'<geom name="{name_prefix}_{i}" type="box" '
            f'pos="{mx:.4f} {my:.4f} {MARK_HEIGHT/2:.4f}" '
            f'quat="{qw:.6f} 0 0 {qz:.6f}" '
            f'size="{length/2:.4f} {MARK_WIDTH/2:.4f} {MARK_HEIGHT/2:.4f}" '
            f'rgba="0.95 0.95 0.95 1" contype="0" conaffinity="0"/>'
        )
    return "\n    ".join(geoms)


def _track_bounds(track: Track, margin: float):
    n = 300
    xs, ys = [], []
    for i in range(n + 1):
        s = track.total_length * i / n
        x, y = track.point_at(s)
        xs.append(x)
        ys.append(y)
    return (min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin)


def generate_lane_marking_geoms(track: Track, lane_half_width: float) -> str:
    """Just the marking <geom> elements, for splicing into an existing
    worldbody (e.g. the vehicle scene) rather than a standalone document."""
    left_pts = _sample_boundary(track, +1.0, lane_half_width, SAMPLE_SPACING)
    right_pts = _sample_boundary(track, -1.0, lane_half_width, SAMPLE_SPACING)
    left_geoms = _segment_geoms(left_pts, "lane_left")
    right_geoms = _segment_geoms(right_pts, "lane_right")
    return left_geoms + "\n    " + right_geoms


def generate_track_mjcf(track: Track, lane_half_width: float) -> str:
    left_geoms_and_right = generate_lane_marking_geoms(track, lane_half_width)

    x0, x1, y0, y1 = _track_bounds(track, ROAD_MARGIN)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    hx, hy = (x1 - x0) / 2.0, (y1 - y0) / 2.0

    return f"""<mujoco model="ackerman_track">
  <visual>
    <!-- Offscreen framebuffer large enough for preview rendering. If this
         track is merged into the vehicle scene, keep only ONE <visual>
         block in the final MJCF — a second <global offwidth/offheight>
         from the vehicle file would conflict with this one. Match to
         whichever resolution the merged scene actually needs (the onboard
         camera itself runs at 320x240 per the bridge protocol; this larger
         size is only for human-eyeball preview rendering). -->
    <global offwidth="1280" offheight="960"/>
  </visual>
  <worldbody>
    <geom name="road_surface" type="plane" pos="{cx:.3f} {cy:.3f} 0"
          size="{hx:.3f} {hy:.3f} 0.01" rgba="0.25 0.25 0.25 1"/>
    {left_geoms_and_right}
  </worldbody>
</mujoco>
"""


if __name__ == "__main__":
    from perception.dataset.track_definitions import REFERENCE_TRACK, LANE_HALF_WIDTH
    xml = generate_track_mjcf(REFERENCE_TRACK, LANE_HALF_WIDTH)
    print(xml)