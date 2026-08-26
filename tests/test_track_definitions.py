"""
Tests for the reference track (track_definitions.py).

Two things matter before this track is trusted for label generation:
it must close exactly (position and heading), and it must actually
contain more than one curvature magnitude — otherwise FR-4's "varying
radii" is nominal, not real.
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from perception.dataset.track_definitions import (
    REFERENCE_TRACK, RADIUS_1, RADIUS_2,
)
from perception.dataset.geometry import wrap_to_pi


def test_track_closes_position():
    start = REFERENCE_TRACK.point_at(0.0)
    end = REFERENCE_TRACK.point_at(REFERENCE_TRACK.total_length)
    dist = math.hypot(end[0] - start[0], end[1] - start[1])
    assert dist < 1e-6, f"loop does not close: gap of {dist:.6f} m"


def test_track_closes_heading():
    start_h = REFERENCE_TRACK.heading_at(0.0)
    end_h = REFERENCE_TRACK.heading_at(REFERENCE_TRACK.total_length - 1e-9)
    # heading just before closing back to the start should match start heading
    diff = abs(wrap_to_pi(end_h - start_h))
    assert diff < 1e-4, f"heading does not close: {diff:.6f} rad gap"


def test_track_has_two_distinct_radii():
    """Sample curvature along the track and confirm both radii are present,
    not just one — this is what makes the curvature label meaningful rather
    than a near-constant the CNN can shortcut around."""
    n = 500
    curvatures = [
        REFERENCE_TRACK.curvature_at(s)
        for s in [REFERENCE_TRACK.total_length * i / n for i in range(n)]
    ]
    magnitudes = {round(abs(k), 3) for k in curvatures if abs(k) > 1e-6}
    expected = {round(1.0 / RADIUS_1, 3), round(1.0 / RADIUS_2, 3)}
    assert expected.issubset(magnitudes), (
        f"expected curvature magnitudes {expected}, found {magnitudes}"
    )


def test_track_has_straight_sections():
    n = 500
    curvatures = [
        REFERENCE_TRACK.curvature_at(s)
        for s in [REFERENCE_TRACK.total_length * i / n for i in range(n)]
    ]
    zero_count = sum(1 for k in curvatures if abs(k) < 1e-6)
    assert zero_count > 0, "no straight sections found on the track"
