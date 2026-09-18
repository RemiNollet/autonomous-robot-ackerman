"""
Tests for perception/dataset/windowed_relabel.py -- windowed curvature
relabeling investigated for ADR-14's kappa problem, both mechanisms tried.

Deterministic, full coverage of the track's 8 primitive joins (replacing
the M3 kickoff investigation's ad-hoc 96/48-sample experiment): a fixed
0.5 m offset before and after each join (16 cases, all 8 joins covered,
not a subsample), plus representative away-from-join cases on each of the
track's 3 distinct primitive types (straight, R=3m arc, R=5m arc).

Near-join cases mostly report rather than assert a pass/fail threshold --
there is no unambiguous "correct" kappa when the window straddles a
curvature discontinuity; measuring it is the point. The one exception is
windowed_curvature_average's boundedness property (asserted below), which
holds by construction near a join, not just observed to.

Away-from-join cases SHOULD agree closely with compute_lane_state's
point-projection kappa (both describe the same constant-curvature ground
truth there). windowed_lane_state (Cartesian fit) does not -- see its
xfailed tests and windowed_relabel.py's module docstring for why.
windowed_curvature_average does, exactly -- see its asserted tests below.
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from perception.dataset.geometry import compute_lane_state  # noqa: E402
from perception.dataset.track_definitions import (  # noqa: E402
    REFERENCE_TRACK,
)
from perception.dataset.windowed_relabel import (  # noqa: E402
    WINDOW_NEAR_M, windowed_curvature_average, windowed_lane_state,
)

TRACK = REFERENCE_TRACK
JOIN_OFFSET_M = 0.5  # fixed small offset, well under the window's 2.1 m span
PRIMITIVE_CASES = [
    # (label, primitive_index, pose_offset). pose_offset must satisfy
    # offset + WINDOW_FAR_M < primitive.length, or the window spills into
    # the next primitive -- contaminating an "away from any join" case
    # with exactly the join proximity it's meant to test without
    # (primitive lengths: 4.0, 4.712, 6.0, 7.854).
    ("straight", 0, 1.0),
    ("R=3m arc", 1, 1.5),
    ("R=5m arc", 3, 1.5),
]


def _pose_at_s(s):
    x, y = TRACK.point_at(s)
    heading = TRACK.heading_at(s)
    return x, y, heading


@pytest.fixture(params=list(enumerate(TRACK.starts)),
                ids=lambda p: f"join_{p[0]}")
def join(request):
    idx, join_s = request.param
    return idx, join_s


def test_before_crossing_each_join_reports_discrepancy(join):
    """Vehicle JOIN_OFFSET_M before the join -- the join sits inside the
    visible window [0.32, 2.42] m ahead. No pass/fail assertion: the
    fitted kappa here describes a window that genuinely straddles two
    different curvatures, and there is no single 'correct' scalar to
    check it against -- reported for the ADR instead."""
    idx, join_s = join
    s_vehicle = (join_s - JOIN_OFFSET_M) % TRACK.total_length
    x, y, heading = _pose_at_s(s_vehicle)

    _, _, kappa_fit = windowed_lane_state(TRACK, x, y, heading)
    kappa_before = TRACK.curvature_at(s_vehicle)
    kappa_after = TRACK.curvature_at((join_s + 1e-6) % TRACK.total_length)

    print(f"join {idx} (s={join_s:.3f}), before crossing: "
          f"pointwise kappa (at vehicle) = {kappa_before:.4f}, "
          f"pointwise kappa (just past join) = {kappa_after:.4f}, "
          f"windowed fit kappa = {kappa_fit:.4f}")

    assert math.isfinite(kappa_fit)


def test_after_crossing_each_join_matches_new_primitive(join):
    """Vehicle JOIN_OFFSET_M past the join -- the window [0.32, 2.42] m
    ahead no longer reaches back to the join at all (0.5 + 0.32 > 0), so
    this is effectively an away-from-join case: the windowed fit should
    recover the analytic curvature of whichever primitive now fully
    contains the window, same as the away-from-join tests below. No
    pass/fail assertion here either, for the same reason those don't
    pass cleanly (see the "away from any join" test below) -- reported so
    the per-join pattern is visible in the ADR, not asserted against a
    threshold already known not to hold."""
    idx, join_s = join
    s_vehicle = (join_s + JOIN_OFFSET_M) % TRACK.total_length
    x, y, heading = _pose_at_s(s_vehicle)

    _, _, kappa_fit = windowed_lane_state(TRACK, x, y, heading)
    true_kappa = TRACK.curvature_at(
        (s_vehicle + WINDOW_NEAR_M) % TRACK.total_length)

    print(f"join {idx}, after crossing: true kappa (in new primitive) = "
          f"{true_kappa:.4f}, windowed fit kappa = {kappa_fit:.4f}, "
          f"abs diff = {abs(kappa_fit - true_kappa):.4f}")

    assert math.isfinite(kappa_fit)


_XFAIL_ARC = pytest.mark.xfail(
    strict=True,
    reason=("Plain unweighted quadratic fit over [0.32, 2.42] m does not "
            "closely match pointwise ground truth on a pure arc, even "
            "with zero discontinuity present (windowed_relabel.py "
            "docstring; M3 'quadratic curvature relabeling' report) -- "
            "must keep failing with these specific numbers until the fit "
            "method changes, not silently stop being checked."),
)


@pytest.mark.parametrize("primitive_type,primitive_index,pose_offset", [
    pytest.param(*PRIMITIVE_CASES[0]),  # straight -- passes cleanly, see below
    pytest.param(*PRIMITIVE_CASES[1], marks=_XFAIL_ARC),
    pytest.param(*PRIMITIVE_CASES[2], marks=_XFAIL_ARC),
])
def test_windowed_kappa_matches_pointwise_away_from_any_join(
        primitive_type, primitive_index, pose_offset):
    """Away from any join, the windowed fit's kappa should agree closely
    with compute_lane_state's point-projection kappa -- both describe the
    same constant-curvature ground truth there, so this is the sanity
    check the module's own guidance requires before trusting it near
    joins. Measured: the straight case passes exactly (0.0000 discrepancy
    -- a straight line has no higher-order content for a quadratic fit to
    mismodel, at any window position). The two arc cases do not: 41% on
    R=3m, 13% on R=5m, both xfailed above -- the bias is specific to true
    curvature (quartic content in a circular arc's shape leaking into the
    fitted quadratic term, worse on tighter curves), not a generic
    property of fitting a quadratic over an off-center window."""
    s_vehicle = TRACK.starts[primitive_index] + pose_offset
    x, y, heading = _pose_at_s(s_vehicle)

    pointwise = compute_lane_state(TRACK, x, y, heading)
    _, _, kappa_windowed = windowed_lane_state(TRACK, x, y, heading)

    if pointwise.curvature != 0:
        discrepancy = (abs(kappa_windowed - pointwise.curvature)
                       / abs(pointwise.curvature))
    else:
        discrepancy = abs(kappa_windowed)

    print(f"{primitive_type}: pointwise kappa={pointwise.curvature:.4f}, "
          f"windowed kappa={kappa_windowed:.4f}, "
          f"discrepancy={discrepancy:.4f}")

    tolerance = 0.05  # 5% relative (absolute on the straight case) -- an
    # engineering bar for "close agreement", not derived from anything in
    # particular; the measured discrepancies are 4-8x over it regardless
    # of exactly where this number is drawn.
    assert discrepancy < tolerance, (
        f"{primitive_type}: windowed fit kappa ({kappa_windowed:.4f}) "
        f"disagrees with pointwise ground truth "
        f"({pointwise.curvature:.4f}) by {discrepancy:.1%} away from any "
        f"join -- the fit itself is biased, not just the old "
        f"point-projection label it's meant to replace.")


# --------------------------------------------------------------------------
# windowed_curvature_average -- the curvature-space mechanism, validated
# clean (see windowed_relabel.py's module docstring). Real assertions
# below, not xfail: this method has no known failure mode on v0.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "primitive_type,primitive_index,pose_offset", PRIMITIVE_CASES)
def test_windowed_curvature_average_matches_pointwise_away_from_any_join(
        primitive_type, primitive_index, pose_offset):
    """Away from any join, curvature-space averaging should recover the
    true constant curvature exactly, not approximately: it never fits a
    polynomial to the centerline's position, so there's no shape to
    mismodel -- averaging a constant function over any window (any
    weighting) returns that same constant. Measured: 0.00000000 diff, all
    three primitive types. A tight tolerance here is the point, not
    over-precision -- this is what distinguishes this method from
    windowed_lane_state's 13-41% away-from-join bias just above."""
    s_vehicle = TRACK.starts[primitive_index] + pose_offset
    x, y, heading = _pose_at_s(s_vehicle)

    pointwise = compute_lane_state(TRACK, x, y, heading)
    kappa_avg = windowed_curvature_average(TRACK, x, y)

    print(f"{primitive_type}: pointwise kappa={pointwise.curvature:.6f}, "
          f"windowed average kappa={kappa_avg:.6f}")

    assert kappa_avg == pytest.approx(pointwise.curvature, abs=1e-9)


def test_windowed_curvature_average_is_bounded_near_a_join(join):
    """Near a join (window straddling the discontinuity), the average of a
    step function over an interval is always between the step's two
    values -- a property that holds by construction (it's a convex
    combination), not something that merely happened to measure true in
    these 8 cases. Checked at both the 'before crossing' pose (join inside
    the window) and stepping slightly closer/further to confirm it holds
    across the transition, not just at one arbitrarily chosen offset."""
    idx, join_s = join
    # % total_length on both: join_0 is at s=0.0, and Track._locate doesn't
    # handle a raw negative s (it falls through to its primitives[0]
    # fallback instead of correctly wrapping to the last primitive) --
    # caught by this test itself failing against join_0 before this fix.
    kappa_before = TRACK.curvature_at((join_s - 1e-6) % TRACK.total_length)
    kappa_after = TRACK.curvature_at((join_s + 1e-6) % TRACK.total_length)
    lo, hi = sorted([kappa_before, kappa_after])

    for offset in (0.1, 0.5, 1.0, 2.0):
        s_vehicle = (join_s - offset) % TRACK.total_length
        x, y = TRACK.point_at(s_vehicle)
        kappa_avg = windowed_curvature_average(TRACK, x, y)
        assert lo - 1e-9 <= kappa_avg <= hi + 1e-9, (
            f"join {idx}, offset {offset}: windowed average kappa "
            f"({kappa_avg:.4f}) fell outside [{lo:.4f}, {hi:.4f}], the "
            f"range bounded by the two curvatures either side of the join "
            f"-- boundedness should hold unconditionally for a weighted "
            f"average of a two-valued step function.")
