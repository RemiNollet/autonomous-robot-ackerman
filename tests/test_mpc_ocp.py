"""
Tests for control/mpc/ocp.py -- AcadosOcp build and solve (ADR-19).

acados_template needed (a live acados install with codegen + a compiled
core library, VM-only per docs/decisions.md ADR-1) -- skips entirely on
the Mac, same pattern as tests/test_perception_node.py /
test_bridge_node.py.

Minimal build-correctness gate: OCP builds, and solve status is 0 (not
diverged/max-iter) on three fixed, hand-picked (c0,c1,c2,v) cases. Output
plausibility (does the trajectory make physical sense, tuning of the
soft-constraint/DARE weights, etc.) is a separate, dedicated sanity-check
task -- not attempted here.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

acados_template = pytest.importorskip("acados_template")

import numpy as np  # noqa: E402

from control.mpc.ocp import build_ocp, solve_fixed_reference  # noqa: E402
from control.mpc.params import R_MIN  # noqa: E402
from perception.dataset.track_definitions import (  # noqa: E402
    REFERENCE_TRACK,
)
from perception.dataset.windowed_relabel import (  # noqa: E402
    windowed_curvature_average,
)


def _make_solver(c0=0.0, c1=0.0, c2=0.0, v=1.0):
    from acados_template import AcadosOcpSolver
    ocp = build_ocp(c0, c1, c2, v)
    return AcadosOcpSolver(ocp, verbose=False)


def test_ocp_builds_without_error():
    ocp = build_ocp()
    assert ocp is not None
    assert ocp.model.x.rows() == 4
    assert ocp.model.u.rows() == 1
    assert ocp.model.p.rows() == 4


def test_solve_succeeds_straight():
    """c0=c1=c2=0: a straight reference, x0 already on it -- the easiest
    case, expected to converge immediately."""
    solver = _make_solver(c0=0.0, c1=0.0, c2=0.0, v=1.0)
    status = solve_fixed_reference(solver, 0.0, 0.0, 0.0, 1.0, x0=np.zeros(4))
    assert status == 0


def test_solve_succeeds_tight_arc_away_from_join():
    """R=3m arc, away from any primitive join -- windowed_curvature_average
    agrees exactly with the analytic curvature there (ADR-18): the true
    curvature-space kappa, kappa=1/R_MIN, no blending involved."""
    kappa = 1.0 / R_MIN
    c2 = kappa / 2.0
    solver = _make_solver(c0=0.0, c1=0.0, c2=c2, v=1.0)
    status = solve_fixed_reference(solver, 0.0, 0.0, c2, 1.0, x0=np.zeros(4))
    assert status == 0


def test_solve_succeeds_tight_arc_near_join():
    """Kappa blended per windowed_relabel.py's validated method (ADR-18),
    at a pose 0.5 m before the straight->R=3m join -- the exact
    near-join scenario ADR-18's measurement covers, not an arbitrary
    blend."""
    track = REFERENCE_TRACK
    join_s = track.starts[1]  # straight -> R=3m arc join
    s_vehicle = (join_s - 0.5) % track.total_length
    x, y = track.point_at(s_vehicle)
    kappa = windowed_curvature_average(track, x, y)
    c2 = kappa / 2.0

    solver = _make_solver(c0=0.0, c1=0.0, c2=c2, v=1.0)
    status = solve_fixed_reference(solver, 0.0, 0.0, c2, 1.0, x0=np.zeros(4))
    assert status == 0
