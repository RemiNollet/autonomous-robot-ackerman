"""
Tests for control/mpc/model.py -- the kinematic bicycle dynamics (ADR-19).

casadi-only (no acados_template needed): runs on the Mac, unlike
tests/test_mpc_ocp.py which needs a live acados install (VM-only, same
skip pattern as tests/test_perception_node.py / test_bridge_node.py).
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

casadi = pytest.importorskip("casadi")

import casadi as ca
import numpy as np

from control.mpc.model import continuous_dynamics_expr
from control.mpc.params import L


def _rk4_step(f, x, u, p, dt):
    k1 = f(x, u, p)
    k2 = f(x + dt / 2 * k1, u, p)
    k3 = f(x + dt / 2 * k2, u, p)
    k4 = f(x + dt * k3, u, p)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def _dynamics_function():
    x = ca.SX.sym("x", 4)
    u = ca.SX.sym("u", 1)
    p = ca.SX.sym("p", 4)
    return ca.Function("f", [x, u, p], [continuous_dynamics_expr(x, u, p)])


def test_kinematic_bicycle_reproduces_arc_radius():
    """Constant delta, constant v, zero steering rate -> the vehicle
    traces a circle of radius R = L / tan(delta) -- the standard
    Ackermann kinematic relationship this model is built on. Integrate
    (RK4) over a quarter turn and check the final position sits at
    distance R from the turn's center, and the numerically integrated
    heading change matches the analytic angular rate psidot=(v/L)tan(delta)."""
    delta = 0.3   # rad, well within DELTA_MAX=0.6
    v = 1.0       # m/s
    R_expected = L / math.tan(delta)

    f = _dynamics_function()
    psidot = (v / L) * math.tan(delta)
    T_quarter_turn = (math.pi / 2) / psidot

    n_steps = 2000
    dt = T_quarter_turn / n_steps

    state = np.array([0.0, 0.0, 0.0, delta])
    u_val = np.array([0.0])                # ddelta=0 -- delta held constant
    p_val = np.array([0.0, 0.0, 0.0, v])   # c0,c1,c2 unused by the dynamics

    for _ in range(n_steps):
        state = np.array(_rk4_step(f, state, u_val, p_val, dt)).flatten()

    X, Y, psi, delta_final = state

    assert delta_final == pytest.approx(delta, abs=1e-9)   # ddelta=0 throughout
    assert psi == pytest.approx(math.pi / 2, abs=1e-4)

    # Circle center at (0, R) for a left turn starting at the origin,
    # heading along +X (psi=0): the radius vector from the center to the
    # vehicle is always perpendicular to its heading.
    cx, cy = 0.0, R_expected
    dist_from_center = math.hypot(X - cx, Y - cy)
    assert dist_from_center == pytest.approx(R_expected, rel=1e-3)


def test_straight_line_delta_zero():
    """delta=0 -> psidot=0 -> straight line along the initial heading,
    X increasing at exactly v, Y and psi unchanged."""
    v = 1.0
    f = _dynamics_function()

    state = np.array([0.0, 0.0, 0.0, 0.0])
    u_val = np.array([0.0])
    p_val = np.array([0.0, 0.0, 0.0, v])

    T = 1.0
    n_steps = 200
    dt = T / n_steps
    for _ in range(n_steps):
        state = np.array(_rk4_step(f, state, u_val, p_val, dt)).flatten()

    X, Y, psi, delta = state
    assert X == pytest.approx(v * T, abs=1e-6)
    assert Y == pytest.approx(0.0, abs=1e-9)
    assert psi == pytest.approx(0.0, abs=1e-9)
