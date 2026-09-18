"""AcadosOcp setup for the lateral MPC (ADR-19): body-frame path tracking
against the /lane_state contract's quadratic reference (docs/lane-state-
contract.md section 1), against control/mpc/model.py's kinematic bicycle.

Cost: NONLINEAR_LS, stage residual y = [e_lat, e_psi,
delta - atan(L*kappa), ddelta] in raw physical units, kappa=2*c2 --
delta's target is the curved-reference equilibrium at every stage
(ADR-19, ADR-21). Weight normalization W = diag(1/envelope^2) per term
(control/mpc/params.py), applied to the weight rather than the residual
(mathematically equivalent for a quadratic cost). Terminal cost uses
the same weights fed into a DARE (scipy.linalg.solve_discrete_are),
evaluated once at a nominal speed and kept kappa-agnostic (checked:
<=0.62% max entrywise effect at this track's tightest curvature --
_terminal_dare_matrix's docstring, docs/decisions.md ADR-19).

Constraints: delta and ddelta hard-bounded (params.DELTA_MAX,
params.DELTA_DOT_MAX). e_lat soft-constrained (slacked) at
+-LANE_HALF_WIDTH so the controller can command its way back toward
the lane rather than report infeasible on a brief disturbance.

Solver: ERK4 for the dynamics integration (different from the
first-order Euler discretization used only for the DARE linearization
above). PARTIAL_CONDENSING_HPIPM QP solver, SQP, not SQP_RTI (ADR-19).
"""
import casadi as ca
import numpy as np
import scipy.linalg

from control.mpc.model import kinematic_bicycle_model
from control.mpc.params import (
    DELTA_DOT_MAX, DELTA_MAX, L, N_HORIZON, TS,
    V_NOMINAL_FOR_DARE, W_DELTA, W_E_LAT, W_E_PSI, W_U,
)
from perception.dataset.track_definitions import LANE_HALF_WIDTH

# Slack penalty on the soft e_lat constraint, both sides. Initial
# values, not yet tuned against closed-loop behavior.
_E_LAT_SLACK_LINEAR = 1.0e2
_E_LAT_SLACK_QUADRATIC = 1.0e4


def _path_tracking_residual(x, p):
    """e_lat, e_psi of the predicted state x=[X,Y,psi,delta] against the
    reference quadratic y(x) = c0 + c1*X + c2*X^2 (p=[c0,c1,c2,v]),
    evaluated at the predicted X -- body-frame path tracking (ADR-19)
    rather than Frenet regulation."""
    X, Y, psi = x[0], x[1], x[2]
    c0, c1, c2 = p[0], p[1], p[2]
    y_ref = c0 + c1 * X + c2 * X ** 2
    slope_ref = c1 + 2 * c2 * X
    e_lat = Y - y_ref
    e_psi = psi - ca.atan(slope_ref)
    return e_lat, e_psi


def _terminal_dare_matrix() -> np.ndarray:
    """DARE terminal cost (ADR-19). Linearized closed-loop error dynamics
    about e_lat=e_psi=0, small delta, at a single nominal speed
    (V_NOMINAL_FOR_DARE), discretized with a first-order Euler hold at
    Ts (not the ERK4 integration the OCP's actual dynamics use).

    Linearized about delta=0 (kappa=0) regardless of the actual
    reference curvature -- a checked approximation, not an oversight:
    re-linearizing about the true equilibrium changes the resulting P
    by <=0.62% (max entry) at this track's tightest curvature, smaller
    than the other approximations this computation already makes.
    Kept kappa-agnostic on that basis (docs/decisions.md ADR-19)."""
    v = V_NOMINAL_FOR_DARE
    A_c = np.array([
        [0.0, v, 0.0],
        [0.0, 0.0, v / L],
        [0.0, 0.0, 0.0],
    ])
    B_c = np.array([[0.0], [0.0], [1.0]])

    A_d = np.eye(3) + TS * A_c
    B_d = TS * B_c

    Q = np.diag([W_E_LAT, W_E_PSI, W_DELTA])
    R = np.array([[W_U]])

    return scipy.linalg.solve_discrete_are(A_d, B_d, Q, R)


def build_ocp(
        c0: float = 0.0, c1: float = 0.0, c2: float = 0.0, v: float = 1.0):
    """Returns a configured AcadosOcp. c0/c1/c2/v seed
    ocp.parameter_values (the nominal values used at build/codegen
    time); actual solves should set the same values at every shooting
    node via ocp_solver.set(i, "p", ...) -- see solve_fixed_reference
    below."""
    from acados_template import AcadosOcp

    ocp = AcadosOcp()
    model = kinematic_bicycle_model()
    ocp.model = model

    ocp.solver_options.N_horizon = N_HORIZON
    ocp.solver_options.tf = N_HORIZON * TS

    # delta's target is atan(L*kappa) at every stage, not just the
    # terminal node -- fixes an internal inconsistency, does not by
    # itself fix the closed-loop plateau bias (root cause: the
    # quadratic reference is a truncation of the true circular arc,
    # docs/decisions.md ADR-19/ADR-21).
    kappa_ref = 2 * model.p[2]
    delta_eq = ca.atan(L * kappa_ref)

    e_lat, e_psi = _path_tracking_residual(model.x, model.p)
    delta = model.x[3]
    ddelta = model.u[0]

    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.model.cost_y_expr = ca.vertcat(
        e_lat, e_psi, delta - delta_eq, ddelta)
    ocp.cost.yref = np.zeros(4)
    ocp.cost.W = np.diag([W_E_LAT, W_E_PSI, W_DELTA, W_U])

    # Terminal cost: same delta_eq, folded into cost_y_expr_e (not
    # yref_e, which acados requires to be constant numeric).
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.model.cost_y_expr_e = ca.vertcat(e_lat, e_psi, delta - delta_eq)
    ocp.cost.yref_e = np.zeros(3)
    ocp.cost.W_e = _terminal_dare_matrix()

    # --- constraints: hard on delta (state) -----------------------------
    ocp.constraints.lbx = np.array([-DELTA_MAX])
    ocp.constraints.ubx = np.array([DELTA_MAX])
    ocp.constraints.idxbx = np.array([3])

    # --- constraints: hard on ddelta (control), ADR-20 ------------------
    ocp.constraints.lbu = np.array([-DELTA_DOT_MAX])
    ocp.constraints.ubu = np.array([DELTA_DOT_MAX])
    ocp.constraints.idxbu = np.array([0])

    # --- constraints: soft on e_lat (nonlinear h, slacked) --------------
    ocp.model.con_h_expr = e_lat
    ocp.constraints.lh = np.array([-LANE_HALF_WIDTH])
    ocp.constraints.uh = np.array([LANE_HALF_WIDTH])
    ocp.constraints.idxsh = np.array([0])
    ocp.cost.zl = np.array([_E_LAT_SLACK_LINEAR])
    ocp.cost.Zl = np.array([_E_LAT_SLACK_QUADRATIC])
    ocp.cost.zu = np.array([_E_LAT_SLACK_LINEAR])
    ocp.cost.Zu = np.array([_E_LAT_SLACK_QUADRATIC])

    ocp.model.con_h_expr_e = e_lat
    ocp.constraints.lh_e = np.array([-LANE_HALF_WIDTH])
    ocp.constraints.uh_e = np.array([LANE_HALF_WIDTH])
    ocp.constraints.idxsh_e = np.array([0])
    ocp.cost.zl_e = np.array([_E_LAT_SLACK_LINEAR])
    ocp.cost.Zl_e = np.array([_E_LAT_SLACK_QUADRATIC])
    ocp.cost.zu_e = np.array([_E_LAT_SLACK_LINEAR])
    ocp.cost.Zu_e = np.array([_E_LAT_SLACK_QUADRATIC])

    # --- initial state and parameters -----------------------------------
    ocp.constraints.x0 = np.array([0.0, 0.0, 0.0, 0.0])
    ocp.parameter_values = np.array([c0, c1, c2, v])

    # --- solver options ---------------------------------------------------
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.sim_method_num_stages = 4  # ERK4, explicit (ADR-19)
    ocp.solver_options.nlp_solver_type = "SQP"  # not SQP_RTI yet (ADR-19)

    return ocp


def solve_fixed_reference(
        ocp_solver, c0: float, c1: float, c2: float, v: float,
        x0=None) -> int:
    """Sets p=[c0,c1,c2,v] identically at every shooting node (0..N) --
    one /lane_state measurement, evaluated at many points along the
    horizon, not a different measurement per node -- sets x0, and
    solves. Returns the acados status (0 = success)."""
    p = np.array([c0, c1, c2, v])
    for i in range(N_HORIZON + 1):
        ocp_solver.set(i, "p", p)
    if x0 is not None:
        ocp_solver.set(0, "lbx", x0)
        ocp_solver.set(0, "ubx", x0)
    return ocp_solver.solve()
