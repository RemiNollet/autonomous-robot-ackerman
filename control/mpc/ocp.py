"""
AcadosOcp setup for the lateral MPC (ADR-19): body-frame path tracking
against the /lane_state contract's quadratic reference (docs/lane-state-
contract.md section 1), against control/mpc/model.py's kinematic bicycle.

Cost: NONLINEAR_LS, stage residual y = [e_lat, e_psi, delta, ddelta] in
RAW physical units (m, rad, rad, rad/s) -- weight normalization (ADR-19)
is applied via W = diag(1/envelope^2) per term (control/mpc/params.py),
not by pre-scaling the residual itself; the two are mathematically
equivalent for a quadratic cost ((r/s)^2 == r^2 * (1/s^2)) and this way
the residual expression stays simple to read. Terminal cost residual is
[e_lat, e_psi, delta - atan(L*kappa)] (kappa=2*c2): the terminal target for
delta is the curved-reference equilibrium, not 0 unconditionally (terminal-
cost fix completing ADR-19, docs/decisions.md -- delta=0 was only correct
because ADR-19 was written when curvature was always 0). Uses the same
weights (Q=diag(W_E_LAT,W_E_PSI,W_DELTA), R=[[W_U]]) fed into a discrete
algebraic Riccati equation (scipy.linalg.solve_discrete_are) on the
linearized error dynamics, evaluated once at a nominal speed
(params.V_NOMINAL_FOR_DARE) and kept kappa-agnostic (checked: <=0.62% max
entrywise effect at this track's tightest curvature, see
_terminal_dare_matrix's own docstring) -- acados does not support a
speed- or curvature-varying terminal cost weight without a much more
involved parametric-Riccati setup, out of scope here.

Constraints: delta hard-bounded (params.DELTA_MAX, measured -- car.xml's
own steering joint range). ddelta hard-bounded (params.DELTA_DOT_MAX,
measured -- ADR-20, see that constant's own docstring: car.xml's own
position-actuator step response, not a bench test against real hardware,
since none exists yet). e_lat SOFT-constrained (slacked) at
+-LANE_HALF_WIDTH: a receding-horizon controller should be able to
command its way back toward the lane rather than report infeasible if a
disturbance briefly pushes e_lat past the lane edge within the horizon.

Solver: ERK (4-stage, acados' default RK order for integrator_type='ERK')
for the actual dynamics integration -- note this is DIFFERENT from the
first-order Euler discretization used only for the DARE linearization
above (a deliberate approximation specific to the terminal-cost
computation, not the dynamics the solver actually integrates).
PARTIAL_CONDENSING_HPIPM QP solver, SQP (not SQP_RTI -- that's a later
task once this formulation is validated, ADR-19).
"""
import casadi as ca
import numpy as np
import scipy.linalg

from control.mpc.model import kinematic_bicycle_model
from control.mpc.params import (
    DELTA_DOT_MAX, DELTA_MAX, L, N_HORIZON, TS,
    V_NOMINAL_FOR_DARE, W_DELTA, W_E_LAT, W_E_PSI, W_U,
)

# Slack penalty on the soft e_lat constraint: linear + quadratic terms,
# both sides. Initial tuning values (large enough to discourage violation
# strongly without being astronomically stiff), not yet tuned against
# closed-loop behavior -- same "not yet tuned" status as ADR-15's K1
# placeholder, stated plainly rather than presented as final.
_E_LAT_SLACK_LINEAR = 1.0e2
_E_LAT_SLACK_QUADRATIC = 1.0e4


def _path_tracking_residual(x, p):
    """e_lat, e_psi of the predicted state x=[X,Y,psi,delta] against the
    reference quadratic y(x) = c0 + c1*X + c2*X^2 (p=[c0,c1,c2,v]),
    evaluated at the predicted X -- this is what makes it BODY-FRAME PATH
    TRACKING (ADR-19) rather than Frenet regulation: the reference is a
    function of the predicted X along the horizon, not a single fixed
    target."""
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
    (V_NOMINAL_FOR_DARE):
      e_lat_dot = v * e_psi           (sin(e_psi) ~= e_psi)
      e_psi_dot = (v / L) * delta     (tan(delta) ~= delta, curvature term
                                        dropped -- the terminal cost is a
                                        LOCAL stabilizing approximation
                                        near the reference, not a model of
                                        the full nonlinear tracking problem
                                        the stage cost/dynamics already
                                        handle exactly)
      delta_dot = u
    discretized with a first-order (Euler) hold at Ts -- an approximation
    specific to this linearization, not the ERK integration the OCP's
    actual stage dynamics use.

    Linearized about delta=0 (kappa=0) regardless of the actual reference
    curvature, even though the terminal cost's target delta is now
    atan(L*kappa) rather than 0 (terminal-cost fix completing ADR-19,
    docs/decisions.md). This is a deliberate, checked approximation, not an
    oversight: re-linearizing tan(delta) about the true equilibrium
    delta_eq=atan(L*kappa) instead of 0 changes the e_psi/delta coupling
    term from v/L to (v/L)*(1+(L*kappa)^2) (sec^2(delta_eq)); at this
    track's tightest curvature (kappa_max=1/R_min=0.333 1/m,
    L*kappa_max=0.0867) that changes the resulting P by 0.26% (Frobenius
    norm) / 0.62% (max entry) -- smaller than the Euler-vs-ERK4 and
    single-nominal-speed approximations this same computation already
    makes, and acados' W_e must be a fixed numeric matrix in any case (a
    kappa-varying P needs the same parametric-Riccati machinery already
    ruled out of scope above for a speed-varying one). Kept kappa-agnostic
    on that basis, not left unexamined.
    """
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


def build_ocp(c0: float = 0.0, c1: float = 0.0, c2: float = 0.0, v: float = 1.0):
    """Returns a configured AcadosOcp. c0/c1/c2/v seed ocp.parameter_values
    (the nominal values used at build/codegen time); actual solves should
    set the same values at every shooting node via
    ocp_solver.set(i, "p", ...) -- see solve_fixed_reference below."""
    from acados_template import AcadosOcp

    ocp = AcadosOcp()
    model = kinematic_bicycle_model()
    ocp.model = model

    nx = model.x.rows()
    nu = model.u.rows()

    ocp.solver_options.N_horizon = N_HORIZON
    ocp.solver_options.tf = N_HORIZON * TS

    # --- cost: NONLINEAR_LS, stage ------------------------------------
    e_lat, e_psi = _path_tracking_residual(model.x, model.p)
    delta = model.x[3]
    ddelta = model.u[0]

    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.model.cost_y_expr = ca.vertcat(e_lat, e_psi, delta, ddelta)
    ocp.cost.yref = np.zeros(4)
    ocp.cost.W = np.diag([W_E_LAT, W_E_PSI, W_DELTA, W_U])

    # --- cost: NONLINEAR_LS, terminal (DARE) --------------------------
    # Terminal target for delta is NOT unconditionally 0: the true terminal
    # equilibrium on a curved reference (kappa = 2*c2, lane-state-
    # contract.md section 1) is delta_eq = atan(L*kappa) -- at
    # equilibrium, e_psi_dot = (v/L)*tan(delta) - v*kappa = 0 requires
    # tan(delta) = L*kappa. Folded into the residual itself
    # (cost_y_expr_e), not into yref_e: acados' NONLINEAR_LS cost only
    # supports a constant numeric yref/yref_e, not a parameter-dependent
    # one -- cost_y_expr_e is the symbolic side that CAN depend on
    # model.p (c0,c1,c2,v), the same pattern e_lat/e_psi already use for
    # their own (X-dependent) reference. yref_e stays all-zero; on a
    # straight reference (c2=0, kappa=0) this residual reduces to
    # delta - atan(0) = delta - 0, so straight-case behavior is
    # unchanged (verified in sanity_check.py).
    kappa_ref = 2 * model.p[2]
    delta_eq = ca.atan(L * kappa_ref)

    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.model.cost_y_expr_e = ca.vertcat(e_lat, e_psi, delta - delta_eq)
    ocp.cost.yref_e = np.zeros(3)
    ocp.cost.W_e = _terminal_dare_matrix()

    # --- constraints: hard on delta (state) ---------------------------
    ocp.constraints.lbx = np.array([-DELTA_MAX])
    ocp.constraints.ubx = np.array([DELTA_MAX])
    ocp.constraints.idxbx = np.array([3])

    # --- constraints: hard on ddelta (control) -------------------------
    # MEASURED (params.DELTA_DOT_MAX docstring, ADR-20) -- the sim's own
    # position-actuator step response, not a real hardware bench test
    # (none exists yet).
    ocp.constraints.lbu = np.array([-DELTA_DOT_MAX])
    ocp.constraints.ubu = np.array([DELTA_DOT_MAX])
    ocp.constraints.idxbu = np.array([0])

    # --- constraints: SOFT on e_lat (nonlinear h, slacked) -------------
    from perception.dataset.track_definitions import LANE_HALF_WIDTH
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

    # --- initial state and parameters ----------------------------------
    ocp.constraints.x0 = np.array([0.0, 0.0, 0.0, 0.0])
    ocp.parameter_values = np.array([c0, c1, c2, v])

    # --- solver options -------------------------------------------------
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.sim_method_num_stages = 4  # ERK4, explicit (ADR-19)
    ocp.solver_options.nlp_solver_type = "SQP"  # not SQP_RTI yet (ADR-19)

    return ocp


def solve_fixed_reference(ocp_solver, c0: float, c1: float, c2: float, v: float,
                           x0=None) -> int:
    """Sets p=[c0,c1,c2,v] identically at every shooting node (0..N) --
    one /lane_state measurement, evaluated at many points along the
    horizon, not a different measurement per node (model.py's own
    docstring) -- sets x0, and solves. Returns the acados status (0 =
    success)."""
    p = np.array([c0, c1, c2, v])
    for i in range(N_HORIZON + 1):
        ocp_solver.set(i, "p", p)
    if x0 is not None:
        ocp_solver.set(0, "lbx", x0)
        ocp_solver.set(0, "ubx", x0)
    return ocp_solver.solve()
