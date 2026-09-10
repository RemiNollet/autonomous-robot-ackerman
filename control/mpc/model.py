"""
Kinematic bicycle model, body frame, for the lateral MPC (ADR-19).

State x = [X, Y, psi, delta]:
  X, Y   -- position in a body-fixed frame anchored at the vehicle's own
            pose at the start of the horizon (X=Y=psi=0 at t=0), NOT world
            coordinates. This is what makes body-frame path tracking
            (ADR-19, reversing ADR-15's Frenet choice now that ADR-18 makes
            curvature real) work directly against the /lane_state
            contract's own y(x) = c0 + c1 x + c2 x^2
            (docs/lane-state-contract.md section 1): the reference path
            and the predicted trajectory are expressed in the same frame
            by construction, with no separate transform needed.
  psi    -- heading in that same body frame (0 at t=0)
  delta  -- steering angle

Control u = [ddelta] (steering RATE, not angle): the OCP controls the
rate directly and constrains it (params.DELTA_DOT_MAX, measured -- ADR-20,
see that module), a standard choice for lane-keeping MPC:
it gives direct authority over how fast delta moves and makes delta a
state the cost/constraints act on smoothly, rather than a discontinuous
control input.

Parameters p = [c0, c1, c2, v]:
  c0, c1, c2 -- the reference path's quadratic coefficients (lateral_error,
                tan(heading_error), curvature/2 -- lane-state-contract.md
                section 1's own equivalence transform). Identical at every
                shooting node: one /lane_state measurement evaluated at
                many points along the horizon, not a different
                measurement per node.
  v          -- vehicle speed, also identical across nodes. This OCP does
                not control speed (a separate longitudinal loop does);
                it is a known, given parameter for both the lateral
                dynamics below and the path-tracking cost (control/ocp.py).

Dynamics (kinematic bicycle, no slip -- justified for this vehicle's
measured operating envelope in docs/decisions.md ADR-16, a_lat=0.034g at
v=1.0 m/s / R_min=3m, ~11% of the ~0.3g slip-angle rule of thumb):
  Xdot     = v * cos(psi)
  Ydot     = v * sin(psi)
  psidot   = (v / L) * tan(delta)
  deltadot = u
"""
import casadi as ca

from control.mpc.params import L


def kinematic_bicycle_model(name: str = "lateral_mpc_kinematic_bicycle"):
    """Returns an AcadosModel (import deferred to inside this function: the
    model's *symbolic* construction only needs casadi, which is installed
    on the Mac too -- tests/test_mpc_model.py exercises this without
    needing acados_template or a live acados install, only
    tests/test_mpc_ocp.py needs those, VM-only)."""
    from acados_template import AcadosModel

    X = ca.SX.sym("X")
    Y = ca.SX.sym("Y")
    psi = ca.SX.sym("psi")
    delta = ca.SX.sym("delta")
    x = ca.vertcat(X, Y, psi, delta)

    ddelta = ca.SX.sym("ddelta")
    u = ca.vertcat(ddelta)

    c0 = ca.SX.sym("c0")
    c1 = ca.SX.sym("c1")
    c2 = ca.SX.sym("c2")
    v = ca.SX.sym("v")
    p = ca.vertcat(c0, c1, c2, v)

    xdot_sym = ca.SX.sym("xdot", x.shape[0])

    f_expl = ca.vertcat(
        v * ca.cos(psi),
        v * ca.sin(psi),
        (v / L) * ca.tan(delta),
        ddelta,
    )

    model = AcadosModel()
    model.name = name
    model.x = x
    model.u = u
    model.p = p
    model.xdot = xdot_sym
    model.f_expl_expr = f_expl
    model.f_impl_expr = xdot_sym - f_expl
    model.x_labels = ["X [m]", "Y [m]", "psi [rad]", "delta [rad]"]
    model.u_labels = ["ddelta [rad/s]"]
    model.t_label = "t [s]"

    return model


def continuous_dynamics_expr(x, u, p):
    """The same f_expl_expr as kinematic_bicycle_model, as a bare CasADi
    expression rather than wrapped in an AcadosModel -- for
    tests/test_mpc_model.py's own RK4 integration, which needs to run
    without acados_template (casadi only, Mac-testable). x, u, p are
    CasADi SX/MX/DM vectors matching kinematic_bicycle_model's layout:
    x=[X,Y,psi,delta], u=[ddelta], p=[c0,c1,c2,v] (c0/c1/c2 unused here,
    kept for a consistent parameter vector shape with the OCP model)."""
    X, Y, psi, delta = x[0], x[1], x[2], x[3]
    ddelta = u[0]
    v = p[3]
    return ca.vertcat(
        v * ca.cos(psi),
        v * ca.sin(psi),
        (v / L) * ca.tan(delta),
        ddelta,
    )
