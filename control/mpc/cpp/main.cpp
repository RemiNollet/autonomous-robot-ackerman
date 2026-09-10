// Standalone C++ executable, no ROS2, no Python interface: confirms the
// acados solver works outside both, the actual deployment path for M4
// (ADR-19's formulation, control/mpc/ocp.py's generated solver).
//
// Loads the SAME generated solver control/mpc/ocp.py's Python interface
// uses (codegen_ocp_lateral_mpc_kinematic_bicycle_<hash>/, produced by
// AcadosOcpSolver(build_ocp(...))). This file's function names are tied
// to the current build's content hash (3d70c8ac as of the terminal-cost
// fix completing ADR-19, docs/decisions.md -- the curvature-aware
// terminal target changed cost_y_expr_e, which changed the hash from
// ADR-19's original a3fa00e4) -- regenerating with a different OCP
// structure changes that hash, and this file's function names need
// updating to match. Not hidden behind a macro on purpose: the coupling
// is real, and pretending otherwise would just make the next person
// rediscover it the hard way.
//
// Usage: ./mpc_standalone [c0] [c1] [c2] [v]
//   defaults to the straight case (0 0 0 1.0) if no args given.

#include <cstdio>
#include <cstdlib>

#include "acados_c/ocp_nlp_interface.h"
#include "acados_solver_ocp_lateral_mpc_kinematic_bicycle_3d70c8ac.h"

#define NX OCP_LATERAL_MPC_KINEMATIC_BICYCLE_3D70C8AC_NX
#define NU OCP_LATERAL_MPC_KINEMATIC_BICYCLE_3D70C8AC_NU
#define NP OCP_LATERAL_MPC_KINEMATIC_BICYCLE_3D70C8AC_NP
#define NBX0 OCP_LATERAL_MPC_KINEMATIC_BICYCLE_3D70C8AC_NBX0

int main(int argc, char **argv) {
    double c0 = 0.0, c1 = 0.0, c2 = 0.0, v = 1.0;
    if (argc == 5) {
        c0 = atof(argv[1]);
        c1 = atof(argv[2]);
        c2 = atof(argv[3]);
        v = atof(argv[4]);
    } else if (argc != 1) {
        fprintf(stderr, "usage: %s [c0 c1 c2 v]  (defaults: 0 0 0 1.0)\n", argv[0]);
        return 1;
    }

    auto *capsule = ocp_lateral_mpc_kinematic_bicycle_3d70c8ac_acados_create_capsule();
    int N = OCP_LATERAL_MPC_KINEMATIC_BICYCLE_3D70C8AC_N;
    int status = ocp_lateral_mpc_kinematic_bicycle_3d70c8ac_acados_create_with_discretization(
        capsule, N, nullptr);
    if (status) {
        fprintf(stderr, "acados_create() returned status %d. Exiting.\n", status);
        return 1;
    }

    ocp_nlp_config *nlp_config =
        ocp_lateral_mpc_kinematic_bicycle_3d70c8ac_acados_get_nlp_config(capsule);
    ocp_nlp_dims *nlp_dims =
        ocp_lateral_mpc_kinematic_bicycle_3d70c8ac_acados_get_nlp_dims(capsule);
    ocp_nlp_in *nlp_in = ocp_lateral_mpc_kinematic_bicycle_3d70c8ac_acados_get_nlp_in(capsule);
    ocp_nlp_out *nlp_out = ocp_lateral_mpc_kinematic_bicycle_3d70c8ac_acados_get_nlp_out(capsule);

    // Parameters p=[c0,c1,c2,v], identical at every shooting node
    // (control/mpc/model.py's own convention: one /lane_state measurement
    // evaluated at many horizon points, one speed, not per-node values).
    double p[NP] = {c0, c1, c2, v};
    for (int i = 0; i <= N; i++) {
        ocp_lateral_mpc_kinematic_bicycle_3d70c8ac_acados_update_params(capsule, i, p, NP);
    }

    // x0 = [0,0,0,0]: the body-frame origin, matching
    // control/mpc/ocp.py's solve_fixed_reference default so results are
    // directly comparable against the Python side.
    double x0[NBX0] = {0.0, 0.0, 0.0, 0.0};
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "lbx", x0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "ubx", x0);

    status = ocp_lateral_mpc_kinematic_bicycle_3d70c8ac_acados_solve(capsule);

    printf("c0=%+.6f c1=%+.6f c2=%+.6f v=%.2f\n", c0, c1, c2, v);
    printf("status: %d\n", status);

    double x[NX];
    double u[NU];
    printf("\ntrajectory (x = [X, Y, psi, delta]):\n");
    for (int i = 0; i <= N; i++) {
        ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, i, "x", x);
        printf("  step %2d: X=%+.6f Y=%+.6f psi=%+.6f delta=%+.6f\n", i, x[0], x[1], x[2], x[3]);
    }
    printf("\ncontrols (u = [ddelta]):\n");
    for (int i = 0; i < N; i++) {
        ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, i, "u", u);
        printf("  step %2d: ddelta=%+.6f\n", i, u[0]);
    }

    ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, 0, "u", u);
    printf("\nfirst applied ddelta: %+.6f rad/s\n", u[0]);

    ocp_lateral_mpc_kinematic_bicycle_3d70c8ac_acados_free(capsule);
    ocp_lateral_mpc_kinematic_bicycle_3d70c8ac_acados_free_capsule(capsule);

    return status;
}
