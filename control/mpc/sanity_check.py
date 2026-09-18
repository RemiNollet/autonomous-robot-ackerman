"""Sanity-check MPC output on fixed lane states (ADR-19 follow-up, M3).

Reports the full predicted trajectory (not just solve status) for a set
of fixed (c0, c1, c2, v) cases, and computes explicit numeric criteria:
does the controller steer toward the reference (e_lat/e_psi magnitude
decreasing over the horizon), does anything saturate on a small offset
(miscalibrated-weight symptom), does left/right mirror correctly, does
a constant-curvature case's terminal delta approach the geometric
feedforward atan(L*kappa).

Run on the VM (acados_template + a live acados install needed):
    source ~/venvs/ackerman_ros2/bin/activate
    export ACADOS_SOURCE_DIR=$HOME/acados
    export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH
    python3 control/mpc/sanity_check.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from control.mpc.ocp import build_ocp, solve_fixed_reference  # noqa: E402
from control.mpc.params import (  # noqa: E402
    DELTA_DOT_MAX, DELTA_MAX, L, N_HORIZON, R_MIN,
)
from perception.dataset.track_definitions import (  # noqa: E402
    REFERENCE_TRACK,
)
from perception.dataset.windowed_relabel import (  # noqa: E402
    windowed_curvature_average,
)


def _get_trajectory(solver):
    X = np.array([solver.get(i, "x") for i in range(N_HORIZON + 1)])
    U = np.array([solver.get(i, "u") for i in range(N_HORIZON)])
    return X, U


def _e_lat_e_psi(X, c0, c1, c2):
    Xp, Y, psi = X[:, 0], X[:, 1], X[:, 2]
    y_ref = c0 + c1 * Xp + c2 * Xp ** 2
    slope_ref = c1 + 2 * c2 * Xp
    e_lat = Y - y_ref
    e_psi = psi - np.arctan(slope_ref)
    return e_lat, e_psi


def run_case(c0, c1, c2, v, x0=None):
    from acados_template import AcadosOcpSolver
    ocp = build_ocp(c0, c1, c2, v)
    solver = AcadosOcpSolver(ocp, verbose=False)
    status = solve_fixed_reference(
        solver, c0, c1, c2, v, x0=x0 if x0 is not None else np.zeros(4))
    X, U = _get_trajectory(solver)
    return status, X, U


def report_case(name, c0, c1, c2, v):
    status, X, U = run_case(c0, c1, c2, v)
    e_lat, e_psi = _e_lat_e_psi(X, c0, c1, c2)
    delta_first = float(U[0, 0])
    delta_traj = X[:, 3]

    print(f"\n=== {name} ===")
    print(f"  c0={c0:+.4f}  c1={c1:+.4f}  c2={c2:+.6f}  v={v:.2f}   "
          f"status={status}")
    print(f"  first applied ddelta (control) = {delta_first:+.6f} rad/s")
    print(f"  delta (state) trajectory: t=0 -> {delta_traj[0]:+.6f}, "
          f"mid -> {delta_traj[N_HORIZON // 2]:+.6f}, "
          f"end -> {delta_traj[-1]:+.6f} rad")
    e_lat_decreased = abs(e_lat[-1]) < abs(e_lat[0]) or abs(e_lat[0]) < 1e-9
    print(f"  e_lat: t=0 -> {e_lat[0]:+.6f} m, end -> {e_lat[-1]:+.6f} m "
          f"(|decreased|: {e_lat_decreased})")
    e_psi_decreased = abs(e_psi[-1]) < abs(e_psi[0]) or abs(e_psi[0]) < 1e-9
    print(f"  e_psi: t=0 -> {e_psi[0]:+.6f} rad, end -> {e_psi[-1]:+.6f} "
          f"rad (|decreased|: {e_psi_decreased})")
    ddelta_sat = bool(np.any(np.abs(U[:, 0]) >= 0.999 * DELTA_DOT_MAX))
    delta_sat = bool(np.any(np.abs(delta_traj) >= 0.999 * DELTA_MAX))
    print(f"  ddelta saturates anywhere in horizon: {ddelta_sat} "
          f"(max |ddelta|={np.max(np.abs(U[:, 0])):.4f} "
          f"vs limit {DELTA_DOT_MAX})")
    print(f"  delta saturates anywhere in horizon: {delta_sat} "
          f"(max |delta|={np.max(np.abs(delta_traj)):.4f} "
          f"vs limit {DELTA_MAX})")

    return dict(status=status, X=X, U=U, e_lat=e_lat, e_psi=e_psi,
                delta_first=delta_first, delta_traj=delta_traj,
                c0=c0, c1=c1, c2=c2, v=v)


def main():
    track = REFERENCE_TRACK
    kappa_r3_away = 1.0 / R_MIN  # exact away from join (ADR-18)
    join_s = track.starts[1]
    s_near = (join_s - 0.5) % track.total_length
    x_near, y_near = track.point_at(s_near)
    kappa_r3_near = windowed_curvature_average(track, x_near, y_near)

    cases = [
        ("1. straight", 0.0, 0.0, 0.0, 1.0),
        ("2. tight arc away-from-join (R=3m)",
            0.0, 0.0, kappa_r3_away / 2, 1.0),
        ("3. tight arc near-join (R=3m blend)",
            0.0, 0.0, kappa_r3_near / 2, 1.0),
        ("4a. lateral offset +0.1 m", 0.1, 0.0, 0.0, 1.0),
        ("4b. lateral offset -0.1 m (mirror of 4a)", -0.1, 0.0, 0.0, 1.0),
        ("5a. heading offset +0.1 rad", 0.0, math.tan(0.1), 0.0, 1.0),
        ("5b. heading offset -0.1 rad (mirror of 5a)",
            0.0, -math.tan(0.1), 0.0, 1.0),
        ("6. combined offset + curvature",
            0.1, math.tan(0.1), kappa_r3_away / 2, 1.0),
    ]

    results = {}
    for name, c0, c1, c2, v in cases:
        results[name] = report_case(name, c0, c1, c2, v)

    print("\n" + "=" * 70)
    print("EXPLICIT SANITY CRITERIA")
    print("=" * 70)

    print("\n-- sign correctness: does e_lat/e_psi magnitude decrease "
          "over the horizon (steers toward reference, not away)? --")
    sign_cases = [
        "4a. lateral offset +0.1 m",
        "4b. lateral offset -0.1 m (mirror of 4a)",
        "5a. heading offset +0.1 rad",
        "5b. heading offset -0.1 rad (mirror of 5a)",
        "6. combined offset + curvature",
    ]
    for name in sign_cases:
        r = results[name]
        e0, eN = r["e_lat"][0], r["e_lat"][-1]
        p0, pN = r["e_psi"][0], r["e_psi"][-1]
        print(f"  {name}: |e_lat| {abs(e0):.5f} -> {abs(eN):.5f}  "
              f"|e_psi| {abs(p0):.5f} -> {abs(pN):.5f}")

    print("\n-- saturation on small offsets (miscalibrated-weight "
          "symptom) --")
    for name in ["4a. lateral offset +0.1 m", "5a. heading offset +0.1 rad"]:
        r = results[name]
        sat = bool(np.any(np.abs(r["U"][:, 0]) >= 0.999 * DELTA_DOT_MAX))
        pct = r['delta_first'] / DELTA_DOT_MAX * 100
        print(f"  {name}: ddelta saturates = {sat}, "
              f"first ddelta={r['delta_first']:+.6f} rad/s "
              f"({pct:.1f}% of the limit)")

    print("\n-- left/right symmetry (mirrored offset -> mirrored "
          "control) --")
    mirror_pairs = [
        ("4a. lateral offset +0.1 m",
            "4b. lateral offset -0.1 m (mirror of 4a)"),
        ("5a. heading offset +0.1 rad",
            "5b. heading offset -0.1 rad (mirror of 5a)"),
    ]
    for pair in mirror_pairs:
        a, b = results[pair[0]], results[pair[1]]
        s = a["delta_first"] + b["delta_first"]
        print(f"  {pair[0]} first ddelta = {a['delta_first']:+.6f}, "
              f"{pair[1]} first ddelta = {b['delta_first']:+.6f}, "
              f"sum = {s:+.8f} (0 = exact symmetry)")
        traj_diff = np.max(np.abs(a["delta_traj"] + b["delta_traj"]))
        print(f"    max|delta_traj_a + delta_traj_b| across horizon = "
              f"{traj_diff:.8f}")

    print("\n-- curvature steady-state feedforward: does terminal delta "
          "approach atan(L*kappa), not zero? (the actual test of "
          "ADR-19's formulation reversal) --")
    ff_cases = [
        ("2. tight arc away-from-join (R=3m)", kappa_r3_away),
        ("6. combined offset + curvature", kappa_r3_away),
    ]
    for name, kappa in ff_cases:
        r = results[name]
        delta_ff = math.atan(L * kappa)
        delta_end = r["delta_traj"][-1]
        diff = abs(delta_end - delta_ff)
        print(f"  {name}: kappa={kappa:.5f}  "
              f"atan(L*kappa)={delta_ff:.6f} rad  "
              f"delta_end={delta_end:.6f} rad  diff={diff:.6f} "
              f"({diff/delta_ff*100:.1f}% of feedforward value)")


if __name__ == "__main__":
    main()
