"""
Closed-loop MPC simulation with perfect (analytic) simulator-truth state --
no perception CNN, no ROS2, no carsim_bridge, no /lane_state message at all.

Purpose: validate closed-loop behavior (convergence, oscillation, stability
across curvature transitions) in isolation from perception, so any bad
behavior found here can only be attributed to the controller/formulation,
not to noisy or biased perception -- a clean separation from the M1
mirror-probe finding that perception degrades most near curvature joins
(tests/test_mirror_augmentation.py and related M1 work): this script uses
TRUE curvature there, so any degradation it shows is a controller-only
effect.

At each Ts=40ms control step: read the vehicle's true pose from MuJoCo,
compute (c0, c1, c2) from Track's exact analytic geometry at that pose --
lateral_error/heading_error via compute_lane_state's point-wise Frenet
projection (perception/dataset/geometry.py, unchanged since M1), curvature
via windowed_curvature_average (perception/dataset/windowed_relabel.py,
ADR-18's validated method, not compute_lane_state's own point-wise
curvature -- the OCP's single quadratic reference is meant to represent
the path over the whole ~1.2 m preview horizon, not just the vehicle's
instantaneous position, and windowed_curvature_average is what ADR-18
established as the correct representation of that near a primitive join).
Solve the acados OCP with these as parameters, apply the resulting
steering command, step MuJoCo, repeat.

Run on the VM (needs BOTH acados_template -- this project's established
VM-only constraint -- AND mujoco, which normally runs on the Mac side of
the ZMQ bridge split, ADR-1/ADR-2; mujoco==3.12.0, matching
requirements-sim.txt's pinned version, was pip-installed into
~/venvs/ackerman_ros2 for this script specifically -- headless physics
only, no rendering, so it doesn't conflict with that architectural split,
it just needs to coexist with acados in one process here):

    source ~/venvs/ackerman_ros2/bin/activate
    export ACADOS_SOURCE_DIR=$HOME/acados
    export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH
    python3 control/mpc/closed_loop_sim.py [--laps 3] [--v-target 1.0]

Diagnostic only, per M3's task instructions: reports findings (error time
series, solve status, join-type comparison, solve timing), does not tune
weights or touch ocp.py based on what it finds.
"""
import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from acados_template import AcadosOcpSolver

from control.mpc.ocp import build_ocp, solve_fixed_reference
from control.mpc.params import DELTA_MAX, N_HORIZON, TS
from perception.dataset.geometry import compute_lane_state
from perception.dataset.track_definitions import REFERENCE_TRACK
from perception.dataset.windowed_relabel import windowed_curvature_average

VEHICLE_XML = os.path.join(os.path.dirname(__file__), "..", "..", "sim", "models", "car.xml")

# Inlined from sim/sim_server.py rather than imported: that module imports
# a bare `protocol` (relies on being run from sim/'s own directory on
# sys.path, its ZMQ server's own launch convention) and pulls in pyzmq for
# two pure functions -- not worth fighting either just for these. Keep in
# sync with sim_server.py's own WHEELBASE/TRACK/MAX_TORQUE/ackermann/
# yaw_from_quat if those ever change.
SIM_WHEELBASE = 0.26  # m, sim/models/car.xml, matches control/mpc/params.py's L
SIM_TRACK = 0.21      # m, front axle track width
MAX_TORQUE = 2.0      # N.m, sim_server.py's drive-motor cap


def ackermann(delta):
    """Bicycle-model steering angle -> (left, right) front wheel angles."""
    if abs(delta) < 1e-4:
        return delta, delta
    R = SIM_WHEELBASE / math.tan(delta)
    return (math.atan(SIM_WHEELBASE / (R - SIM_TRACK / 2)),
            math.atan(SIM_WHEELBASE / (R + SIM_TRACK / 2)))


def yaw_from_quat(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

T_SETTLE_GRAVITY = 0.5   # s, let the car come to rest on the ground first
V_TARGET_DEFAULT = 1.0   # m/s, matches params.V_NOMINAL_FOR_DARE / the sanity-check cases
N_LAPS_DEFAULT = 3.0

# Simple PI forward-speed controller -- NOT the thing under test (ADR-19:
# "This OCP does not control speed (a separate longitudinal loop does)").
# Tuned empirically against this vehicle/track just to hold roughly
# constant v so the lateral controller has a meaningful, consistent
# reference speed to plan against; not otherwise justified or reused
# anywhere else in the repo.
KP_V = 1.5
KI_V = 0.8
V_INTEGRAL_CLAMP = 1.0

# Curvature-jump magnitudes at this track's 8 joins (track_definitions.py:
# RADIUS_1=3.0 "tight", RADIUS_2=5.0 "wide" -- every join is straight<->arc,
# never arc<->arc, so there are exactly two jump sizes, each appearing 4
# times per lap [2 R1 joins + 2 R2 joins, x2 for the two identical halves]).
KAPPA_R1 = 1.0 / 3.0
KAPPA_R2 = 1.0 / 5.0
JOIN_CLASS_TOL = 1e-6


def classify_joins():
    """Arc-length position (mod total_length) and |delta kappa| of each of
    the 8 primitive joins, classified as 'large' (straight<->R1, tight arc)
    or 'small' (straight<->R2, wide arc) -- both are jumps to/from a
    straight (kappa=0), so |delta kappa| is just the arc's own kappa."""
    joins = []
    n = len(REFERENCE_TRACK.primitives)
    for i in range(n):
        s_join = REFERENCE_TRACK.starts[i]
        prev_kappa = REFERENCE_TRACK.primitives[i - 1].curvature_at(
            REFERENCE_TRACK.primitives[i - 1].length - 1e-6)
        next_kappa = REFERENCE_TRACK.primitives[i].curvature_at(0.0)
        jump = abs(next_kappa - prev_kappa)
        if jump < JOIN_CLASS_TOL:
            continue  # not actually a curvature discontinuity (shouldn't happen on this track)
        cls = "large" if abs(jump - KAPPA_R1) < 1e-3 else "small"
        joins.append((s_join, jump, cls))
    return joins


def nearest_join_distance(s, joins, total_length):
    """Signed-arc-length-agnostic distance (always >= 0) from s to the
    nearest join, accounting for wraparound."""
    best = total_length
    for s_join, _, _ in joins:
        d = abs(s - s_join)
        d = min(d, total_length - d)
        best = min(best, d)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--laps", type=float, default=N_LAPS_DEFAULT)
    ap.add_argument("--v-target", type=float, default=V_TARGET_DEFAULT)
    ap.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--tag", default="closed_loop")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(VEHICLE_XML)
    data = mujoco.MjData(model)
    dt = model.opt.timestep
    n_sub = max(1, round(TS / dt))

    a_fl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "a_steer_fl")
    a_fr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "a_steer_fr")
    a_drl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "a_drive_rl")
    a_drr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "a_drive_rr")
    j_fl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "steer_fl")
    j_fr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "steer_fr")
    q_fl, q_fr = model.jnt_qposadr[j_fl], model.jnt_qposadr[j_fr]

    data.ctrl[:] = 0.0
    for _ in range(int(T_SETTLE_GRAVITY / dt)):
        mujoco.mj_step(model, data)

    ocp = build_ocp(c0=0.0, c1=0.0, c2=0.0, v=args.v_target)
    solver = AcadosOcpSolver(ocp, json_file="/tmp/closed_loop_sim_acados_ocp.json")

    joins = classify_joins()
    total_length = REFERENCE_TRACK.total_length
    print(f"track total_length={total_length:.4f} m, {len(joins)} joins "
          f"({sum(1 for j in joins if j[2] == 'large')} large / "
          f"{sum(1 for j in joins if j[2] == 'small')} small)")

    log = {k: [] for k in ["t", "x", "y", "yaw", "s", "lateral_error",
                            "heading_error", "kappa", "v", "delta", "ddelta",
                            "solve_status", "solve_time", "join_dist"]}

    v_integral = 0.0
    s_prev = None
    total_unwrapped_s = 0.0
    lap_target_s = args.laps * total_length

    t = 0.0
    step_i = 0
    max_steps = int(lap_target_s / max(args.v_target, 0.1) / TS * 2.0)  # generous safety cap

    status_events = []

    while total_unwrapped_s < lap_target_s and step_i < max_steps:
        pos = data.body("chassis").xpos
        quat = data.body("chassis").xquat
        x, y = float(pos[0]), float(pos[1])
        yaw = yaw_from_quat(quat)

        lane_state = compute_lane_state(REFERENCE_TRACK, x, y, yaw)
        kappa = windowed_curvature_average(REFERENCE_TRACK, x, y)
        c0 = lane_state.lateral_error
        c1 = math.tan(lane_state.heading_error)
        c2 = kappa / 2.0

        s = lane_state.s
        if s_prev is not None:
            ds = s - s_prev
            if ds < -total_length * 0.5:
                ds += total_length
            elif ds > total_length * 0.5:
                ds -= total_length
            total_unwrapped_s += ds
        s_prev = s

        v_meas = float(data.sensor("s_vel").data[0])
        delta_meas = 0.5 * (float(data.qpos[q_fl]) + float(data.qpos[q_fr]))

        x0 = np.array([0.0, 0.0, 0.0, delta_meas])
        t0_solve = time.perf_counter()
        status = solve_fixed_reference(solver, c0, c1, c2, max(v_meas, 0.05), x0=x0)
        solve_dt = time.perf_counter() - t0_solve

        if status != 0:
            status_events.append({
                "t": t, "status": status, "x": x, "y": y, "yaw": yaw,
                "c0": c0, "c1": c1, "c2": c2, "kappa": kappa, "v": v_meas,
            })

        u0 = float(solver.get(0, "u")[0])
        delta_cmd = float(solver.get(1, "x")[3])
        delta_cmd = max(-DELTA_MAX, min(DELTA_MAX, delta_cmd))

        v_err = args.v_target - v_meas
        v_integral = float(np.clip(v_integral + v_err * TS, -V_INTEGRAL_CLAMP, V_INTEGRAL_CLAMP))
        accel_cmd = float(np.clip(KP_V * v_err + KI_V * v_integral, -1.0, 1.0))
        torque = accel_cmd * MAX_TORQUE

        dl, dr = ackermann(delta_cmd)
        data.ctrl[a_fl] = dl
        data.ctrl[a_fr] = dr
        data.ctrl[a_drl] = torque
        data.ctrl[a_drr] = torque

        for _ in range(n_sub):
            mujoco.mj_step(model, data)

        log["t"].append(t)
        log["x"].append(x)
        log["y"].append(y)
        log["yaw"].append(yaw)
        log["s"].append(s)
        log["lateral_error"].append(c0)
        log["heading_error"].append(lane_state.heading_error)
        log["kappa"].append(kappa)
        log["v"].append(v_meas)
        log["delta"].append(delta_meas)
        log["ddelta"].append(u0)
        log["solve_status"].append(status)
        log["solve_time"].append(solve_dt)
        log["join_dist"].append(nearest_join_distance(s, joins, total_length))

        t += TS
        step_i += 1

    for k in log:
        log[k] = np.array(log[k])

    n_steps = len(log["t"])
    laps_completed = total_unwrapped_s / total_length
    print(f"\nran {n_steps} steps ({log['t'][-1]:.2f} s sim time), "
          f"{laps_completed:.2f} laps completed "
          f"({'reached max_steps safety cap -- did NOT complete requested laps' if step_i >= max_steps else 'target laps reached'})")

    # --- solve status -----------------------------------------------------
    n_bad = len(status_events)
    print(f"\nsolve status != 0: {n_bad} / {n_steps} steps "
          f"({100.0 * n_bad / n_steps:.3f}%)")
    for e in status_events[:20]:
        print(f"  t={e['t']:.3f}s status={e['status']} pose=({e['x']:+.3f},{e['y']:+.3f},yaw={e['yaw']:+.3f}) "
              f"c0={e['c0']:+.4f} c1={e['c1']:+.4f} c2={e['c2']:+.4f} kappa={e['kappa']:+.4f} v={e['v']:.3f}")
    if n_bad > 20:
        print(f"  ... and {n_bad - 20} more")

    # --- solve timing -------------------------------------------------------
    st = log["solve_time"] * 1000.0  # ms
    print(f"\nsolve time (ms): mean={st.mean():.3f} median={np.median(st):.3f} "
          f"p95={np.percentile(st, 95):.3f} p99={np.percentile(st, 99):.3f} max={st.max():.3f}")

    # --- error magnitude stats, overall and by proximity to a join --------
    abs_lat = np.abs(log["lateral_error"])
    abs_psi = np.abs(log["heading_error"])
    print(f"\n|lateral_error| (m): mean={abs_lat.mean():.4f} max={abs_lat.max():.4f}")
    print(f"|heading_error| (rad): mean={abs_psi.mean():.4f} max={abs_psi.max():.4f}")

    # Exclude the pre-motion startup window (T_SETTLE_GRAVITY already
    # elapsed before t=0, but the vehicle needs a moment under throttle to
    # actually start moving) from join-crossing statistics: the track's
    # s=0 join (wraparound, last primitive -> first) is a genuine "small"
    # join, and the vehicle starts sitting exactly there with x=y=psi=0 --
    # a trivial zero-error data point, not a real crossing, that would
    # otherwise bias that join's stats down. Kept in the raw plotted time
    # series (that should show the whole run honestly), excluded only here.
    warm = log["t"] >= 1.0

    near_join = (log["join_dist"] < 1.0) & warm  # within 1m of a join, roughly one horizon-length
    far_join = ~(log["join_dist"] < 1.0) & warm
    if near_join.any() and far_join.any():
        print(f"\nnear a join (<1.0 m, {near_join.sum()} steps): "
              f"|lateral_error| mean={abs_lat[near_join].mean():.4f} max={abs_lat[near_join].max():.4f}  "
              f"|heading_error| mean={abs_psi[near_join].mean():.4f} max={abs_psi[near_join].max():.4f}")
        print(f"away from a join (>=1.0 m, {far_join.sum()} steps): "
              f"|lateral_error| mean={abs_lat[far_join].mean():.4f} max={abs_lat[far_join].max():.4f}  "
              f"|heading_error| mean={abs_psi[far_join].mean():.4f} max={abs_psi[far_join].max():.4f}")

    # --- large vs small curvature-jump joins -------------------------------
    def contiguous_runs(mask):
        """Start/end indices of each contiguous True run in a boolean
        array -- one run per actual approach to a join, not one per lap
        (a fixed-modulo join distance repeats every lap, so a naive
        threshold mask has one run per crossing already; this just
        segments them so each crossing's peak is counted once, not
        conflated with every other crossing's)."""
        runs = []
        in_run = False
        start = 0
        for i, v in enumerate(mask):
            if v and not in_run:
                start, in_run = i, True
            elif not v and in_run:
                runs.append((start, i))
                in_run = False
        if in_run:
            runs.append((start, len(mask)))
        return runs

    for cls in ["large", "small"]:
        cls_joins_s = [sj for sj, _, c in joins if c == cls]
        peak_lat, peak_psi = [], []
        for s_join in cls_joins_s:
            d = np.abs(((log["s"] - s_join + total_length / 2) % total_length) - total_length / 2)
            mask = (d < 1.0) & warm
            for start, end in contiguous_runs(mask):
                peak_lat.append(abs_lat[start:end].max())
                peak_psi.append(abs_psi[start:end].max())
        if peak_lat:
            print(f"\n'{cls}' curvature-jump joins ({len(cls_joins_s)} per lap, "
                  f"{len(peak_lat)} crossings observed): "
                  f"peak |lateral_error| mean={np.mean(peak_lat):.4f} max={np.max(peak_lat):.4f}  "
                  f"peak |heading_error| mean={np.mean(peak_psi):.4f} max={np.max(peak_psi):.4f}")

    # --- plot ---------------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(log["t"], log["lateral_error"], lw=0.8)
    axes[0].set_ylabel("lateral_error [m]")
    axes[0].axhline(0, color="k", lw=0.5)

    axes[1].plot(log["t"], log["heading_error"], lw=0.8, color="tab:orange")
    axes[1].set_ylabel("heading_error [rad]")
    axes[1].axhline(0, color="k", lw=0.5)

    axes[2].plot(log["t"], log["kappa"], lw=0.8, color="tab:green")
    axes[2].set_ylabel("kappa [1/m]")

    axes[3].plot(log["t"], log["solve_time"] * 1000.0, lw=0.5, color="tab:red")
    axes[3].set_ylabel("solve time [ms]")
    axes[3].set_xlabel("t [s]")

    # mark join crossings across all laps within the plotted time range
    for ax in axes[:3]:
        for s_join, _, cls in joins:
            color = "tab:red" if cls == "large" else "tab:purple"
            for lap in range(int(np.ceil(args.laps)) + 1):
                s_target = s_join + lap * total_length
                t_target = s_target / max(args.v_target, 1e-6)
                if t_target <= log["t"][-1]:
                    ax.axvline(t_target, color=color, lw=0.4, alpha=0.4, ls="--")

    for e in status_events:
        for ax in axes[:3]:
            ax.axvline(e["t"], color="black", lw=1.0, alpha=0.7)

    fig.suptitle(f"Closed-loop MPC, perfect simulator-truth state "
                 f"({laps_completed:.2f} laps, v_target={args.v_target} m/s)")
    fig.tight_layout()
    out_path = os.path.join(args.out_dir, f"{args.tag}_error_timeseries.png")
    fig.savefig(out_path, dpi=130)
    print(f"\nplot saved to {out_path}")


if __name__ == "__main__":
    main()
