#!/usr/bin/env python3
"""/lane_state -> acados lateral MPC -> /carsim/cmd.

Cote VM. Le seul noeud qui connait la formulation OCP : reutilise
control/mpc/ocp.py (build_ocp/solve_fixed_reference) tel quel, une seule
fois au demarrage -- pas de reconstruction/reformulation ici, cote ROS2
c'est uniquement le branchement dans le graphe.

    /lane_state   carsim_msgs/LaneState   (in, ~25 Hz, perception_node)
    /carsim/odom  nav_msgs/Odometry       (in, ~50 Hz, bridge_node -- v only)
    /carsim/cmd   geometry_msgs/Twist     (out -- linear.x=accel, angular.z=steer)

Design decisions this file makes (ADR-24):

1. **No joint-state feedback for `delta` (steering state).** Unlike
   control/mpc/closed_loop_sim.py, which reads MuJoCo's own qpos directly,
   there is no steering-angle topic anywhere in this ROS2 graph. `delta`'s
   state estimate is this node's OWN last-commanded value (updated from
   the solver's own x[1] prediction each successful solve, the same
   x[1]-not-integrated-u0 pattern closed_loop_sim.py uses), not a
   measurement. Open item, not solved here: this ignores real actuator
   lag (the position servo takes ~0.23s to settle, ADR-20) -- ADR-24
   accepts that mismatch for M3, flagged for the dedicated fallback/
   staleness ticket or hardware integration to revisit.

2. **Delay compensation (ADR-13's t_sim fix, used for the first time)
   needs a clock-offset calibration this project didn't have yet.**
   `header.stamp` is `t_sim` (sim-clock-relative, RclpyTime(seconds=t_sim)
   in bridge_node.py's sim_time_to_stamp) -- NOT wall-clock epoch time,
   and no node in this graph sets `use_sim_time`. Directly subtracting
   `self.get_clock().now()` (wall-clock epoch) from `header.stamp`
   (sim-clock, small numbers) -- which is exactly what
   perception_node.py's own age-telemetry used to do -- produces a huge,
   meaningless number, not a millisecond-scale age (confirmed while
   building this: that existing telemetry had been silently wrong since
   ADR-13 landed; ADR-25 later ported this same calibration back into
   perception_node.py instead of leaving it divergent). This node
   calibrates instead, via the shared `AgeCalibrator`
   (carsim_bridge/clock_utils.py, ADR-25): tracks the MINIMUM observed
   `wall_now_ns - stamp_ns` across all messages seen (a proxy for the
   zero-latency clock offset, since if the sim runs in realtime the only
   thing that should vary message-to-message is real pipeline latency,
   never negative). Age for any message is then
   `(wall_now_ns - stamp_ns) - running_minimum`, which is offset-agnostic
   and converges after the first few messages (message #1 always reads
   age=0 by construction -- an acceptable cold-start artifact, not a
   correctness bug for the messages after it).

3. **Solve-failure fallback: decay toward zero, not repeat-last.** A
   repeated/stuck failure repeating a stale nonzero command commits
   indefinitely to whatever delta happened to be in effect when solving
   broke -- for an unknown reason, on an unknown state. Decaying toward
   delta=0 (straight wheel) over a few cycles is a universally-safe
   default regardless of what triggered the failure, closer to how a
   real ADAS safe-state degrades.

5. **Perception-health fallback (ADR-27), orthogonal to solve status.**
   ADR-24 above was deliberately minimal -- solver-failure only. This is
   the sibling mechanism ADR-24 flagged as a separate, later ticket:
   `/lane_state` can be unhealthy (stale, or persistently low-confidence)
   even when the solver itself would happily solve against it -- garbage
   in, confidently-computed garbage out. Triggers on EITHER staleness
   (age, via the same AgeCalibrator already computed for delay
   compensation above -- not a second age measurement) exceeding
   `STALE_AGE_THRESHOLD_S`, or `msg.valid == False` for
   `N_INVALID_CONSECUTIVE_THRESHOLD` consecutive messages (not a single
   one -- a lone low-confidence frame is expected sensor noise, not a
   dropout; see those constants' own comments for the reasoning behind
   each number). On trigger: the SAME delta-decay `_decay_delta()` the
   solve-failure path uses (one shared function, not two "steer toward
   straight" implementations that could quietly drift apart) PLUS
   `v_target` forced to 0 for that cycle's longitudinal PI -- reusing the
   existing PI/gains, not a second braking mechanism, since ADR-24 point
   4 already owns v's only control path. No solve is attempted at all
   when unhealthy: the input is already known untrustworthy, so there is
   nothing a solve against it would add, only wasted compute and a
   "successful" `x[1]` prediction that would be worse than just decaying.
   **Recovery is deliberately NOT special-cased**: the next fresh+valid
   message flows straight back into the normal solve path, using
   whatever `self.delta_est` decayed to (and whatever `self.v_meas`
   naturally settled to, via the same PI's own braking effect during the
   unhealthy period) as `x0` -- a stopped, straight-wheeled vehicle is a
   perfectly ordinary MPC initial condition, not a special state needing
   a ramp back up. Stated explicitly here because it's a deliberate
   design choice, not an accident of not having written a ramp.

4. **Longitudinal command is a placeholder, explicitly not the thing
   under test.** ADR-19: this OCP does not control speed. Reuses
   closed_loop_sim.py's own simple PI-to-constant-v-target exactly
   (same gains, same non-authoritative status) rather than inventing a
   second ad hoc version. ADR-27 adds a second SETPOINT (0, on an
   unhealthy trigger) but not a second control mechanism -- still the
   same PI, same gains, same single implementation.
"""
import math
import os
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles

# Two levels up, not one: this needs `control.mpc.*` at the repo root, the
# same reach perception_inference.py needs for `perception.*` (both files
# sit at carsim_bridge/carsim_bridge/, the standard nested ament_python
# layout) -- NOT perception_node.py's one-level form, which only needs to
# resolve a sibling module inside this same package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from carsim_bridge.clock_utils import AgeCalibrator  # noqa: E402
from carsim_msgs.msg import LaneState  # noqa: E402
from control.mpc.ocp import build_ocp, solve_fixed_reference  # noqa: E402
from control.mpc.params import DELTA_MAX, L, TS  # noqa: E402

# Longitudinal PI, identical to control/mpc/closed_loop_sim.py's own
# non-authoritative placeholder -- see that module's docstring for why
# these specific gains, and ADR-19 for why this isn't the MPC's job.
V_TARGET_DEFAULT = 1.0
KP_V = 1.5
KI_V = 0.8
V_INTEGRAL_CLAMP = 1.0

# Solve-failure fallback (ADR-24, point 3 above): geometric decay toward
# delta=0 over a handful of cycles, not an instant zero (a discontinuous
# jump is itself a disturbance) and not an indefinite repeat. Shared with
# the perception-health fallback (ADR-27, point 5 above) via
# _decay_delta() -- one implementation, two triggers.
FALLBACK_DECAY = 0.7

# Perception-health fallback (ADR-27). Both starting values, not
# measured: the dedicated dropout-scenario test task tunes these against
# actual observed behavior, not this task.
#
# STALE_AGE_THRESHOLD_S: /lane_state's nominal period is ~40ms (~25 Hz,
# this node's own module docstring). 150ms is ~3.75x that -- generous
# enough that ordinary jitter (a slow inference cycle, a scheduling
# hiccup) doesn't false-trigger, tight enough that a real dropout is
# caught within a handful of missed cycles, not dozens. Picked at the
# LOWER end of the task's stated 150-200ms range: when a placeholder is
# still pending real measurement, this project's own convention (e.g.
# DELTA_DOT_MAX_PLACEHOLDER's original reasoning, ADR-19) is to fail
# toward triggering sooner, not later, until it's actually tuned.
STALE_AGE_THRESHOLD_S = 0.15

# N_INVALID_CONSECUTIVE_THRESHOLD: a single low-confidence frame is
# expected sensor noise (perception/README.md's own confidence numbers:
# even the trained model's invalid-recall isn't 100% on every frame) --
# reacting to one frame would false-trigger on normal operation, not
# just real dropouts. 3 consecutive is enough of a run to distinguish
# "the camera saw something ambiguous for an instant" from "perception
# has actually lost the lane," while still being caught within ~120ms
# (3 cycles at the ~40ms nominal period) of a real dropout starting --
# comparable to, not dramatically slower than, the staleness threshold
# above, so neither trigger dominates the other's response time.
N_INVALID_CONSECUTIVE_THRESHOLD = 3

# Sub-steps for the delay-compensation forward-propagation (ADR-24, point
# 2) -- Euler, matching the OCP's own DARE linearization's discretization
# choice (control/mpc/ocp.py's _terminal_dare_matrix), not the ERK4 the
# solver's actual stage dynamics use: this is a short, small-angle,
# constant-(v,delta) extrapolation over a known short interval, not an
# optimization, so a few Euler sub-steps are enough without pulling in
# casadi for a numeric-only forward simulation.
PROPAGATION_SUBSTEPS = 4


def propagate_state(age_s: float, v: float, delta: float):
    """Body-frame [X, Y, psi, delta] at the image-capture instant is
    [0, 0, 0, delta] by construction (that's the frame (c0,c1,c2) is
    expressed in) -- this integrates the SAME kinematic bicycle dynamics
    control/mpc/model.py's OCP model uses forward by age_s seconds of
    constant (v, delta) (ddelta assumed 0 over the delay window: the
    control applied during it isn't known here), landing at the vehicle's
    estimated CURRENT pose in that same, still-valid reference frame --
    this is what makes it a delay-compensated x0, not a re-zeroed one."""
    X = Y = psi = 0.0
    if age_s <= 0.0:
        return np.array([X, Y, psi, delta])
    dt = age_s / PROPAGATION_SUBSTEPS
    for _ in range(PROPAGATION_SUBSTEPS):
        X += v * math.cos(psi) * dt
        Y += v * math.sin(psi) * dt
        psi += (v / L) * math.tan(delta) * dt
    return np.array([X, Y, psi, delta])


class MpcNode(Node):

    def __init__(self, **kwargs):
        # **kwargs forwarded to rclpy.node.Node, same pattern as
        # PerceptionNode -- lets tests construct this node with
        # parameter_overrides without a live graph.
        super().__init__('mpc_node', **kwargs)

        self.declare_parameter('v_target', V_TARGET_DEFAULT)
        self.v_target = self.get_parameter('v_target').value

        ocp = build_ocp(c0=0.0, c1=0.0, c2=0.0, v=self.v_target)
        from acados_template import AcadosOcpSolver
        self.solver = AcadosOcpSolver(ocp, json_file='/tmp/mpc_node_acados_ocp.json')

        self.pub_cmd = self.create_publisher(Twist, 'carsim/cmd', 10)
        self.create_subscription(LaneState, 'lane_state', self.on_lane_state, 10)
        self.create_subscription(Odometry, 'carsim/odom', self.on_odom,
                                  QoSPresetProfiles.SENSOR_DATA.value)

        self.v_meas = 0.0
        self.have_odom = False
        self.delta_est = 0.0     # last-commanded delta, this node's own state estimate
        self.v_integral = 0.0
        self.n_fail_consecutive = 0
        self.n_invalid_consecutive = 0  # ADR-27 perception-health trigger

        # Clock-offset calibration (ADR-24, point 2; ADR-25 extracted this
        # into a shared helper also used by perception_node.py) -- running
        # minimum of (wall_now - header.stamp), a proxy for the sim<->wall
        # clock's constant offset, refined as more messages arrive.
        self._age_calibrator = AgeCalibrator()

        self.n_solves = 0
        self.n_solve_failures = 0
        self._solve_time_samples = []
        self.create_timer(2.0, self.report)

        self.get_logger().info('mpc_node actif')

    def on_odom(self, msg: Odometry):
        self.v_meas = float(msg.twist.twist.linear.x)
        self.have_odom = True

    def _age_s(self, stamp) -> float:
        now_ns = self.get_clock().now().nanoseconds
        return self._age_calibrator.age_s(now_ns, stamp)

    def _decay_delta(self) -> float:
        """Shared by both fallback triggers (ADR-24 solve-failure, ADR-27
        perception-health) -- same decay, same constant, one
        implementation of "steer toward straight" rather than two that
        could quietly drift apart."""
        self.delta_est *= FALLBACK_DECAY
        return self.delta_est

    def on_lane_state(self, msg: LaneState):
        t0 = time.perf_counter()
        age_s = self._age_s(msg.header.stamp)

        if msg.valid:
            self.n_invalid_consecutive = 0
        else:
            self.n_invalid_consecutive += 1

        stale = age_s > STALE_AGE_THRESHOLD_S
        unhealthy = stale or self.n_invalid_consecutive >= N_INVALID_CONSECUTIVE_THRESHOLD

        if unhealthy:
            # No solve attempted at all: the input is already known
            # untrustworthy (ADR-27) -- a "successful" solve against a
            # stale or persistently-invalid reference would be confident
            # garbage, not a better answer than decaying. v_target=0 for
            # THIS cycle only (not sticky state -- recovery is the next
            # message simply not being unhealthy, see ADR-27 point 5).
            reason = 'stale' if stale else 'invalid'
            self.get_logger().warning(
                f'/lane_state unhealthy ({reason}, age={age_s*1000:.1f} ms, '
                f'invalid_consecutive={self.n_invalid_consecutive}) -- '
                f'safe state: v_target=0, decaying delta toward 0')
            delta_cmd = self._decay_delta()
            v_target_effective = 0.0
        else:
            c0 = float(msg.lateral_error)
            c1 = math.tan(float(msg.heading_error))
            c2 = float(msg.curvature) / 2.0
            v = max(self.v_meas, 0.05) if self.have_odom else self.v_target

            x0 = propagate_state(age_s, v, self.delta_est)
            status = solve_fixed_reference(self.solver, c0, c1, c2, v, x0=x0)
            solve_dt = time.perf_counter() - t0

            if status == 0:
                self.n_fail_consecutive = 0
                delta_cmd = float(self.solver.get(1, "x")[3])
                delta_cmd = max(-DELTA_MAX, min(DELTA_MAX, delta_cmd))
                self.delta_est = delta_cmd
            else:
                self.n_solve_failures += 1
                self.n_fail_consecutive += 1
                self.get_logger().warning(
                    f'acados solve status={status} (failure #{self.n_fail_consecutive} '
                    f'consecutive) -- decaying delta toward 0')
                delta_cmd = self._decay_delta()

            v_target_effective = self.v_target
            self.n_solves += 1
            self._solve_time_samples.append(solve_dt)

        v_err = v_target_effective - self.v_meas
        self.v_integral = float(np.clip(self.v_integral + v_err * TS,
                                         -V_INTEGRAL_CLAMP, V_INTEGRAL_CLAMP))
        accel_cmd = float(np.clip(KP_V * v_err + KI_V * self.v_integral, -1.0, 1.0))

        cmd = Twist()
        cmd.angular.z = delta_cmd
        cmd.linear.x = accel_cmd
        self.pub_cmd.publish(cmd)

    def report(self):
        if not self._solve_time_samples:
            return
        a = np.asarray(self._solve_time_samples) * 1000.0
        self.get_logger().info(
            f'{self.n_solves} solves ({self.n_solve_failures} failures) | '
            f'solve+callback (ms) mean={a.mean():.3f} p95={np.percentile(a, 95):.3f} '
            f'max={a.max():.3f}')
        self._solve_time_samples.clear()


def main(args=None):
    rclpy.init(args=args)
    node = MpcNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
