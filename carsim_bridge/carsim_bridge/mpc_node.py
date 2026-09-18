#!/usr/bin/env python3
"""/lane_state -> acados lateral MPC -> /carsim/cmd.

Runs on the VM. The only node that knows the OCP formulation: reuses
control/mpc/ocp.py (build_ocp/solve_fixed_reference) as-is, built once
at startup -- no reconstruction/reformulation here, this file is only
the ROS2 wiring.

    /lane_state   carsim_msgs/LaneState   (in, ~25 Hz, perception_node)
    /carsim/odom  nav_msgs/Odometry       (in, ~50 Hz, bridge_node -- v only)
    /carsim/cmd   geometry_msgs/Twist     (out -- accel/steer)

Design decisions this file makes (docs/decisions.md ADR-24, ADR-27 for
the perception-health fallback):

1. No joint-state feedback for `delta`: this graph has no steering-angle
   topic, so `delta`'s state estimate is this node's own last-commanded
   value (the solver's x[1] prediction), not a measurement. Ignores real
   actuator lag (ADR-20); flagged, not solved here.

2. Delay compensation needs the shared `AgeCalibrator`
   (carsim_bridge/clock_utils.py, ADR-25): `header.stamp` is sim-clock
   (ADR-13), not wall-clock, so a direct `now() - stamp` subtraction is
   meaningless. `propagate_state()` below forward-integrates the
   compensated x0 by the calibrated age.

3. Solve-failure fallback decays delta toward zero rather than
   repeating the last command, so a stuck failure doesn't commit
   indefinitely to whatever delta was in effect when it broke.

4. Perception-health fallback (ADR-27), orthogonal to solve status:
   triggers on staleness or on N consecutive invalid messages (see
   STALE_AGE_THRESHOLD_S / N_INVALID_CONSECUTIVE_THRESHOLD below), and
   shares `_decay_delta()` with the solve-failure path rather than a
   second implementation. Recovery is deliberately not special-cased:
   the next fresh+valid message flows straight back into the normal
   solve path (ADR-27).

5. The longitudinal command is a placeholder, not the thing under test
   (ADR-19): reuses control/mpc/closed_loop_sim.py's PI-to-constant-v
   exactly, including for the fallback's v_target=0 setpoint.
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

# Two levels up: needs `control.mpc.*` at the repo root (same reach
# perception_inference.py needs for `perception.*`).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from carsim_bridge.clock_utils import AgeCalibrator  # noqa: E402
from carsim_msgs.msg import LaneState  # noqa: E402
from control.mpc.ocp import build_ocp, solve_fixed_reference  # noqa: E402
from control.mpc.params import DELTA_MAX, L, TS  # noqa: E402

# Longitudinal PI, identical to control/mpc/closed_loop_sim.py's own
# non-authoritative placeholder (ADR-19).
V_TARGET_DEFAULT = 1.0
KP_V = 1.5
KI_V = 0.8
V_INTEGRAL_CLAMP = 1.0

# Geometric decay toward delta=0, shared by both fallback triggers
# (ADR-24, ADR-27) via _decay_delta().
FALLBACK_DECAY = 0.7

# ADR-27. Both starting values, not measured -- the dedicated
# dropout-scenario test tunes these against observed behavior.
STALE_AGE_THRESHOLD_S = 0.15
N_INVALID_CONSECUTIVE_THRESHOLD = 3

# Euler sub-steps for the delay-compensation forward-propagation
# (ADR-24 point 2) -- a short constant-(v,delta) extrapolation, not an
# optimization, so a few Euler steps are enough without pulling casadi
# into a numeric-only forward simulation.
PROPAGATION_SUBSTEPS = 4


def propagate_state(age_s: float, v: float, delta: float):
    """Body-frame [X, Y, psi, delta] at the image-capture instant is
    [0, 0, 0, delta] by construction -- forward-integrates the same
    kinematic bicycle dynamics control/mpc/model.py's OCP model uses,
    by age_s seconds of constant (v, delta), landing at the vehicle's
    estimated current pose in that same reference frame (a
    delay-compensated x0, not a re-zeroed one)."""
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
        # **kwargs forwarded to Node -- lets tests construct this node
        # with parameter_overrides, without a live graph.
        super().__init__('mpc_node', **kwargs)

        self.declare_parameter('v_target', V_TARGET_DEFAULT)
        self.v_target = self.get_parameter('v_target').value

        ocp = build_ocp(c0=0.0, c1=0.0, c2=0.0, v=self.v_target)
        from acados_template import AcadosOcpSolver
        self.solver = AcadosOcpSolver(
            ocp, json_file='/tmp/mpc_node_acados_ocp.json')

        self.pub_cmd = self.create_publisher(Twist, 'carsim/cmd', 10)
        self.create_subscription(
            LaneState, 'lane_state', self.on_lane_state, 10)
        self.create_subscription(
            Odometry, 'carsim/odom', self.on_odom,
            QoSPresetProfiles.SENSOR_DATA.value)

        self.v_meas = 0.0
        self.have_odom = False
        self.delta_est = 0.0  # last-commanded delta, this node's own estimate
        self.v_integral = 0.0
        self.n_fail_consecutive = 0
        self.n_invalid_consecutive = 0  # ADR-27 perception-health trigger
        self._age_calibrator = AgeCalibrator()  # ADR-25

        self.n_solves = 0
        self.n_solve_failures = 0
        self._solve_time_samples = []
        self.create_timer(2.0, self.report)

        self.get_logger().info('mpc_node active')

    def on_odom(self, msg: Odometry):
        self.v_meas = float(msg.twist.twist.linear.x)
        self.have_odom = True

    def _age_s(self, stamp) -> float:
        now_ns = self.get_clock().now().nanoseconds
        return self._age_calibrator.age_s(now_ns, stamp)

    def _decay_delta(self) -> float:
        """Shared by both fallback triggers (ADR-24, ADR-27) -- one
        "steer toward straight" implementation, not two that could
        quietly drift apart."""
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
        unhealthy = (
            stale
            or self.n_invalid_consecutive >= N_INVALID_CONSECUTIVE_THRESHOLD)

        if unhealthy:
            # No solve attempted: the input is already known
            # untrustworthy (ADR-27). Not sticky state -- recovery is
            # just the next message not being unhealthy.
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
            status = solve_fixed_reference(
                self.solver, c0, c1, c2, v, x0=x0)
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
                    f'acados solve status={status} '
                    f'(failure #{self.n_fail_consecutive} consecutive) -- '
                    f'decaying delta toward 0')
                delta_cmd = self._decay_delta()

            v_target_effective = self.v_target
            self.n_solves += 1
            self._solve_time_samples.append(solve_dt)

        v_err = v_target_effective - self.v_meas
        self.v_integral = float(np.clip(
            self.v_integral + v_err * TS,
            -V_INTEGRAL_CLAMP, V_INTEGRAL_CLAMP))
        accel_cmd = float(np.clip(
            KP_V * v_err + KI_V * self.v_integral, -1.0, 1.0))

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
            f'solve+callback (ms) mean={a.mean():.3f} '
            f'p95={np.percentile(a, 95):.3f} max={a.max():.3f}')
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
