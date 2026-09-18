"""
How fast can the steering actuator actually move? control/mpc/params.py's
DELTA_DOT_MAX hard-constrains the MPC's steering-rate control input
(ddelta), and until this script was written that constant was
DELTA_DOT_MAX_PLACEHOLDER = 2.0 -- a guess, not a measurement (M3 kickoff
investigation item 6; ADR-19's own explicit flag). No physical actuator
exists to bench-test (ADR-1: this project has no hardware yet, sim only),
so this measures the next best thing: the simulated position actuator's
own closed-loop step response, given the real kp=15/kv=0.5 gains and
steer-joint damping=0.4 already in sim/models/car.xml -- not a separate
model of the servo, the actual one the sim (and therefore any sim-tested
controller) is subject to.

Method: load the full car vehicle, let it settle under gravity with the
wheels resting on the ground (steering torque has to work against whatever
ground-contact scrubbing resistance a real turn-in-place would also have,
not a frictionless bench rig), command the steering actuators to
-DELTA_MAX and let that settle too, then step the command to +DELTA_MAX
(car.xml's full +-0.6 rad range) and record steer_fl's joint position and
velocity every simulation timestep until it settles.

DELTA_DOT_MAX is reported as an AVERAGE rate over the full sweep, using a
2% settling-time criterion (a standard control-engineering convention):
    DELTA_DOT_MAX = (2 * DELTA_MAX) / t_settle_2pct
matching how the placeholder's own docstring already reasoned about this
constant ("a full sweep ... would take >=0.6s at this rate") -- i.e. the
same distance/time definition, now measured instead of guessed.

Important caveat, and it applies to the headline settling-based number
itself, not only to the peak rate: car.xml's position actuators have no
forcerange (actuator_forcelimited is False) -- the simulated servo has
unlimited torque/current. The PEAK instantaneous slew rate right after
the step (~18-19 rad/s) is an obvious modeling artifact of that, not a
usable capability estimate. But settling time in a PD-controlled 2nd-
order system depends on how much torque is available to arrest the
swing, not just on how fast it starts -- so the settling-based AVERAGE
this script reports as DELTA_DOT_MAX is inflated by the same unlimited-
torque idealization, just less obviously than the peak is. A
torque-limited real servo would settle more slowly than this, not just
peak lower. This script's output is therefore an UPPER BOUND from an
idealized actuator, not a conservative estimate of sustained rate -- do
not treat it as safe to use unmargined. Re-measure against a
torque-limited model or real hardware if/when it exists.

Usage:
    python3 tools/steering_rate_step_response.py
        -> prints the measurement report to stdout
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from control.mpc.params import DELTA_MAX  # noqa: E402

VEHICLE_XML = "sim/models/car.xml"

T_SETTLE_GRAVITY = 0.5  # s, let the car come to rest on the ground first
T_SETTLE_HOLD = 1.0     # s, let steering settle at -DELTA_MAX before the step
T_RECORD = 1.0          # s, recording window (>> observed settling time)
SETTLE_BAND = 0.02      # 2% settling-time criterion


def _cross_time(t_hist, q_hist, start, target, frac):
    """First time q_hist rises through start + frac*(target-start) (target
    > start assumed -- this script always steps low-to-high), linearly
    interpolated between the bracketing samples. None if never reached."""
    assert target > start
    thresh = start + frac * (target - start)
    idx = int(np.argmax(q_hist >= thresh))
    if q_hist[idx] < thresh:
        return None
    if idx == 0:
        return t_hist[0]
    t0, t1 = t_hist[idx - 1], t_hist[idx]
    q0, q1 = q_hist[idx - 1], q_hist[idx]
    return t0 + (thresh - q0) / (q1 - q0) * (t1 - t0)


def measure_delta_dot_max(settle_band: float = SETTLE_BAND) -> dict:
    """Runs the step-response test once and returns the measurement report
    as a dict. Deterministic (no randomness anywhere in this sim)."""
    model = mujoco.MjModel.from_xml_path(VEHICLE_XML)
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "steer_fl")
    jid_fr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "steer_fr")
    qadr, dadr = model.jnt_qposadr[jid], model.jnt_dofadr[jid]
    qadr_fr = model.jnt_qposadr[jid_fr]
    a_fl = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "a_steer_fl")
    a_fr = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "a_steer_fr")

    forcelimited = bool(model.actuator_forcelimited[a_fl])

    data.ctrl[:] = 0.0
    for _ in range(int(T_SETTLE_GRAVITY / dt)):
        mujoco.mj_step(model, data)

    data.ctrl[a_fl] = -DELTA_MAX
    data.ctrl[a_fr] = -DELTA_MAX
    for _ in range(int(T_SETTLE_HOLD / dt)):
        mujoco.mj_step(model, data)
    settled_at_start = float(data.qpos[qadr])

    data.ctrl[a_fl] = DELTA_MAX
    data.ctrl[a_fr] = DELTA_MAX
    t0 = data.time
    t_hist, q_hist, v_hist = [], [], []
    for _ in range(int(T_RECORD / dt)):
        mujoco.mj_step(model, data)
        t_hist.append(data.time - t0)
        q_hist.append(data.qpos[qadr])
        v_hist.append(data.qvel[dadr])
    t_hist = np.array(t_hist)
    q_hist = np.array(q_hist)
    v_hist = np.array(v_hist)

    t_settle = _cross_time(
        t_hist, q_hist, settled_at_start, DELTA_MAX, 1.0 - settle_band)
    span = DELTA_MAX - settled_at_start
    rate = span / t_settle

    peak_idx = int(np.argmax(np.abs(v_hist)))

    return {
        "forcelimited": forcelimited,
        "settled_at_start": settled_at_start,
        "final_qpos": float(q_hist[-1]),
        "fl_fr_symmetric": (
            abs(float(data.qpos[qadr]) - float(data.qpos[qadr_fr])) < 1e-6),
        "settle_band": settle_band,
        "t_settle": float(t_settle),
        "span_rad": float(span),
        "delta_dot_max_measured": float(rate),
        "peak_qvel": float(np.abs(v_hist).max()),
        "peak_qvel_time": float(t_hist[peak_idx]),
    }


def main():
    report = measure_delta_dot_max()
    if report['forcelimited']:
        torque_desc = 'torque-limited'
    else:
        torque_desc = 'UNLIMITED torque/current -- idealized PD servo'
    print(f"Actuator forcelimited: {report['forcelimited']} "
          f"({torque_desc})")
    print(f"Settled start position: {report['settled_at_start']:.6f} rad "
          f"(commanded -{DELTA_MAX}, fl==fr: {report['fl_fr_symmetric']})")
    print(f"Full sweep: {report['settled_at_start']:.4f} -> {DELTA_MAX} rad "
          f"(span {report['span_rad']:.4f} rad)")
    print(f"{report['settle_band']*100:.0f}% settling time: "
          f"{report['t_settle']:.4f} s")
    print(f"DELTA_DOT_MAX (measured, avg over full sweep) = "
          f"{report['delta_dot_max_measured']:.4f} rad/s -- UPPER BOUND "
          f"from an unlimited-torque actuator, not a conservative "
          f"estimate (forcelimited={report['forcelimited']}); a real "
          f"servo would settle slower.")
    print(f"Peak instantaneous |qvel| = {report['peak_qvel']:.4f} rad/s "
          f"at t={report['peak_qvel_time']:.4f} s "
          f"(NOT usable as a sustained-rate bound -- see module docstring)")
    print(f"Final qpos after {T_RECORD}s: {report['final_qpos']:.6f} "
          f"(target {DELTA_MAX})")


if __name__ == "__main__":
    main()
