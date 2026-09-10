"""
Shared constants for the lateral MPC (ADR-19).

Physical constants are imported from where they're already established in
this project rather than restated, so this can't silently drift from the
values perception/the sim actually use (the same failure mode ADR-8 fixed
for the sampling envelope, and targets.py's own docstring states for
itself): L from sim/models/car.xml (cross-checked against
sim_server.py's WHEELBASE), LANE_HALF_WIDTH/POS_HEADING_RANGE/RADIUS_1
from perception/dataset/track_definitions.py and generate_dataset.py.
"""

from perception.dataset.track_definitions import (
    LANE_HALF_WIDTH, RADIUS_1, RADIUS_2,
)
from perception.dataset.generate_dataset import POS_HEADING_RANGE

# --- Vehicle ---------------------------------------------------------------
# m, wheelbase. sim/models/car.xml: front axle (steer_fl/fr) at local x=0.13,
# rear axle (wheel_rl/rr) at x=-0.13 -> 0.13-(-0.13)=0.26. Cross-checked
# against sim_server.py's hardcoded WHEELBASE=0.26 (M3 kickoff investigation
# item 6) -- both agree, not assumed.
L = 0.26

# rad, hard steering limit. car.xml: <default class="steer"><joint
# range="-0.6 0.6"/> and the position actuators' own ctrlrange="-0.6 0.6"
# (both agree independently). +-0.6 rad = +-34.4 deg.
DELTA_MAX = 0.6

# rad/s, steering RATE limit -- MEASURED (ADR-20), not the ADR-19
# placeholder this constant used to be. No hardware exists to bench-test
# (ADR-1: sim only), so this is the sim's own position actuator's closed-
# loop step response: car.xml's real kp=15/kv=0.5 gains and steer-joint
# damping=0.4, full vehicle (wheels resting on the ground, so steering has
# to work against ground-contact scrubbing, not a frictionless bench rig),
# stepped the full -DELTA_MAX -> +DELTA_MAX range and timed to a 2%
# settling band: DELTA_DOT_MAX = 2*DELTA_MAX / t_settle_2pct = 1.2 / 0.2308
# = 5.1996 rad/s, rounded to 5.2. Reproducible via
# tools/steering_rate_step_response.py (deterministic, no randomness).
#
# Caveat carried forward, not hidden -- and this applies to the 5.2
# figure ITSELF, not only to the peak rate below: car.xml's position
# actuators have no forcerange (actuator_forcelimited is False) --
# unlimited torque/current. Peak instantaneous slew rate right after the
# step is ~18.9 rad/s, an obvious modeling artifact, not a usable
# capability figure. But settling time in a PD-controlled 2nd-order
# system depends on how much torque is available to arrest the swing,
# not just on how fast it starts -- so the 2%-settling AVERAGE this
# constant is set to is ALSO inflated by the same unlimited-torque
# idealization. A torque-limited real servo would settle more slowly
# than this, not just peak lower.
#
# DELTA_DOT_MAX = 5.2 is therefore an UPPER BOUND from an idealized
# actuator, not a conservative estimate -- unlike the placeholder it
# replaces, which was guessed conservative on purpose. Do not treat this
# as safe to use unmargined; re-measure against a torque-limited model or
# real hardware if/when it exists (see ADR-20's caveat for the full
# reasoning).
DELTA_DOT_MAX = 5.2

# --- Track / perception envelope -------------------------------------------
R_MIN = min(RADIUS_1, RADIUS_2)  # m, tightest turn on the track (3.0 m)

# --- Horizon -----------------------------------------------------------
TS = 0.04   # s, 40 ms
N_HORIZON = 30  # steps -> T_preview = N*Ts = 1.2 s

# --- Cost weight normalization -----------------------------------------
# Weights are 1/(physical envelope)^2 -- a residual sitting exactly at its
# physical envelope boundary contributes unit cost, regardless of the
# term's own units (m, rad, rad, rad/s). Same normalization PHILOSOPHY as
# perception/model/targets.py's E_Y_SCALE/E_PSI_SCALE/KAPPA_SCALE (there
# applied to the residual directly; here to the weight, since acados' NLS
# cost takes a weight matrix rather than requiring the residual itself to
# be pre-scaled -- mathematically equivalent: (r/s)^2 == r^2 * (1/s^2)).
W_E_LAT = 1.0 / LANE_HALF_WIDTH ** 2
W_E_PSI = 1.0 / POS_HEADING_RANGE ** 2
W_DELTA = 1.0 / DELTA_MAX ** 2
W_U = 1.0 / DELTA_DOT_MAX ** 2

# Nominal speed for the DARE terminal-cost linearization (ADR-19): the
# terminal weight matrix P is computed once, at OCP-build time, for a
# single operating speed -- acados does not support a speed-varying
# terminal cost without a much more involved parametric-Riccati setup,
# out of scope here. 1.0 m/s matches this project's established reference
# speed (ADR-16's a_lat calculation) and sits comfortably under the
# horizon's speed ceiling (see ocp.py's v_max check).
V_NOMINAL_FOR_DARE = 1.0
