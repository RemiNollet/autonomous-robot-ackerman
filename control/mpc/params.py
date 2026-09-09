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

# rad/s, steering RATE limit -- UNMEASURED. No steering rate limit exists
# anywhere in car.xml (position actuators have kp/kv servo gains, not a
# hard rate limit) -- confirmed in the M3 kickoff investigation item 6,
# which flagged this needs a bench test. This value is a conservative
# placeholder, not a measurement: chosen deliberately low (a full
# -DELTA_MAX -> +DELTA_MAX sweep, 1.2 rad, would take >=0.6 s at this rate)
# so a wrong guess fails safe (too slow, not too fast) until it's replaced
# with a real number.
#
# TODO(bench test): characterize the actual servo slew rate on hardware
# (or the sim's position-actuator step response) and replace this
# placeholder. Do not let this quietly become "the real number" without
# that measurement -- see ADR-19's explicit flag on this.
DELTA_DOT_MAX_PLACEHOLDER = 2.0

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
W_U = 1.0 / DELTA_DOT_MAX_PLACEHOLDER ** 2

# Nominal speed for the DARE terminal-cost linearization (ADR-19): the
# terminal weight matrix P is computed once, at OCP-build time, for a
# single operating speed -- acados does not support a speed-varying
# terminal cost without a much more involved parametric-Riccati setup,
# out of scope here. 1.0 m/s matches this project's established reference
# speed (ADR-16's a_lat calculation) and sits comfortably under the
# horizon's speed ceiling (see ocp.py's v_max check).
V_NOMINAL_FOR_DARE = 1.0
