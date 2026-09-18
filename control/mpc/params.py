"""Shared constants for the lateral MPC (ADR-19; cost weight tuning
ADR-22).

Physical constants are imported from where they're already established
in this project (perception/sim), not restated, so this can't silently
drift from the values those actually use (ADR-8).
"""

from perception.dataset.track_definitions import (
    LANE_HALF_WIDTH, RADIUS_1, RADIUS_2,
)
from perception.dataset.generate_dataset import POS_HEADING_RANGE

# --- Vehicle -----------------------------------------------------------
L = 0.26  # m, wheelbase (sim/models/car.xml axle spacing, ADR-19)
DELTA_MAX = 0.6  # rad, hard steering limit (car.xml joint/actuator range)

# rad/s, steering rate limit -- measured via
# tools/steering_rate_step_response.py, sim-only idealized ceiling, not
# a validated hardware bound (ADR-20, ADR-22).
DELTA_DOT_MAX = 5.2

# --- Track / perception envelope ----------------------------------------
R_MIN = min(RADIUS_1, RADIUS_2)  # m, tightest turn on the track (3.0 m)

# --- Horizon -------------------------------------------------------------
TS = 0.04       # s, 40 ms
N_HORIZON = 30  # steps -> T_preview = N*Ts = 1.2 s

# --- Cost weight normalization ---------------------------------------
# Weights are 1/(physical envelope)^2, same normalization philosophy as
# perception/model/targets.py's *_SCALE constants (ADR-22).
_Q_ISOTROPIC_SCALE = 2.0  # ADR-22 sweep result

W_E_LAT = _Q_ISOTROPIC_SCALE / LANE_HALF_WIDTH ** 2
W_E_PSI = _Q_ISOTROPIC_SCALE / POS_HEADING_RANGE ** 2
W_DELTA = 1.0 / DELTA_MAX ** 2
W_U = 1.0 / DELTA_DOT_MAX ** 2  # kept at baseline, not loosened (ADR-22)

# Nominal speed for the DARE terminal-cost linearization, computed once
# at OCP-build time (ADR-19; acados has no speed-varying terminal cost
# without a parametric-Riccati setup, out of scope here).
V_NOMINAL_FOR_DARE = 1.0
