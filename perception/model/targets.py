"""
Normalization for the three regressed /lane_state scalars. One place, so
train.py, dataset.py, and inference all agree (NFR-8's train/inference
consistency requirement, applied to targets the same way
normalization_stats.json applies it to inputs).

Scales are the physical envelope each quantity cannot exceed on a valid
sample, imported from the modules that own them rather than restated here,
so this can't silently drift from the dataset generator (the exact failure
mode ADR-8 fixed for the sampling envelope itself):
  - LANE_HALF_WIDTH: a valid sample's |lateral_error| is strictly inside this
    (perception/dataset/track_definitions.py)
  - POS_HEADING_RANGE: a valid sample's |heading_error| is within this
    (perception/dataset/generate_dataset.py)
  - RADIUS_1: the tighter of the track's two arc radii, so |curvature| on
    any sample is within 1/RADIUS_1 (track_definitions.py)

Confidence is not normalized -- it's already a logit/probability pair.
"""

from perception.dataset.track_definitions import LANE_HALF_WIDTH, RADIUS_1
from perception.dataset.generate_dataset import POS_HEADING_RANGE

E_Y_SCALE = LANE_HALF_WIDTH        # m
E_PSI_SCALE = POS_HEADING_RANGE    # rad
KAPPA_SCALE = 1.0 / RADIUS_1       # 1/m


def normalize_targets(e_y: float, e_psi: float, kappa: float):
    return e_y / E_Y_SCALE, e_psi / E_PSI_SCALE, kappa / KAPPA_SCALE


def denormalize_targets(e_y_n: float, e_psi_n: float, kappa_n: float):
    return e_y_n * E_Y_SCALE, e_psi_n * E_PSI_SCALE, kappa_n * KAPPA_SCALE
