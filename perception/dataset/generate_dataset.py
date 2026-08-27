"""
Dataset label generation for the /lane_state perception task (M1).

Samples vehicle poses on and around REFERENCE_TRACK and computes their
ground-truth /lane_state via the SAME projection routine validated in
tests/test_lane_state_geometry.py. There is no independent labeling
logic here, by design — see docs/lane-state-contract.md and ADR-7.

This module has no MuJoCo dependency and is fully testable headless.
Camera rendering (turning these poses into pixels) is a separate,
Mac-only step — see render_dataset_images.py. Splitting the two means
a labeling bug and a rendering bug can never be confused with each other.
"""

import os
import sys
import math
import csv
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from perception.dataset.track_definitions import REFERENCE_TRACK, LANE_HALF_WIDTH
from perception.dataset.geometry import compute_lane_state
from perception.dataset.camera_visibility import lane_is_visible, any_lane_visible

N_TOTAL = 2000
NEGATIVE_FRACTION = 0.10
SEED = 42

# Positive envelope: the vehicle must remain INSIDE the marked lane.
# Tied to LANE_HALF_WIDTH rather than set independently — the first
# generation pass used a hard-coded 0.7 m against a 0.4 m half-width,
# which produced "valid" samples with the vehicle fully outside the
# markings. The labels were arithmetically correct; the envelope was not.
POS_LATERAL_MARGIN = 0.05    # m, keep clear of the marking itself
POS_LATERAL_RANGE = LANE_HALF_WIDTH - POS_LATERAL_MARGIN   # 0.35 m
POS_HEADING_RANGE = 0.5      # rad (~28.6 deg)

# Negative envelope: sampled wide, then REJECTED unless the camera
# genuinely sees no lane marking anywhere on the track. Envelope tuning
# alone cannot guarantee this — on a compact 45 m closed loop, a vehicle
# even 5-12 m off course still has some part of the loop in frame roughly
# 2 times out of 3. Rejection sampling makes the guarantee absolute.
NEG_LATERAL_RANGE = (2.0, 10.0)
NEG_HEADING_RANGE = (1.0, math.pi)
MAX_REJECTION_ATTEMPTS = 200

SPLIT_ZONES = 10
# 7/2/1 zones -> roughly 70/20/10by contiguous arc-length blocks, not
# per-sample random assignment. This is the M1 no-leakage requirement
# applied to pose sampling: without spatial binning, two nearly-identical
# poses (close in s, tiny offset difference) could land in different
# splits and inflate validation performance.
SPLIT_ASSIGNMENT = (["train"] * 7) + (["val"] * 2) + (["test"] * 1)


def _raw_sample(rng: np.random.Generator, negative: bool):
    s = rng.uniform(0.0, REFERENCE_TRACK.total_length)
    cx, cy = REFERENCE_TRACK.point_at(s)
    track_heading = REFERENCE_TRACK.heading_at(s)
    nx, ny = -math.sin(track_heading), math.cos(track_heading)  # left normal

    if not negative:
        lateral = rng.uniform(-POS_LATERAL_RANGE, POS_LATERAL_RANGE)
        heading_off = rng.uniform(-POS_HEADING_RANGE, POS_HEADING_RANGE)
    else:
        lateral = rng.choice([-1, 1]) * rng.uniform(*NEG_LATERAL_RANGE)
        heading_off = rng.choice([-1, 1]) * rng.uniform(*NEG_HEADING_RANGE)

    x = cx + lateral * nx
    y = cy + lateral * ny
    heading = track_heading + heading_off
    return x, y, heading


def sample_pose(rng: np.random.Generator, negative: bool):
    """Rejection sampling against the actual camera geometry.

    Positive: vehicle inside the lane AND the lane it should follow is
    visible ahead. Negative: no lane marking anywhere in frame.

    The visibility test is what makes the confidence label mean something.
    Checking it here rather than eyeballing rendered images caught that
    roughly half of the first pass's negatives still had clearly visible
    markings — invisible in a 50-image spot check, where only ~5 samples
    would have been negative.
    """
    for _ in range(MAX_REJECTION_ATTEMPTS):
        x, y, heading = _raw_sample(rng, negative)
        if not negative:
            if lane_is_visible(REFERENCE_TRACK, x, y, heading, LANE_HALF_WIDTH):
                return x, y, heading, 1.0
        else:
            if not any_lane_visible(REFERENCE_TRACK, x, y, heading, LANE_HALF_WIDTH):
                return x, y, heading, 0.0
    raise RuntimeError(
        f"rejection sampling failed after {MAX_REJECTION_ATTEMPTS} attempts "
        f"(negative={negative}). The envelope and the visibility criterion "
        f"are inconsistent — do not silently fall back to an unfiltered sample."
    )


def zone_for_s(s: float) -> str:
    zone_idx = min(SPLIT_ZONES - 1, int(s / REFERENCE_TRACK.total_length * SPLIT_ZONES))
    return SPLIT_ASSIGNMENT[zone_idx]


def quat_from_heading(heading: float):
    """MuJoCo quaternion convention: (w, x, y, z), rotation about world z."""
    return (math.cos(heading / 2.0), 0.0, 0.0, math.sin(heading / 2.0))


def generate_labels(seed: int = SEED, n_total: int = N_TOTAL):
    rng = np.random.default_rng(seed)
    n_negative = int(round(n_total * NEGATIVE_FRACTION))
    n_positive = n_total - n_negative

    flags = [False] * n_positive + [True] * n_negative
    rng.shuffle(flags)

    rows = []
    for i, is_negative in enumerate(flags):
        x, y, heading, confidence_target = sample_pose(rng, is_negative)
        ls = compute_lane_state(REFERENCE_TRACK, x, y, heading)
        split = zone_for_s(ls.s)

        rows.append({
            "filename": f"img_{i:05d}.png",
            "x": x, "y": y, "heading": heading,
            "s": ls.s,
            "lateral_error": ls.lateral_error,
            "heading_error": ls.heading_error,
            "curvature": ls.curvature,
            "confidence": confidence_target,
            "valid": confidence_target >= 0.5,
            "split": split,
        })
    return rows


def write_labels_csv(rows, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    rows = generate_labels()
    write_labels_csv(rows, "data/raw/labels.csv")
    print(f"Generated {len(rows)} labels -> data/raw/labels.csv")
    print(f"  positive (valid): {sum(1 for r in rows if r['valid'])}")
    print(f"  negative (invalid): {sum(1 for r in rows if not r['valid'])}")
    for split in ("train", "val", "test"):
        print(f"  {split}: {sum(1 for r in rows if r['split'] == split)}")