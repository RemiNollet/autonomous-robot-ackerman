"""
Tests for perception/dataset/generate_dataset.py — the label-generation
core only (no MuJoCo/rendering dependency).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from perception.dataset.generate_dataset import (
    generate_labels, POS_LATERAL_RANGE, POS_HEADING_RANGE, zone_for_s,
)
from perception.dataset.track_definitions import REFERENCE_TRACK, LANE_HALF_WIDTH
from perception.dataset.camera_visibility import lane_is_visible, any_lane_visible


def test_generation_is_deterministic():
    a = generate_labels(seed=123, n_total=200)
    b = generate_labels(seed=123, n_total=200)
    assert a == b, "same seed must produce identical dataset (NFR-3 reproducibility)"


def test_positive_negative_split_matches_config():
    rows = generate_labels(seed=1, n_total=1000)
    n_valid = sum(1 for r in rows if r["valid"])
    n_invalid = sum(1 for r in rows if not r["valid"])
    assert n_valid == 900
    assert n_invalid == 100


def test_positive_samples_within_declared_envelope():
    rows = generate_labels(seed=1, n_total=2000)
    for r in rows:
        if r["valid"]:
            assert abs(r["lateral_error"]) <= POS_LATERAL_RANGE + 1e-6
            assert abs(r["heading_error"]) <= POS_HEADING_RANGE + 1e-6


def test_negative_samples_exist_outside_positive_envelope():
    """At least one axis (lateral or heading) should be clearly outside
    the positive envelope for every negative sample — otherwise a
    'negative' sample would be geometrically indistinguishable from a
    positive one, which would teach the confidence head the wrong thing."""
    rows = generate_labels(seed=1, n_total=2000)
    for r in rows:
        if not r["valid"]:
            lateral_far = abs(r["lateral_error"]) > POS_LATERAL_RANGE
            heading_far = abs(r["heading_error"]) > POS_HEADING_RANGE
            assert lateral_far or heading_far, (
                f"negative sample not actually outside positive envelope: {r}"
            )


def test_positive_samples_stay_inside_the_marked_lane():
    """Regression test. The first generation pass allowed positive samples
    up to 0.7 m lateral against a 0.4 m lane half-width, putting the vehicle
    fully outside the markings while still labeled valid=True. The envelope
    is now derived from LANE_HALF_WIDTH so the two cannot drift apart."""
    rows = generate_labels(seed=5, n_total=1000)
    for r in rows:
        if r["valid"]:
            assert abs(r["lateral_error"]) < LANE_HALF_WIDTH, (
                f"valid sample outside the marked lane: lateral_error="
                f"{r['lateral_error']:.3f}, lane half-width={LANE_HALF_WIDTH}"
            )


def test_positive_samples_actually_see_the_lane():
    """Every valid sample must have the lane it should follow visible in
    the camera frame — otherwise the label describes something the CNN
    cannot possibly infer from the image."""
    rows = generate_labels(seed=5, n_total=400)
    for r in rows:
        if r["valid"]:
            assert lane_is_visible(
                REFERENCE_TRACK, r["x"], r["y"], r["heading"], LANE_HALF_WIDTH
            ), f"valid sample with no lane visible ahead: {r['filename']}"


def test_negative_samples_show_no_lane_at_all():
    """Regression test for the more damaging of the two original bugs:
    roughly half of the first pass's negatives still had clearly visible
    markings somewhere in frame, because a 45 m closed loop is hard to
    escape visually. Those samples would have trained the confidence head
    to report zero confidence on perfectly readable images."""
    rows = generate_labels(seed=5, n_total=400)
    for r in rows:
        if not r["valid"]:
            assert not any_lane_visible(
                REFERENCE_TRACK, r["x"], r["y"], r["heading"], LANE_HALF_WIDTH
            ), f"invalid sample with lane markings visible: {r['filename']}"


def test_all_splits_present():
    rows = generate_labels(seed=1, n_total=2000)
    splits = {r["split"] for r in rows}
    assert splits == {"train", "val", "test"}


def test_filenames_are_unique():
    rows = generate_labels(seed=1, n_total=500)
    names = [r["filename"] for r in rows]
    assert len(names) == len(set(names))


def test_split_has_no_leakage_across_track_zones():
    """Splits are assigned by contiguous arc-length sub-zone within each
    track primitive, not per-sample, so that two nearly-identical poses
    (same track segment, tiny offset delta) can never land in different
    splits. Verify the sub-zone -> split mapping is a pure function of s
    (via zone_for_s) on the actually-generated data, not just on
    SPLIT_ASSIGNMENT in isolation."""
    rows = generate_labels(seed=1, n_total=2000)
    for r in rows:
        assert r["split"] == zone_for_s(r["s"]), (
            f"row split {r['split']!r} does not match zone_for_s({r['s']}) "
            f"= {zone_for_s(r['s'])!r}"
        )


def test_every_curvature_bin_present_in_every_split():
    """Regression test: a single loop-wide 70/20/10 cut concentrated the
    entire test split inside one R=5m arc (no straight, no R=3m samples) and
    left val with no R=3m coverage at all -- a real defect caught by
    checking split x curvature-bin counts programmatically, not by eye.
    Splitting per-primitive (see zone_for_s) must guarantee every curvature
    magnitude on the track appears in every split."""
    rows = generate_labels(seed=1, n_total=2000)
    curvature_bins = {0.0, round(1 / 5, 3), round(1 / 3, 3)}
    for split in ("train", "val", "test"):
        got = {round(abs(r["curvature"]), 3) for r in rows if r["split"] == split}
        assert got == curvature_bins, (
            f"split {split!r} missing curvature bin(s): {curvature_bins - got}"
        )