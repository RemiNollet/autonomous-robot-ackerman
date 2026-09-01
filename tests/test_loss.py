"""
Tests for perception/model/loss.py.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

torch = pytest.importorskip("torch")

from perception.model.loss import lane_loss, LAMBDA_CONF, COMPONENT_WEIGHTS, component_losses


def test_regression_loss_is_exactly_zero_on_all_invalid_batch():
    """The regression term must be ZERO-weighted on invalid samples, not
    merely down-weighted -- an invalid sample's lateral_error/heading_error/
    curvature describe a pose the vehicle was never meant to track
    (docs/lane-state-contract.md section 4), so nothing should pull the
    network toward matching them. Predictions are deliberately far from
    target here (large random values), so a nonzero regression_loss would
    be caught, not masked by coincidental near-agreement."""
    torch.manual_seed(0)
    batch = 5
    pred = torch.randn(batch, 4) * 100.0  # wildly wrong on purpose
    target = torch.randn(batch, 3) * 100.0
    valid = torch.zeros(batch)  # every sample invalid

    total, regression_loss, confidence_loss = lane_loss(pred, target, valid)

    assert regression_loss.item() == 0.0
    # Confidence is still supervised on every sample -- total should equal
    # exactly lambda_conf * confidence_loss when regression contributes zero.
    assert torch.isclose(total, LAMBDA_CONF * confidence_loss, atol=1e-6)


def test_kappa_weight_is_zero_by_default():
    """The point-wise Frenet kappa label is measurably wrong within L_usable
    of a curvature transition (docs/decisions.md ADR-11 finding 3, confirmed
    by the M2 transition-proximity diagnostic: mean|kappa_pred| correlates
    at r=-0.67 with distance to the next transition). Its default weight
    must be 0, not merely small -- changing pred[:,2] (kappa) alone must not
    move regression_loss at all."""
    assert COMPONENT_WEIGHTS["kappa"] == 0.0
    torch.manual_seed(4)
    batch = 5
    pred_a = torch.randn(batch, 4)
    pred_b = pred_a.clone()
    pred_b[:, 2] = torch.randn(batch) * 1000.0  # kappa channel only, wildly different
    target = torch.randn(batch, 3)
    valid = torch.ones(batch)

    _, reg_a, _ = lane_loss(pred_a, target, valid)
    _, reg_b, _ = lane_loss(pred_b, target, valid)
    assert torch.isclose(reg_a, reg_b, atol=1e-6)


def test_regression_loss_matches_weighted_mean_of_e_y_and_e_psi_by_default():
    """With kappa's default weight at 0, regression_loss is the mean of
    e_y's and e_psi's SmoothL1 alone -- not that mean diluted by dividing
    by 3 (an always-zero third term must not shrink the other two)."""
    torch.manual_seed(0)
    batch = 5
    pred = torch.randn(batch, 4)
    target = torch.randn(batch, 3)
    valid = torch.ones(batch)

    total, regression_loss, confidence_loss = lane_loss(pred, target, valid)

    per_target = torch.nn.functional.smooth_l1_loss(pred[:, :3], target, reduction="none").mean(dim=0)
    expected_reg = (per_target[0] + per_target[1]) / 2.0
    assert torch.isclose(regression_loss, expected_reg, atol=1e-6)
    assert regression_loss.item() > 0.0


def test_component_weights_equal_reproduces_unweighted_mean():
    """Explicitly passing equal weights recovers the old (pre-ADR-12)
    behaviour -- the mean over all three targets -- confirming the weighting
    mechanism itself, independent of what the current default happens to be."""
    torch.manual_seed(0)
    batch = 5
    pred = torch.randn(batch, 4)
    target = torch.randn(batch, 3)
    valid = torch.ones(batch)

    _, regression_loss, _ = lane_loss(
        pred, target, valid, component_weights={"e_y": 1.0, "e_psi": 1.0, "kappa": 1.0}
    )
    expected_reg = torch.nn.functional.smooth_l1_loss(pred[:, :3], target)
    assert torch.isclose(regression_loss, expected_reg, atol=1e-6)


def test_mixed_batch_regression_loss_ignores_invalid_rows():
    """A batch with some invalid rows should give the same regression loss
    as the same batch with those rows removed entirely -- confirms masking,
    not just down-weighting, at the per-sample level (not only the
    all-invalid edge case above)."""
    torch.manual_seed(1)
    pred_valid_rows = torch.randn(3, 4)
    target_valid_rows = torch.randn(3, 3)
    pred_invalid_rows = torch.randn(2, 4) * 50.0
    target_invalid_rows = torch.randn(2, 3) * 50.0

    pred_mixed = torch.cat([pred_valid_rows, pred_invalid_rows], dim=0)
    target_mixed = torch.cat([target_valid_rows, target_invalid_rows], dim=0)
    valid_mixed = torch.cat([torch.ones(3), torch.zeros(2)])

    _, reg_mixed, _ = lane_loss(pred_mixed, target_mixed, valid_mixed)
    _, reg_valid_only, _ = lane_loss(pred_valid_rows, target_valid_rows, torch.ones(3))

    assert torch.isclose(reg_mixed, reg_valid_only, atol=1e-6)


def test_component_losses_sum_matches_combined_regression_loss():
    """component_losses is an UNWEIGHTED diagnostic breakdown (kappa's loss
    still computed and logged for monitoring, even though it doesn't
    contribute to training by default) -- it must not silently apply
    COMPONENT_WEIGHTS itself, or the loss_curves.png plot would misreport
    what's actually happening to kappa. Combined with the default weights,
    the e_y/e_psi mean (not all three) must equal what lane_loss reports."""
    torch.manual_seed(2)
    batch = 6
    pred = torch.randn(batch, 4)
    target = torch.randn(batch, 3)
    valid = torch.tensor([1., 1., 0., 1., 0., 1.])

    _, regression_loss, confidence_loss = lane_loss(pred, target, valid)
    comps = component_losses(pred, target, valid)

    mean_of_active_components = (comps["e_y"] + comps["e_psi"]) / 2.0
    assert abs(mean_of_active_components - regression_loss.item()) < 1e-5
    assert abs(comps["confidence"] - confidence_loss.item()) < 1e-5


def test_component_losses_zero_on_all_invalid_batch():
    torch.manual_seed(3)
    pred = torch.randn(4, 4) * 50.0
    target = torch.randn(4, 3) * 50.0
    valid = torch.zeros(4)
    comps = component_losses(pred, target, valid)
    assert comps["e_y"] == 0.0 and comps["e_psi"] == 0.0 and comps["kappa"] == 0.0
