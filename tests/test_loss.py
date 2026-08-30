"""
Tests for perception/model/loss.py.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

torch = pytest.importorskip("torch")

from perception.model.loss import lane_loss, LAMBDA_CONF


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


def test_regression_loss_nonzero_and_matches_smooth_l1_on_all_valid_batch():
    torch.manual_seed(0)
    batch = 5
    pred = torch.randn(batch, 4)
    target = torch.randn(batch, 3)
    valid = torch.ones(batch)

    total, regression_loss, confidence_loss = lane_loss(pred, target, valid)

    expected_reg = torch.nn.functional.smooth_l1_loss(pred[:, :3], target)
    assert torch.isclose(regression_loss, expected_reg, atol=1e-6)
    assert regression_loss.item() > 0.0


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
