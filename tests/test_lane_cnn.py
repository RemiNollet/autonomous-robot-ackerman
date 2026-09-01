"""
Tests for perception/model/lane_cnn.py.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

torch = pytest.importorskip("torch")

from perception.model.lane_cnn import LaneCNN, IN_CHANNELS, IN_HEIGHT, IN_WIDTH, count_params, count_macs

TARGET_PARAMS = 417_000
TARGET_MACS = 33.2e6
TOLERANCE = 0.05


def test_output_shape_and_finite_on_random_batch():
    torch.manual_seed(0)
    model = LaneCNN(width_mult=1.0)
    model.eval()
    x = torch.randn(6, IN_CHANNELS, IN_HEIGHT, IN_WIDTH)
    with torch.no_grad():
        out, feat = model(x)
    assert out.shape == (6, 4)
    assert feat.shape == (6, 96, 5, 10), "b3 feature map shape changed -- check the encoder strides"
    assert torch.isfinite(out).all()
    assert torch.isfinite(feat).all()


def test_param_count_within_5_percent_of_417k():
    model = LaneCNN(width_mult=1.0)
    n = count_params(model)
    rel_err = abs(n - TARGET_PARAMS) / TARGET_PARAMS
    assert rel_err <= TOLERANCE, f"param count {n:,} is {rel_err:.1%} off target {TARGET_PARAMS:,}"


def test_mac_count_within_5_percent_of_33_2m():
    model = LaneCNN(width_mult=1.0)
    macs = count_macs(model)
    rel_err = abs(macs - TARGET_MACS) / TARGET_MACS
    assert rel_err <= TOLERANCE, f"MAC count {macs:,} is {rel_err:.1%} off target {TARGET_MACS:,.0f}"


def test_width_mult_actually_scales_capacity():
    """The ablation in train.py compares width_mult=1.0 against 0.25 -- if
    the constructor silently ignored width_mult, that comparison would be
    meaningless."""
    full = LaneCNN(width_mult=1.0)
    quarter = LaneCNN(width_mult=0.25)
    assert count_params(quarter) < count_params(full) * 0.5
    assert count_macs(quarter) < count_macs(full) * 0.5

    x = torch.randn(2, IN_CHANNELS, IN_HEIGHT, IN_WIDTH)
    out, feat = quarter(x)
    assert out.shape == (2, 4)
    assert feat.shape[1] == 24, "b3 channel count should also scale with width_mult (96 * 0.25 = 24)"


def test_overfit_eight_samples_below_1en3():
    """Sanity check on the training loop + model, not a realistic training
    config: 8 real samples, 200 steps, loss should collapse near zero.
    Dropout is disabled for this check only -- its entire purpose is to
    prevent memorization, so leaving it on while testing "can this model
    memorize 8 examples" mostly just measures dropout noise, not model or
    training-loop correctness. BatchNorm stays in train mode (adapting to
    the 8-sample batch statistics, as it would during real training)."""
    from perception.model.dataset import LaneDataset
    from perception.model.loss import lane_loss
    from torch.utils.data import DataLoader, Subset

    torch.manual_seed(0)
    ds = LaneDataset(split="train", augment=False)
    subset = Subset(ds, list(range(8)))
    img, target, valid = next(iter(DataLoader(subset, batch_size=8, shuffle=False)))
    assert valid.sum() > 0, "fixture batch must contain at least one valid sample"

    model = LaneCNN(width_mult=1.0)
    model.train()
    model.dropout.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss = None
    for _ in range(200):
        pred, _ = model(img)
        loss, _, _ = lane_loss(pred, target, valid)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert loss.item() < 1e-3, f"final loss {loss.item():.5f} did not collapse on 8 memorized samples"
