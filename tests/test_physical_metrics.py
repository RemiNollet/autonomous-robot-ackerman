"""
Tests for perception/model/physical_metrics.py.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

torch = pytest.importorskip("torch")
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from perception.model.physical_metrics import physical_metrics, format_physical_report
from perception.model.targets import E_Y_SCALE, E_PSI_SCALE, KAPPA_SCALE


class _PerfectModel(torch.nn.Module):
    """Ignores the image, returns the exact normalized target it's handed
    via a side channel -- lets the test control ground truth precisely
    without needing a real dataset or a trained checkpoint."""
    def __init__(self, targets, valid):
        super().__init__()
        self.targets = targets  # (N, 3) normalized
        self.valid = valid      # (N,)
        self.idx = 0
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        n = x.shape[0]
        t = self.targets[self.idx:self.idx + n]
        v = self.valid[self.idx:self.idx + n]
        self.idx += n
        conf_logit = torch.where(v > 0.5, torch.full_like(v, 10.0), torch.full_like(v, -10.0))
        out = torch.cat([t, conf_logit.unsqueeze(1)], dim=1)
        return out + self.dummy * 0, None


def _make_loader(n=20, error_e_y=0.0, error_e_psi=0.0, seed=0):
    """n samples split evenly across the three curvature bins, all valid;
    model predicts target + a fixed offset so MAE is exactly known."""
    rng = np.random.default_rng(seed)
    kappas = [0.0, 1 / 5, 1 / 3] * (n // 3 + 1)
    kappas = kappas[:n]
    targets = []
    for k in kappas:
        e_y = rng.uniform(-0.3, 0.3)
        e_psi = rng.uniform(-0.4, 0.4)
        targets.append([e_y / E_Y_SCALE, e_psi / E_PSI_SCALE, k / KAPPA_SCALE])
    targets = torch.tensor(targets, dtype=torch.float32)
    valid = torch.ones(n)

    imgs = torch.zeros(n, 3, 8, 8)  # model ignores these
    ds = TensorDataset(imgs, targets, valid)

    pred_targets = targets.clone()
    pred_targets[:, 0] += error_e_y / E_Y_SCALE
    pred_targets[:, 1] += error_e_psi / E_PSI_SCALE
    model = _PerfectModel(pred_targets, valid)
    return DataLoader(ds, batch_size=4, shuffle=False), model, targets


def test_perfect_predictions_give_zero_error():
    loader, model, _ = _make_loader(n=12, error_e_y=0.0, error_e_psi=0.0)
    m = physical_metrics(model, loader, torch.device("cpu"))
    assert m["e_y"]["mae"] < 1e-5
    assert m["e_psi"]["mae"] < 1e-5
    assert m["confidence"]["accuracy"] == 1.0


def test_known_constant_offset_gives_known_mae_and_rmse():
    loader, model, _ = _make_loader(n=12, error_e_y=0.05, error_e_psi=0.0)
    m = physical_metrics(model, loader, torch.device("cpu"))
    assert abs(m["e_y"]["mae"] - 0.05) < 1e-4
    # constant error -> RMSE == MAE == max == p50 == p95
    assert abs(m["e_y"]["rmse"] - 0.05) < 1e-4
    assert abs(m["e_y"]["max"] - 0.05) < 1e-4


def test_invalid_samples_excluded_from_e_y_e_psi_but_counted_in_confidence():
    rng = np.random.default_rng(1)
    n = 10
    targets = torch.tensor(
        [[rng.uniform(-0.3, 0.3) / E_Y_SCALE, rng.uniform(-0.4, 0.4) / E_PSI_SCALE, 0.0] for _ in range(n)],
        dtype=torch.float32,
    )
    valid = torch.tensor([1.0] * 7 + [0.0] * 3)
    imgs = torch.zeros(n, 3, 8, 8)
    ds = TensorDataset(imgs, targets, valid)
    model = _PerfectModel(targets.clone(), valid)
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    m = physical_metrics(model, loader, torch.device("cpu"))
    assert m["e_y"]["n"] == 7
    assert m["confidence"]["n_valid"] == 7 and m["confidence"]["n_invalid"] == 3


def test_curvature_bin_breakdown_separates_bins_correctly():
    loader, model, targets = _make_loader(n=15, error_e_y=0.0)
    m = physical_metrics(model, loader, torch.device("cpu"))
    total_binned = sum(m["e_y_by_bin"][label]["n"] for label, _ in
                        [("straight", 0.0), ("R=5m arc", 0.2), ("R=3m arc", 1 / 3)])
    assert total_binned == m["e_y"]["n"]


def test_format_physical_report_runs_and_mentions_kappa_is_not_reported():
    loader, model, _ = _make_loader(n=12)
    m = physical_metrics(model, loader, torch.device("cpu"))
    report = format_physical_report(m, title="test run", in_distribution=True,
                                     lane_half_width=0.4, heading_envelope=0.5)
    assert "kappa: not reported" in report
    assert "e_y" in report and "e_psi" in report
