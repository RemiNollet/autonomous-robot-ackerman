"""
Compares the ADR-14 baseline (kappa loss weight 0, hardcoded curvature=0.0
at publish time) against the ADR-18 retrain (kappa loss weight 1.0,
against data/dataset_v0/labels.csv relabeled with
perception/dataset/windowed_relabel.windowed_curvature_average).

Two questions, kept separate:

1. Does kappa prediction get usable? Near-join vs away-from-join MAE,
   both checkpoints, against a "predict 0" baseline (what's actually
   published today, ADR-14) -- not the old checkpoint's raw untrained
   kappa output, which was never a real baseline to begin with.
2. Do the other three outputs (e_y, e_psi, confidence) regress from
   retraining with the kappa loss term active? physical_metrics() on the
   test split, both checkpoints, same loader.

Evaluated on the FULL dataset for the kappa comparison (all splits, not
test alone) -- ADR-9's per-primitive stratification puts every test
sample within ~0.6 m of a transition by construction, so test alone
cannot represent "away from any join" at all (see
analyze_kappa_transitions.py's own docstring for the same point, made
there first). The e_y/e_psi/confidence check uses the test split, same
in-distribution/interpolation caveat as every other test-split number in
this project (ADR-11 finding 5) -- stated explicitly in the printed
report, not left implicit.

Usage:
    python3 perception/model/analyze_adr18_retrain.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import torch
from torch.utils.data import DataLoader

from perception.dataset.track_definitions import REFERENCE_TRACK, LANE_HALF_WIDTH
from perception.dataset.generate_dataset import POS_HEADING_RANGE
from perception.model.dataset import LaneDataset
from perception.model.lane_cnn import LaneCNN
from perception.model.targets import KAPPA_SCALE
from perception.model.physical_metrics import physical_metrics, format_physical_report

CHECKPOINT_DIR = "perception/model/checkpoints"
OLD_CHECKPOINT = f"{CHECKPOINT_DIR}/lane_cnn_width1.0_best_ADR14_kappa0_baseline.pt"
NEW_CHECKPOINT = f"{CHECKPOINT_DIR}/lane_cnn_width1.0_best.pt"
L_USABLE = 2.356

TRANSITIONS = np.array(REFERENCE_TRACK.starts)
TRACK_LENGTH = REFERENCE_TRACK.total_length


def dist_to_next_transition(s: float) -> float:
    deltas = (TRANSITIONS - s) % TRACK_LENGTH
    deltas = np.where(deltas == 0, TRACK_LENGTH, deltas)
    return float(deltas.min())


def load_model(path):
    ckpt = torch.load(path, map_location="cpu")
    model = LaneCNN(width_mult=ckpt["width_mult"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def kappa_predictions(model, ds):
    preds = []
    with torch.no_grad():
        for i in range(len(ds)):
            img, target, valid = ds[i]
            pred, _ = model(img.unsqueeze(0))
            preds.append(pred[0, 2].item() * KAPPA_SCALE)
    return np.array(preds)


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def main():
    print("=" * 70)
    print("Part 1: kappa prediction, near-join vs away-from-join")
    print("=" * 70)

    full_ds = LaneDataset(split=None, augment=False)
    rows = full_ds.rows
    s_vals = np.array([float(r["s"]) for r in rows])
    kappa_true = np.array([float(r["curvature"]) for r in rows])  # relabeled (ADR-18)
    valid_mask = np.array([r["valid"] == "True" for r in rows])
    dist = np.array([dist_to_next_transition(s) for s in s_vals])
    near = (dist <= L_USABLE) & valid_mask
    away = (dist > L_USABLE) & valid_mask
    print(f"n valid samples: {valid_mask.sum()} (near-join: {near.sum()}, "
          f"away-from-join: {away.sum()})")

    old_model = load_model(OLD_CHECKPOINT)
    new_model = load_model(NEW_CHECKPOINT)
    kappa_pred_old = kappa_predictions(old_model, full_ds)
    kappa_pred_new = kappa_predictions(new_model, full_ds)
    kappa_pred_zero = np.zeros_like(kappa_true)

    print(f"\n{'method':35s} {'near-join MAE':>15s} {'away-from-join MAE':>20s} {'ratio':>8s}")
    for label, pred in [
        ("published today (hardcoded 0.0)", kappa_pred_zero),
        ("ADR-14 checkpoint (untrained kappa)", kappa_pred_old),
        ("ADR-18 retrain (kappa weight 1.0)", kappa_pred_new),
    ]:
        mae_near = mae(pred[near], kappa_true[near])
        mae_away = mae(pred[away], kappa_true[away])
        ratio = mae_near / mae_away if mae_away > 0 else float("nan")
        print(f"{label:35s} {mae_near:15.4f} {mae_away:20.4f} {ratio:8.2f}")

    zero_near_mae = mae(kappa_pred_zero[near], kappa_true[near])
    new_near_mae = mae(kappa_pred_new[near], kappa_true[near])
    new_away_mae = mae(kappa_pred_new[away], kappa_true[away])
    improvement = (1 - new_near_mae / zero_near_mae) * 100 if zero_near_mae > 0 else float("nan")
    near_away_ratio = new_near_mae / new_away_mae if new_away_mae > 0 else float("nan")
    print(f"\nADR-18 vs published-today baseline, near-join: "
          f"{improvement:.1f}% {'improvement' if improvement > 0 else 'regression'}")
    print(f"ADR-18 near-join / away-from-join MAE ratio: {near_away_ratio:.2f}")
    print("Usability bar from the M3 'quadratic curvature relabeling' brief: "
          "near-join materially better than the zero baseline, AND "
          "near/away ratio not worse than ~2x.")

    print()
    print("=" * 70)
    print("Part 2: do e_y / e_psi / confidence regress? (test split)")
    print("=" * 70)
    print("IN-DISTRIBUTION / INTERPOLATION ONLY (ADR-11 finding 5) -- this")
    print("does not measure generalization, same caveat as every other")
    print("test-split number in this project. Not the mirror-generalization")
    print("probe (not re-run here; out of scope for this specific check).")
    print()

    test_ds = LaneDataset(split="test", augment=False)
    test_loader = DataLoader(test_ds, batch_size=64)

    old_phys = physical_metrics(old_model, test_loader, "cpu")
    new_phys = physical_metrics(new_model, test_loader, "cpu")

    print(f"{'output':12s} {'ADR-14 (kappa=0) MAE':>22s} {'ADR-18 (kappa=1) MAE':>22s} {'delta':>10s}")
    for key, unit, scale in [("e_y", "m", 1.0), ("e_psi", "deg", 180.0 / np.pi)]:
        old_v = old_phys[key]["mae"] * scale
        new_v = new_phys[key]["mae"] * scale
        print(f"{key + ' (' + unit + ')':12s} {old_v:22.4f} {new_v:22.4f} {new_v - old_v:+10.4f}")

    print(f"\nconfidence accuracy: ADR-14={old_phys['confidence']['accuracy']:.4f}  "
          f"ADR-18={new_phys['confidence']['accuracy']:.4f}")
    print(f"confidence valid recall: ADR-14={old_phys['confidence']['valid_recall']:.4f}  "
          f"ADR-18={new_phys['confidence']['valid_recall']:.4f}")
    print(f"confidence invalid recall: ADR-14={old_phys['confidence']['invalid_recall']:.4f}  "
          f"ADR-18={new_phys['confidence']['invalid_recall']:.4f}")


if __name__ == "__main__":
    main()
