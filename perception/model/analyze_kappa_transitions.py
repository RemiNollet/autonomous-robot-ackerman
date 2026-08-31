"""
Tests whether kappa's failure (docs/decisions.md ADR-11 finding 4/ADR-12) is
a mislabeling artifact rather than a model failure: the label is the exact
point-wise Frenet curvature at the vehicle's position, but the camera sees
up to L_usable (2.36 m) ahead. A sample near a curvature transition is
labeled with the CURRENT segment's curvature while showing the NEXT
segment's curve in frame.

Deliberately evaluates on the FULL dataset (all splits), not test alone:
ADR-9's per-primitive stratification puts every test-split sample in the
last 10% of its primitive's length, which means every test sample is,
by construction, within ~0.6 m of the next transition -- test alone has
ZERO straight samples beyond L_usable from a transition (verified: max
0.59 m), so it cannot distinguish "near-transition" from "far" at all.

Usage:
    python3 perception/model/analyze_kappa_transitions.py
        -> perception/model/kappa_transition_proximity.png, printed report
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from perception.dataset.track_definitions import REFERENCE_TRACK
from perception.model.dataset import LaneDataset
from perception.model.lane_cnn import LaneCNN
from perception.model.targets import KAPPA_SCALE

CHECKPOINT_PATH = "perception/model/checkpoints/lane_cnn_width1.0_best.pt"
# NOT kappa_transition_proximity.png (no suffix): that file is the frozen,
# committed evidence from the ORIGINAL (kappa loss weight=1/3) checkpoint
# that justified zeroing it -- see docs/decisions.md / results.md. That
# checkpoint no longer exists (save_checkpoint overwrites the same path
# every run), so it can't be regenerated if overwritten. This script always
# reflects "whatever checkpoint currently exists," which is a different,
# ongoing thing -- useful for re-checking after a future retrain, but must
# not silently clobber the historical justification.
OUT_PNG = "perception/model/kappa_transition_proximity_current_checkpoint.png"
L_USABLE = 2.356

TRANSITIONS = np.array(REFERENCE_TRACK.starts)
TRACK_LENGTH = REFERENCE_TRACK.total_length


def dist_to_next_transition(s: float) -> float:
    deltas = (TRANSITIONS - s) % TRACK_LENGTH
    deltas = np.where(deltas == 0, TRACK_LENGTH, deltas)
    return float(deltas.min())


def main():
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model = LaneCNN(width_mult=ckpt["width_mult"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ds = LaneDataset(split=None, augment=False)  # full dataset -- see module docstring
    rows = ds.rows

    preds = []
    with torch.no_grad():
        for i in range(len(ds)):
            img, target, valid = ds[i]
            pred, _ = model(img.unsqueeze(0))
            preds.append(pred[0].numpy())
    preds = np.array(preds)

    kappa_true = np.array([float(r["curvature"]) for r in rows])
    kappa_pred = preds[:, 2] * KAPPA_SCALE
    s_vals = np.array([float(r["s"]) for r in rows])
    valid_mask = np.array([r["valid"] == "True" for r in rows])
    splits = np.array([r["split"] for r in rows])
    dist = np.array([dist_to_next_transition(s) for s in s_vals])

    straight = valid_mask & (np.abs(kappa_true) < 1e-6)
    far = straight & (dist > L_USABLE)
    near = straight & (dist <= L_USABLE)

    print(f"straight samples (all splits): n={straight.sum()}")
    print(f"  far from transition (> {L_USABLE} m): n={far.sum():4d}  "
          f"mean|kappa_pred|={np.abs(kappa_pred[far]).mean():.4f}  std={np.abs(kappa_pred[far]).std():.4f}")
    print(f"  near a transition  (<= {L_USABLE} m): n={near.sum():4d}  "
          f"mean|kappa_pred|={np.abs(kappa_pred[near]).mean():.4f}  std={np.abs(kappa_pred[near]).std():.4f}")

    for split in ("train", "val", "test"):
        d = dist[straight & (splits == split)]
        print(f"  {split:5s} straight dist-to-transition range: "
              f"[{d.min():.2f}, {d.max():.2f}] m  (frac > {L_USABLE}m: {(d > L_USABLE).mean()*100:.1f}%)")

    corr = np.corrcoef(dist[straight], np.abs(kappa_pred[straight]))[0, 1]
    print(f"\ncorr(dist_to_transition, |kappa_pred|) on straights: {corr:.4f}")

    buckets = [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, L_USABLE), (L_USABLE, 3.5), (3.5, 6.0)]
    print("\ndistance bucket -> mean|kappa_pred| (straights only):")
    for lo, hi in buckets:
        m = straight & (dist > lo) & (dist <= hi)
        if m.sum() > 0:
            print(f"  ({lo:.2f}, {hi:.2f}] m:  n={m.sum():4d}  mean|kappa_pred|={np.abs(kappa_pred[m]).mean():.4f}")

    bins = [("straight (kappa=0)", 0.0, "tab:blue"),
            ("R=5m (|kappa|=1/5)", 1 / 5, "tab:orange"),
            ("R=3m (|kappa|=1/3)", 1 / 3, "tab:red")]
    fig, ax = plt.subplots(figsize=(9, 6))
    for label, mag, color in bins:
        m = valid_mask & (np.abs(np.abs(kappa_true) - mag) < 1e-6)
        ax.scatter(dist[m], np.abs(kappa_pred[m]), s=10, alpha=0.4, color=color, label=f"{label} (n={m.sum()})")
    ax.axvline(L_USABLE, color="k", linestyle="--", linewidth=1.2, label=f"L_usable = {L_USABLE} m")
    ax.set_xlabel("arc-length distance to next curvature transition ahead [m]")
    ax.set_ylabel("|kappa_pred| [1/m]")
    ax.set_title("|kappa_pred| vs distance to next transition, all splits\n"
                  "(test alone has zero straight samples beyond L_usable -- ADR-9 stratification "
                  "confines it to each primitive's last 10%)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0, TRACK_LENGTH / 2 + 0.5)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130)
    print(f"\nsaved {OUT_PNG}")


if __name__ == "__main__":
    main()
