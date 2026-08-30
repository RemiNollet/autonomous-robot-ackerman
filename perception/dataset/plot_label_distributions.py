"""
Label distribution check for data/dataset_v0/labels.csv (M1).

The risk this guards against (SRS risk register, M1): a dataset dominated by
straight-line samples would let the CNN learn a degenerate lateral/heading
regression that never has to generalize to curved track sections. Curvature
is fixed by track geometry at the projection point (see geometry.py), so
grouping by curvature bin is exact, not estimated.

Usage:
    python3 perception/dataset/plot_label_distributions.py
"""

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from perception.dataset.track_definitions import LANE_HALF_WIDTH

DATASET_DIR = "data/dataset_v0"
LABELS_CSV = f"{DATASET_DIR}/labels.csv"
OUT_PNG = "docs/dataset/label_distributions.png"

CURVATURE_BIN_TOL = 1e-6
CURVATURE_BINS = [
    ("straight (κ=0)", 0.0),
    ("left, R=5m (κ=1/5)", 1 / 5),
    ("left, R=3m (κ=1/3)", 1 / 3),
]


def curvature_bin_label(curvature: float) -> str:
    for label, mag in CURVATURE_BINS:
        if abs(abs(curvature) - mag) < CURVATURE_BIN_TOL:
            return label
    return f"other (κ={curvature:.3f})"


def main():
    with open(LABELS_CSV) as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        r["lateral_error"] = float(r["lateral_error"])
        r["heading_error"] = float(r["heading_error"])
        r["curvature"] = float(r["curvature"])
        r["valid"] = r["valid"] == "True"
        r["bin"] = curvature_bin_label(r["curvature"])

    bin_labels = [b[0] for b in CURVATURE_BINS]
    valid_rows = [r for r in rows if r["valid"]]

    print(f"Loaded {len(rows)} samples ({len(valid_rows)} valid) from {LABELS_CSV}\n")
    print("Curvature bin coverage (valid samples only):")
    for label in bin_labels:
        n = sum(1 for r in valid_rows if r["bin"] == label)
        pct = 100 * n / len(valid_rows)
        flag = "  <-- under-represented (<10%)" if pct < 10 else ""
        print(f"  {label:22s} n={n:5d} ({pct:5.1f}%){flag}")

    print("\nSplit x curvature-bin counts (valid samples only):")
    for split in ("train", "val", "test"):
        counts = [sum(1 for r in valid_rows if r["split"] == split and r["bin"] == label)
                  for label in bin_labels]
        print(f"  {split:5s} " + "  ".join(f"{label}={c}" for label, c in zip(bin_labels, counts)))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    colors = {bin_labels[0]: "tab:blue", bin_labels[1]: "tab:orange", bin_labels[2]: "tab:green"}

    ax = axes[0, 0]
    counts = [sum(1 for r in valid_rows if r["bin"] == label) for label in bin_labels]
    ax.bar(bin_labels, counts, color=[colors[b] for b in bin_labels])
    ax.set_title("Valid sample count by curvature bin")
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=15)

    ax = axes[0, 1]
    for label in bin_labels:
        vals = [r["lateral_error"] for r in valid_rows if r["bin"] == label]
        ax.hist(vals, bins=30, range=(-LANE_HALF_WIDTH, LANE_HALF_WIDTH),
                 alpha=0.55, label=label, color=colors[label])
    ax.axvline(-LANE_HALF_WIDTH, color="k", linestyle="--", linewidth=0.8)
    ax.axvline(LANE_HALF_WIDTH, color="k", linestyle="--", linewidth=0.8)
    ax.set_title("lateral_error by curvature bin (valid samples)")
    ax.set_xlabel("lateral_error [m]")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for label in bin_labels:
        vals = [r["heading_error"] for r in valid_rows if r["bin"] == label]
        ax.hist(vals, bins=30, alpha=0.55, label=label, color=colors[label])
    ax.set_title("heading_error by curvature bin (valid samples)")
    ax.set_xlabel("heading_error [rad]")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    width = 0.25
    x = np.arange(len(bin_labels))
    for i, split in enumerate(("train", "val", "test")):
        counts = [sum(1 for r in valid_rows if r["split"] == split and r["bin"] == label)
                  for label in bin_labels]
        ax.bar(x + (i - 1) * width, counts, width, label=split)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=15)
    ax.set_title("Curvature-bin coverage per split")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)

    fig.suptitle(f"Label distributions — {DATASET_DIR} (n={len(rows)})")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130)
    print(f"\nFigure written to {OUT_PNG}")


if __name__ == "__main__":
    main()
