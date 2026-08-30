"""
Per-channel image normalization statistics for perception/dataset (M1, NFR-8).

Computed over the TRAIN split only -- val/test must stay unseen statistics
as well as unseen samples, otherwise the split's leakage guarantee is
undermined one step downstream of pose sampling. Written to a shared JSON
file so training (M2) and the ROS2 perception_node's preprocessing use the
exact same numbers, per NFR-8 ("identical between training and inference,
factored into shared code") -- JSON rather than a Python import so the ROS2
node isn't required to import training code to get them.

Usage:
    python3 perception/dataset/compute_normalization_stats.py
"""

import csv
import json
import os

import numpy as np
from PIL import Image

DATASET_DIR = "data/dataset_v0"
LABELS_CSV = f"{DATASET_DIR}/labels.csv"
IMG_DIR = f"{DATASET_DIR}/images"
OUT_JSON = "perception/dataset/normalization_stats.json"
STATS_SPLIT = "train"


def main():
    with open(LABELS_CSV) as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == STATS_SPLIT]

    n_pixels = 0
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sumsq = np.zeros(3, dtype=np.float64)

    for i, row in enumerate(rows):
        path = os.path.join(IMG_DIR, row["filename"])
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
        channel_sum += arr.sum(axis=(0, 1))
        channel_sumsq += (arr ** 2).sum(axis=(0, 1))
        n_pixels += arr.shape[0] * arr.shape[1]

        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(rows)} images")

    mean = channel_sum / n_pixels
    var = channel_sumsq / n_pixels - mean ** 2
    std = np.sqrt(np.maximum(var, 0.0))

    stats = {
        "dataset_dir": DATASET_DIR,
        "split": STATS_SPLIT,
        "n_images": len(rows),
        "n_pixels_per_channel": n_pixels,
        "channel_order": "RGB",
        "value_range": "[0, 1] (uint8 / 255 before normalization)",
        "mean": mean.tolist(),
        "std": std.tolist(),
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nmean (RGB) = {mean}")
    print(f"std  (RGB) = {std}")
    print(f"Written to {OUT_JSON}")


if __name__ == "__main__":
    main()
