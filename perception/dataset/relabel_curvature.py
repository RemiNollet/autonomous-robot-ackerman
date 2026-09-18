"""
Relabels data/dataset_v0/labels.csv's curvature column with
windowed_curvature_average (ADR-18) in place of the point-wise Frenet
projection compute_lane_state uses (ADR-14: the point-wise label is
measurably wrong within L_usable of a curvature transition, ~42% of the
loop). All 4000 rows, base and mirrored -- no re-render, no MuJoCo: only
x/y (already stored per row) and REFERENCE_TRACK are needed.

Mirrored rows share their source row's x/y/heading exactly (generate_dataset.
mirror_row's own docstring: "x/y/heading/s are left as the SOURCE pose's
values, not a fabricated 'mirrored world pose'") -- so mirrored curvature
here is NOT recomputed independently (that would silently drop the sign
flip, since windowed_curvature_average depends only on x/y, which mirrored
rows share unchanged with their source). It is derived the same way
mirror_row() already derives it: negate the (new) base row's value.

Backs up the pre-relabel file to labels_v0_pointwise.csv before writing --
never overwrites without that backup existing first.

Usage:
    python3 perception/dataset/relabel_curvature.py
"""
import csv
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from perception.dataset.track_definitions import (  # noqa: E402
    REFERENCE_TRACK,
)
from perception.dataset.windowed_relabel import (  # noqa: E402
    windowed_curvature_average,
)

DATASET_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "dataset_v0")
LABELS_PATH = os.path.join(DATASET_DIR, "labels.csv")
BACKUP_PATH = os.path.join(DATASET_DIR, "labels_v0_pointwise.csv")


def main():
    if not os.path.exists(BACKUP_PATH):
        shutil.copy2(LABELS_PATH, BACKUP_PATH)
        print(f"Backed up {LABELS_PATH} -> {BACKUP_PATH}")
    else:
        print(f"{BACKUP_PATH} already exists, not overwriting the backup.")

    with open(BACKUP_PATH, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    base_new_kappa = {}
    n_base = n_mirrored = 0
    for row in rows:
        if row["mirrored"] != "True":
            x, y = float(row["x"]), float(row["y"])
            new_kappa = windowed_curvature_average(REFERENCE_TRACK, x, y)
            row["curvature"] = repr(new_kappa)
            base_new_kappa[row["filename"]] = new_kappa
            n_base += 1

    for row in rows:
        if row["mirrored"] == "True":
            source_kappa = base_new_kappa[row["source_filename"]]
            row["curvature"] = repr(-source_kappa)
            n_mirrored += 1

    assert n_base + n_mirrored == len(rows), (
        f"{n_base} base + {n_mirrored} mirrored != {len(rows)} total rows -- "
        "every row must be classified as exactly one or the other."
    )

    with open(LABELS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Relabeled {n_base} base + {n_mirrored} mirrored = "
          f"{len(rows)} rows.")
    print(f"Wrote {LABELS_PATH} (old point-wise labels preserved at "
          f"{BACKUP_PATH}).")


if __name__ == "__main__":
    main()
