"""
torch Dataset over data/dataset_v0/labels.csv + images/. No framework
abstractions -- a plain Dataset subclass, filtered by split (and optionally
by the `mirrored` column, for the ADR-11-finding-5 mirror-generalization
probe in train.py: train on source samples only, evaluate on their mirror
twins, which is the only v0 evaluation that isn't pure interpolation).

Augmentation (perturb_image, preprocess.augment_image) is applied only when
`augment=True`, which callers should only ever set for the train split --
this mirrors the record-level convention of the CSV itself (a `split`
column, not a magic filename pattern).
"""

import csv
import os
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from perception.model.preprocess import crop_and_resize, to_standardized_array, augment_image
from perception.model.targets import normalize_targets

DATASET_DIR = "data/dataset_v0"
LABELS_CSV = f"{DATASET_DIR}/labels.csv"
IMG_DIR = f"{DATASET_DIR}/images"


class LaneDataset(Dataset):
    def __init__(self, labels_csv: str = LABELS_CSV, img_dir: str = IMG_DIR,
                 split: Optional[str] = None, mirrored: Optional[bool] = None,
                 augment: bool = False, seed: int = 0):
        """
        split:    "train" / "val" / "test", or None for all splits.
        mirrored: True/False to keep only mirrored or only source rows, or
                  None for both -- see mirror-generalization probe above.
        augment:  apply preprocess.augment_image (train split only, by
                  convention of the caller, not enforced here).
        """
        self.img_dir = img_dir
        self.augment = augment
        self.rng = np.random.default_rng(seed)

        with open(labels_csv) as f:
            rows = list(csv.DictReader(f))
        if split is not None:
            rows = [r for r in rows if r["split"] == split]
        if mirrored is not None:
            want = "True" if mirrored else "False"
            rows = [r for r in rows if r["mirrored"] == want]
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        img = Image.open(os.path.join(self.img_dir, row["filename"]))
        img = crop_and_resize(img)
        if self.augment:
            img = augment_image(img, self.rng)
        arr = to_standardized_array(img)

        e_y, e_psi, kappa = normalize_targets(
            float(row["lateral_error"]), float(row["heading_error"]), float(row["curvature"])
        )
        target = torch.tensor([e_y, e_psi, kappa], dtype=torch.float32)
        valid = torch.tensor(1.0 if row["valid"] == "True" else 0.0, dtype=torch.float32)
        image = torch.from_numpy(arr)
        return image, target, valid
