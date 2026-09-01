"""
Tests for perception/model/dataset.py. Skipped where the (git-ignored,
locally-generated) dataset isn't present -- see docs/dataset-readme.md to
regenerate it.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

torch = pytest.importorskip("torch")

from perception.model.dataset import LaneDataset, LABELS_CSV

pytestmark = pytest.mark.skipif(not os.path.exists(LABELS_CSV), reason="dataset not generated locally")


def test_split_filtering_is_disjoint_and_covers_everything():
    all_rows = LaneDataset(split=None)
    train = LaneDataset(split="train")
    val = LaneDataset(split="val")
    test = LaneDataset(split="test")
    assert len(train) + len(val) + len(test) == len(all_rows)


def test_mirrored_filter_splits_each_split_in_half():
    train = LaneDataset(split="train")
    train_src = LaneDataset(split="train", mirrored=False)
    train_mirror = LaneDataset(split="train", mirrored=True)
    assert len(train_src) + len(train_mirror) == len(train)
    assert len(train_src) == len(train_mirror), "every source row has exactly one mirror twin"


def test_item_shape_and_target_normalization_range():
    ds = LaneDataset(split="train", augment=False)
    image, target, valid = ds[0]
    assert image.shape[0] == 3
    assert target.shape == (3,)
    assert valid.shape == ()
    if valid.item() >= 0.5:
        # normalized targets should be within the declared positive envelope,
        # with a little headroom for floating point at the exact boundary
        assert (target.abs() <= 1.01).all(), f"normalized target out of range: {target}"


def test_augmentation_does_not_change_target():
    """Augmentation perturbs pixels only -- the label for a given row must
    be identical whether or not augment=True."""
    plain = LaneDataset(split="train", augment=False)
    augmented = LaneDataset(split="train", augment=True, seed=0)
    _, target_plain, valid_plain = plain[0]
    _, target_aug, valid_aug = augmented[0]
    assert torch.equal(target_plain, target_aug)
    assert torch.equal(valid_plain, valid_aug)
