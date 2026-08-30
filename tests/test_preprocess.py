"""
Tests for perception/model/preprocess.py.
"""

import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from perception.model.preprocess import (
    crop_and_resize, to_standardized_array, preprocess, augment_image, RESIZE_TO, CROP,
)

SAMPLE_IMAGE = os.path.join(os.path.dirname(__file__), "..", "data", "dataset_v0", "images", "img_00000.png")


def _make_test_image() -> Image.Image:
    """A synthetic 320x240 image, independent of whether the (git-ignored,
    locally-generated) dataset exists on this machine."""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_crop_and_resize_produces_configured_shape():
    img = _make_test_image()
    out = crop_and_resize(img)
    assert out.size == (RESIZE_TO["width"], RESIZE_TO["height"])


def test_preprocess_produces_stated_shape_and_dtype():
    img = _make_test_image()
    arr = preprocess(img)
    assert arr.shape == (3, RESIZE_TO["height"], RESIZE_TO["width"])
    assert arr.dtype == np.float32


def test_preprocess_is_deterministic():
    img = _make_test_image()
    a = preprocess(img)
    b = preprocess(img)
    assert np.array_equal(a, b)


def test_standardized_array_has_zero_mean_unit_std():
    img = _make_test_image()
    arr = to_standardized_array(crop_and_resize(img))
    assert abs(arr.mean()) < 1e-4
    assert abs(arr.std() - 1.0) < 1e-3


def test_augment_image_is_stochastic_but_shape_preserving():
    img = crop_and_resize(_make_test_image())
    rng_a = np.random.default_rng(0)
    rng_b = np.random.default_rng(0)
    rng_c = np.random.default_rng(1)

    a = augment_image(img, rng_a)
    b = augment_image(img, rng_b)
    c = augment_image(img, rng_c)

    assert a.size == img.size == c.size
    assert np.array_equal(np.asarray(a), np.asarray(b)), "same seed must reproduce the same augmentation"
    assert not np.array_equal(np.asarray(a), np.asarray(c)), "different seeds must differ"


@pytest.mark.skipif(not os.path.exists(SAMPLE_IMAGE), reason="dataset not generated locally")
def test_preprocess_on_a_real_dataset_image():
    img = Image.open(SAMPLE_IMAGE)
    arr = preprocess(img)
    assert arr.shape == (3, RESIZE_TO["height"], RESIZE_TO["width"])
    assert np.isfinite(arr).all()
