"""
Image preprocessing shared between training and inference (NFR-8): crop,
resize, per-image standardize. Crop/resize geometry is read from
perception/dataset/cnn_input_config.json, not hard-coded here -- see that
file's comment and docs/decisions.md ADR-11 finding 7 for why the crop
rectangle needs a single source of truth.

Per-image standardization, not the dataset-level constants in
perception/dataset/normalization_stats.json: those stats were computed
before this crop existed (they're over the full 320x240 frame) and, more
importantly, MuJoCo's directional lighting makes the same scene read very
differently by vehicle heading alone -- verified directly in
tests/test_camera_visibility.py::test_project_to_pixel_matches_rendered_markers,
where one marker rendered anywhere from RGB (105, 0, 105) to (230, 5, 230)
depending only on which way the vehicle was facing. A single dataset-wide
mean/std would leave that lighting variation in the input; normalizing each
image against its own statistics removes it instead of asking the network
to learn around it.

Augmentation (perturb_image) is a separate function, applied by the caller
only for the train split -- see dataset.py. It perturbs pixels, never
geometry, so it never touches the labels.
"""

import json
import os

import numpy as np
from PIL import Image

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "cnn_input_config.json")


def _load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


_CONFIG = _load_config()
CROP = _CONFIG["crop"]
RESIZE_TO = _CONFIG["resize_to"]  # {"width": W, "height": H}


def crop_and_resize(img: Image.Image) -> Image.Image:
    """Crop to the configured rectangle, resize to the configured shape.
    Deterministic -- no randomness, safe to call at both train and
    inference time."""
    left, top = CROP["left"], CROP["top"]
    right, bottom = left + CROP["width"], top + CROP["height"]
    cropped = img.crop((left, top, right, bottom))
    return cropped.resize((RESIZE_TO["width"], RESIZE_TO["height"]), Image.BILINEAR)


def to_standardized_array(img: Image.Image) -> np.ndarray:
    """PIL image -> float32 array, shape (3, H, W), per-image standardized:
    (x - mean) / (std + 1e-6). Assumes img is already the target crop/resize
    shape -- call crop_and_resize first."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0  # (H, W, 3)
    mean = arr.mean()
    std = arr.std()
    arr = (arr - mean) / (std + 1e-6)
    return np.transpose(arr, (2, 0, 1))  # (3, H, W)


def preprocess(img: Image.Image) -> np.ndarray:
    """The full, deterministic train/inference-shared pipeline: crop,
    resize, per-image standardize. Returns (3, H, W) float32."""
    return to_standardized_array(crop_and_resize(img))


# ---------------------------------------------------------------------------
# Augmentation -- train split only, pixel perturbation only.
# ---------------------------------------------------------------------------

MAX_ERASING_PATCHES = 2
MAX_ERASING_AREA_FRAC = 0.15


def _jitter_brightness_contrast_gamma(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """arr: (H, W, 3) float in [0, 1]."""
    brightness = rng.uniform(0.8, 1.2)
    contrast = rng.uniform(0.8, 1.2)
    gamma = rng.uniform(0.8, 1.25)

    arr = arr * brightness
    mean = arr.mean()
    arr = (arr - mean) * contrast + mean
    arr = np.clip(arr, 0.0, 1.0)
    arr = np.power(arr, gamma)
    return np.clip(arr, 0.0, 1.0)


def _add_gaussian_noise(arr: np.ndarray, rng: np.random.Generator, sigma: float = 0.03) -> np.ndarray:
    noise = rng.normal(0.0, sigma, size=arr.shape).astype(np.float32)
    return np.clip(arr + noise, 0.0, 1.0)


def _random_erasing(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Up to MAX_ERASING_PATCHES rectangles, each under
    MAX_ERASING_AREA_FRAC of the image area, filled with uniform noise."""
    h, w = arr.shape[0], arr.shape[1]
    total_area = h * w
    n_patches = rng.integers(0, MAX_ERASING_PATCHES + 1)
    for _ in range(n_patches):
        patch_area = rng.uniform(0.02, MAX_ERASING_AREA_FRAC) * total_area
        aspect = rng.uniform(0.3, 3.3)
        patch_h = min(h, max(1, int(round((patch_area * aspect) ** 0.5))))
        patch_w = min(w, max(1, int(round((patch_area / aspect) ** 0.5))))
        top = rng.integers(0, h - patch_h + 1)
        left = rng.integers(0, w - patch_w + 1)
        arr[top:top + patch_h, left:left + patch_w, :] = rng.uniform(0.0, 1.0)
    return arr


def augment_image(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Applied to an already crop_and_resize'd image, BEFORE standardization.
    Pixel-space only -- no flip (mirrors already exist in the dataset,
    ADR-10, and a mirror pair shares a split, so flipping again would pair a
    sample with its own twin within the same split)."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    arr = _jitter_brightness_contrast_gamma(arr, rng)
    arr = _add_gaussian_noise(arr, rng)
    arr = _random_erasing(arr, rng)
    return Image.fromarray((arr * 255.0).astype(np.uint8))
