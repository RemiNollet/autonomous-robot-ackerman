"""
Tests for carsim_bridge/perception_inference.py -- the rclpy-independent
half of perception_node.py. See that module's docstring for why the split
exists: this project's ROS2 stack only runs in the VM (docs/decisions.md
ADR-1), so keeping the model/preprocessing logic free of an rclpy
dependency is what makes it testable here, on any machine, rather than only
where a full ROS2 graph is buildable. Node-level tests (construction,
publishing) are in tests/test_perception_node.py and are VM-only.
"""

import os
import sys

import numpy as np
import pytest
from PIL import Image as PILImage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

torch = pytest.importorskip("torch")

from carsim_bridge.perception_inference import ros_image_to_pil, run_inference
from perception.model.lane_cnn import LaneCNN
from perception.model.preprocess import preprocess

SAMPLE_IMAGE = os.path.join(os.path.dirname(__file__), "..", "data", "dataset_v0",
                             "images", "img_00000.png")


def _make_test_image():
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
    return PILImage.fromarray(arr, mode="RGB")


def test_ros_image_to_pil_roundtrips_pixels():
    img = _make_test_image()
    raw = np.asarray(img, dtype=np.uint8).tobytes()
    recovered = ros_image_to_pil(width=320, height=240, encoding="rgb8", data=raw)
    assert np.array_equal(np.asarray(img), np.asarray(recovered))


def test_ros_image_to_pil_rejects_wrong_encoding():
    with pytest.raises(ValueError):
        ros_image_to_pil(width=320, height=240, encoding="bgr8", data=b"\x00" * (320 * 240 * 3))


def test_preprocessing_is_byte_for_byte_identical_to_training_path():
    """The specific guarantee the task requires: an image that goes through
    the node's ROS-message-decoding path and then preprocess.preprocess()
    must produce EXACTLY the same array as loading the same pixels directly
    through the training dataset's path -- not close, not correlated,
    identical, since this is what makes the crop/resize/standardization
    genuinely shared code rather than two implementations that happen to
    agree on today's test image."""
    img = _make_test_image()
    direct = preprocess(img)

    raw_bytes = np.asarray(img, dtype=np.uint8).tobytes()
    via_node = preprocess(ros_image_to_pil(width=320, height=240, encoding="rgb8", data=raw_bytes))

    assert np.array_equal(direct, via_node)


@pytest.mark.skipif(not os.path.exists(SAMPLE_IMAGE), reason="dataset not generated locally")
def test_preprocessing_matches_on_a_real_dataset_image():
    img = PILImage.open(SAMPLE_IMAGE)
    direct = preprocess(img)
    raw_bytes = np.asarray(img.convert("RGB"), dtype=np.uint8).tobytes()
    via_node = preprocess(ros_image_to_pil(width=img.width, height=img.height,
                                            encoding="rgb8", data=raw_bytes))
    assert np.array_equal(direct, via_node)


def test_run_inference_shapes_and_finiteness():
    """Doesn't need a trained checkpoint -- a freshly-initialized model is
    enough to check run_inference's contract: three finite floats out,
    confidence in [0, 1], and that it returns separate preprocess/forward
    timings (both >= 0) rather than a single combined number."""
    torch.manual_seed(0)
    model = LaneCNN(width_mult=1.0)
    model.eval()
    img = _make_test_image()

    e_y, e_psi, confidence, t_pre, t_fwd = run_inference(model, img, torch.device("cpu"))

    assert all(np.isfinite(v) for v in (e_y, e_psi, confidence))
    assert 0.0 <= confidence <= 1.0
    assert t_pre >= 0.0 and t_fwd >= 0.0
