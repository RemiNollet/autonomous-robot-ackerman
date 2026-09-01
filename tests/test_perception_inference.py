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
# Second entry: carsim_bridge is a nested ament_python package
# (carsim_bridge/carsim_bridge/), so `import carsim_bridge.X` needs the
# outer carsim_bridge/ directory on sys.path too, not just the repo root
# that resolves `perception.*` above.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "carsim_bridge"))

torch = pytest.importorskip("torch")

from carsim_bridge import protocol as P
from carsim_bridge.perception_inference import (
    distribution_stats, ros_image_to_pil, run_inference,
)
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


def test_wire_protocol_roundtrip_is_byte_for_byte_identical_to_direct_load():
    """The two tests above build BOTH branches from the same already-opened
    PIL object -- 'raw bytes' there is really 'this object's own pixels,
    re-serialized', so a PNG-decode quirk affecting both equally would stay
    invisible. This drives the actual carsim_bridge/protocol.py wire
    functions end to end -- the exact JSON-header-plus-raw-buffer encode/
    decode sim_server.py and bridge_node.py use in production, not a
    hand-rolled equivalent -- so the two branches share no in-memory state:
    only a numpy array goes in on one side, only bytes come out the other."""
    img = _make_test_image()
    direct = preprocess(img)

    arr = np.asarray(img, dtype=np.uint8)
    frames = P.encode_state(seq=0, t_sim=0.0, pose=(0.0, 0.0, 0.0),
                             twist=(0.0, 0.0, 0.0), img=arr)
    header, payload = P.decode_state(frames)
    via_wire = preprocess(ros_image_to_pil(
        width=header["img"]["w"], height=header["img"]["h"],
        encoding=header["img"]["encoding"], data=payload))

    assert np.array_equal(direct, via_wire)


@pytest.mark.skipif(not os.path.exists(SAMPLE_IMAGE), reason="dataset not generated locally")
def test_wire_protocol_roundtrip_matches_a_real_dataset_image():
    """Same independence guarantee as the synthetic version above, on an
    actual rendered dataset PNG -- two separate PILImage.open calls, so
    the 'direct' and 'via_wire' branches don't even share the decoded PNG
    object, only the file path."""
    direct = preprocess(PILImage.open(SAMPLE_IMAGE).convert("RGB"))

    arr = np.asarray(PILImage.open(SAMPLE_IMAGE).convert("RGB"), dtype=np.uint8)
    frames = P.encode_state(seq=0, t_sim=0.0, pose=(0.0, 0.0, 0.0),
                             twist=(0.0, 0.0, 0.0), img=arr)
    header, payload = P.decode_state(frames)
    via_wire = preprocess(ros_image_to_pil(
        width=header["img"]["w"], height=header["img"]["h"],
        encoding=header["img"]["encoding"], data=payload))

    assert np.array_equal(direct, via_wire)


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


def test_distribution_stats_empty_is_none():
    assert distribution_stats([]) is None


def test_distribution_stats_matches_hand_computed_values():
    # 1..100 ms in seconds: mean 50.5, p50 (numpy linear interp, 50th
    # percentile of 1..100) 50.5, p95 95.05, max 100, std = pstdev(1..100).
    samples_s = [i / 1000.0 for i in range(1, 101)]
    stats = distribution_stats(samples_s, scale=1000.0)

    assert stats["n"] == 100
    assert stats["mean"] == pytest.approx(50.5)
    assert stats["p50"] == pytest.approx(50.5)
    assert stats["p95"] == pytest.approx(95.05)
    assert stats["max"] == pytest.approx(100.0)
    assert stats["std"] == pytest.approx(np.std(np.arange(1, 101)))
