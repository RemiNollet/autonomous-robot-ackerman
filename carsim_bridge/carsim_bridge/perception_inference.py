"""
Inference logic for perception_node.py, deliberately independent of rclpy/
ROS2 message types -- this project's ROS2 stack only exists in the VM
(docs/decisions.md ADR-1), so keeping the model/preprocessing path free of
that dependency is what makes it testable on any machine, not only where a
full ROS2 graph is buildable. perception_node.py is the thin rclpy wrapper
around this module; it should not contain logic that needs testing on its
own.

Preprocessing is imported directly from perception/model/preprocess.py, not
reimplemented -- a second copy would drift from the training path, and a
train/inference preprocessing mismatch is silent and very hard to diagnose
(NFR-8).
"""
import os
import sys
import time

import numpy as np
import torch
from PIL import Image as PILImage

# Repo root on sys.path so `perception.*` is importable regardless of how
# this is launched -- matches the pattern perception/model/train.py already
# uses for its own cross-package imports. "../.." because this file lives
# two levels under the repo root (carsim_bridge/carsim_bridge/), the
# standard nested ament_python layout.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import perception  # noqa: E402
from perception.model.lane_cnn import LaneCNN  # noqa: E402
from perception.model.preprocess import preprocess  # noqa: E402
from perception.model.targets import denormalize_targets  # noqa: E402

# Derived from perception.__file__ (wherever Python actually resolved the
# import above), not a __file__-relative walk from this file's own location.
# The walk was fragile in exactly the way this project's build system
# exposed twice: it assumed a fixed nesting depth between carsim_bridge and
# perception/, which changed under the ament_python layout refactor and
# would silently resolve to the wrong place again under any future
# repackaging -- whereas perception.__file__ is correct by construction
# under any sys.path/PYTHONPATH setup that got the import to succeed at all.
DEFAULT_CHECKPOINT = os.path.join(
    os.path.dirname(perception.__file__), "model", "checkpoints",
    "lane_cnn_width1.0_best.pt",
)


def distribution_stats(samples_s, scale=1000.0):
    """seconds -> {n, mean, p50, p95, max, std} in the scaled unit (default
    ms). np.percentile, matching perception/model/physical_metrics.py's
    _stats so VM-side telemetry and dataset-side evaluation report
    percentiles the same way. None on an empty input, not a divide-by-zero
    -- a report over 0 samples has nothing to summarize, and callers (e.g.
    the publish-interval series, which has one fewer sample than the frame
    count) need to tell that apart from a real all-zero series."""
    if not samples_s:
        return None
    a = np.asarray(samples_s, dtype=np.float64) * scale
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
        "std": float(a.std()),
    }


def ros_image_to_pil(width: int, height: int, encoding: str, data) -> PILImage.Image:
    """sensor_msgs/Image fields -> PIL Image. Matches carsim_bridge/
    protocol.py's encode_state: packed HxWx3 uint8, encoding="rgb8",
    step=width*3. Takes raw bytes rather than a constructed Image message so
    it has no rclpy/sensor_msgs dependency and is testable without one."""
    if encoding != "rgb8":
        raise ValueError(f"perception_node expects rgb8, got encoding={encoding!r}")
    arr = np.frombuffer(bytes(data), dtype=np.uint8).reshape(height, width, 3)
    return PILImage.fromarray(arr, mode="RGB")


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """torch.load() already raises FileNotFoundError on a missing path, but
    that's an accidental property of the underlying call, not a documented
    contract of this function -- a perception node without its trained
    weights must refuse to start, not silently fall back to a freshly
    initialised (i.e. random) model, so the check is made explicit here
    with a message that says what to do about it, rather than relying on
    torch's own error surviving unchanged across future refactors."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_path!r}. perception_node refuses "
            "to start without trained weights rather than publish /lane_state "
            "from a randomly-initialised model. Train it on the Mac and "
            "transfer it (scp) -- see perception/README.md's Deployment note."
        )
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = LaneCNN(width_mult=ckpt["width_mult"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def run_inference(model: torch.nn.Module, pil_img: PILImage.Image, device: torch.device):
    """Preprocess + forward pass, timed separately -- the assumption going in
    was that preprocessing would dominate on a model this small, which
    matters for the embedded budget; measured Mac-CPU numbers contradicted
    it (forward ~1.1 ms vs preprocess ~0.26 ms, perception/README.md
    Latency), so the two stages stay timed separately rather than folded
    into one number. Returns (e_y, e_psi, confidence, t_preprocess_s,
    t_forward_s) in physical units / [0,1] -- kappa is deliberately not
    returned; the untrained head's output must never reach a caller
    (docs/decisions.md ADR-12)."""
    t0 = time.perf_counter()
    arr = preprocess(pil_img)  # (3, H, W) float32
    t1 = time.perf_counter()

    with torch.no_grad():
        x = torch.from_numpy(arr).unsqueeze(0).to(device)
        pred, _ = model(x)
    t2 = time.perf_counter()

    e_y_n, e_psi_n = pred[0, 0].item(), pred[0, 1].item()
    e_y, e_psi, _ = denormalize_targets(e_y_n, e_psi_n, 0.0)
    confidence = torch.sigmoid(pred[0, 3]).item()

    return e_y, e_psi, confidence, (t1 - t0), (t2 - t1)
