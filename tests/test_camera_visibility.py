"""
Tests for perception/dataset/camera_visibility.py.

The important one is test_matches_mujoco_extrinsics: the visibility filter
uses an analytic camera transform rather than calling MuJoCo, so that the
dataset generator stays headless and fast. That is only safe as long as the
analytic model provably agrees with the simulator. If cam_front's pose or
fovy changes in sim/models/car.xml, this test fails and the constants in
camera_visibility.py must be updated to match.

Skipped automatically when MuJoCo or the model file is unavailable.
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from perception.dataset.camera_visibility import (
    point_in_camera_frame, point_visible, lane_is_visible, any_lane_visible,
)
from perception.dataset.track_definitions import REFERENCE_TRACK, LANE_HALF_WIDTH

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "sim", "models", "car.xml")


@pytest.mark.skipif(
    not os.path.exists(MODEL_PATH), reason="vehicle MJCF not available"
)
def test_matches_mujoco_extrinsics():
    mujoco = pytest.importorskip("mujoco")
    import numpy as np

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    qadr = model.jnt_qposadr[model.joint("root").id]
    cam_id = model.camera("cam_front").id

    rng = np.random.default_rng(7)
    max_err = 0.0
    for _ in range(30):
        vx, vy = rng.uniform(-5, 5), rng.uniform(-5, 5)
        heading = rng.uniform(-math.pi, math.pi)
        data.qpos[qadr:qadr + 7] = [
            vx, vy, 0.075, math.cos(heading / 2), 0, 0, math.sin(heading / 2)
        ]
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)

        campos = np.array(data.cam_xpos[cam_id])
        cammat = np.array(data.cam_xmat[cam_id]).reshape(3, 3)

        p = np.array([rng.uniform(-8, 8), rng.uniform(-8, 8), 0.02])
        expected = cammat.T @ (p - campos)
        actual = np.array(point_in_camera_frame(p[0], p[1], p[2], vx, vy, heading))
        max_err = max(max_err, float(np.abs(expected - actual).max()))

    assert max_err < 1e-9, (
        f"analytic camera model diverged from MuJoCo by {max_err:.2e} m — "
        f"check CAM_* constants against sim/models/car.xml"
    )


def test_point_directly_ahead_is_visible():
    assert point_visible(3.0, 0.0, 0.02, 0.0, 0.0, 0.0)


def test_point_directly_behind_is_not_visible():
    assert not point_visible(-3.0, 0.0, 0.02, 0.0, 0.0, 0.0)


def test_lane_visible_from_track_start():
    x, y = REFERENCE_TRACK.point_at(0.0)
    h = REFERENCE_TRACK.heading_at(0.0)
    assert lane_is_visible(REFERENCE_TRACK, x, y, h, LANE_HALF_WIDTH)


def test_lane_not_visible_from_far_away_facing_out():
    """Far off the loop, pointing away from it."""
    assert not any_lane_visible(REFERENCE_TRACK, -30.0, -30.0, math.pi * 1.25,
                                 LANE_HALF_WIDTH)
