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
    point_in_camera_frame, point_visible, project_to_pixel,
    lane_is_visible, any_lane_visible, IMG_WIDTH, IMG_HEIGHT,
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


# ---------------------------------------------------------------------------
# Part B: project_to_pixel vs MuJoCo's own scene-camera projection.
#
# test_matches_mujoco_extrinsics (above) verifies the camera EXTRINSICS
# (point_in_camera_frame) against MuJoCo's cam_xpos/cam_xmat. It says nothing
# about the PROJECTION: whether fovy was interpreted correctly (full vs half
# angle, degrees vs radians), whether the aspect-ratio-derived horizontal FOV
# is right, or whether the NDC->viewport pixel mapping is right. This is the
# ADR-8 failure mode again: internally self-consistent arithmetic that never
# had anything external to check it against.
#
# This test builds the expected (u, v) via a completely independent path —
# MuJoCo's own mjvGLCamera frustum (frustum_top/frustum_near for the
# vertical half-angle, the same aspect-derived horizontal extent convention,
# and cam_xpos/cam_xmat for the view transform) — and never calls
# point_in_camera_frame or project_to_pixel on the "expected" side. If this
# disagrees with project_to_pixel, project_to_pixel is wrong, not this test.
# ---------------------------------------------------------------------------

def test_project_to_pixel_matches_mujoco_frustum():
    mujoco = pytest.importorskip("mujoco")
    import numpy as np

    if not os.path.exists(MODEL_PATH):
        pytest.skip("vehicle MJCF not available")

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    qadr = model.jnt_qposadr[model.joint("root").id]
    cam_id = model.camera("cam_front").id

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = cam_id
    opt = mujoco.MjvOption()
    scene = mujoco.MjvScene(model, maxgeom=1000)  # comfortably above the model's own geom count

    rng = np.random.default_rng(11)
    max_err = np.array([0.0, 0.0])
    n_compared = 0

    for _ in range(15):  # N poses
        vx, vy = rng.uniform(-5, 5), rng.uniform(-5, 5)
        heading = rng.uniform(-math.pi, math.pi)
        data.qpos[qadr:qadr + 7] = [
            vx, vy, 0.075, math.cos(heading / 2), 0, 0, math.sin(heading / 2)
        ]
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)

        campos = np.array(data.cam_xpos[cam_id])
        cammat = np.array(data.cam_xmat[cam_id]).reshape(3, 3)

        # Independent vertical half-FOV: derived from MuJoCo's own rendered
        # frustum, not from CAM_FOVY_DEG in camera_visibility.py.
        mujoco.mjv_updateScene(model, data, opt, None, cam,
                                mujoco.mjtCatBit.mjCAT_ALL.value, scene)
        gc = scene.camera[0]
        tan_half_fovy = gc.frustum_top / gc.frustum_near
        # Horizontal extent: the standard aspect-ratio-scaled convention.
        # mjvGLCamera exposes no separate left/right, so this convention
        # itself is what Part C's rendered-marker test closes the loop on.
        tan_half_fovx = tan_half_fovy * (IMG_WIDTH / IMG_HEIGHT)

        ch, sh = math.cos(heading), math.sin(heading)
        for _ in range(8):  # M points per pose
            dist = rng.uniform(0.3, 6.0)
            lateral = rng.uniform(-2.0, 2.0)
            fx = vx + dist * ch - lateral * sh
            fy = vy + dist * sh + lateral * ch
            fz = rng.uniform(0.0, 0.1)

            p = np.array([fx, fy, fz])
            p_cam = cammat.T @ (p - campos)
            depth = -p_cam[2]
            if depth <= 1e-6:
                continue
            ndc_x = p_cam[0] / (depth * tan_half_fovx)
            ndc_y = p_cam[1] / (depth * tan_half_fovy)
            u_expected = (ndc_x + 1.0) * 0.5 * IMG_WIDTH
            v_expected = (1.0 - ndc_y) * 0.5 * IMG_HEIGHT

            u, v, _, _ = project_to_pixel(fx, fy, fz, vx, vy, heading)
            if u is None:
                continue
            n_compared += 1
            max_err = np.maximum(max_err, np.abs([u - u_expected, v - v_expected]))

    assert n_compared > 50, f"too few in-frustum comparison points: {n_compared}"
    assert max_err.max() < 0.5, (
        f"project_to_pixel diverged from MuJoCo's own frustum projection by "
        f"{max_err} px (u, v) — check fovy interpretation / aspect handling "
        f"/ NDC->viewport mapping in project_to_pixel"
    )


# ---------------------------------------------------------------------------
# Part C: project_to_pixel vs an actual rendered image.
#
# Stronger than Part B: agreeing with MuJoCo's frustum math is not the same
# as agreeing with what MuJoCo actually draws (e.g. a flipped image row
# convention would pass B's abstract math and still be wrong on every real
# frame). Bright marker geoms are injected into the scene at known world
# points (visual only, via render_dataset_images.insert_debug_marker — never
# touches the model, never affects labels), the frame is rendered for real,
# and the marker's rendered centroid is found by colour thresholding and
# compared against project_to_pixel.
# ---------------------------------------------------------------------------

def test_project_to_pixel_matches_rendered_markers():
    mujoco = pytest.importorskip("mujoco")
    import numpy as np
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from perception.dataset import render_dataset_images as rdi

    if not os.path.exists(MODEL_PATH):
        pytest.skip("vehicle MJCF not available")
    assert not rdi.DEBUG_INSERT_MARKERS, "must stay off outside this test"

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=IMG_HEIGHT, width=IMG_WIDTH)
    qadr = model.jnt_qposadr[model.joint("root").id]

    MARKER_RGBA = (1.0, 0.0, 1.0, 1.0)  # magenta — not used elsewhere in the scene

    def measure_centroid(pixels):
        # Hue-ratio test, not an absolute-brightness threshold: MuJoCo shades
        # the marker with the scene's directional light, so its apparent
        # brightness swings a lot with vehicle heading (a fixed magenta
        # material rendered as RGB (105, 0, 105) at one heading and (230, 5,
        # 230) at another) -- a brightness cutoff either misses the marker
        # entirely or catches only its brightest sliver, which biases the
        # centroid by several pixels. R~=B with G much smaller than both
        # holds regardless of shading intensity.
        r = pixels[:, :, 0].astype(int)
        g = pixels[:, :, 1].astype(int)
        b = pixels[:, :, 2].astype(int)
        mask = (r > 20) & (b > 20) & (np.abs(r - b) < 20) & (g < 0.4 * np.minimum(r, b) + 5)
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None
        return float(xs.mean()), float(ys.mean()), int(mask.sum())

    # Poses on REFERENCE_TRACK (straight, both arc radii), each with a small
    # lateral offset -- keeps the camera looking down open track/ground
    # rather than at arbitrary world coordinates, which can face a wall or
    # other scene geometry close enough to occlude every marker.
    poses = []
    for s in (1.0, 10.0, 20.0, 35.0, 42.0):
        cx, cy = REFERENCE_TRACK.point_at(s)
        h = REFERENCE_TRACK.heading_at(s)
        nx, ny = -math.sin(h), math.cos(h)
        poses.append((cx + 0.1 * nx, cy + 0.1 * ny, h))
    # (forward_dist, lateral_offset) relative to the vehicle, ground height.
    offsets = [(1.0, 0.0), (1.5, 0.3), (1.5, -0.3), (2.5, 0.2), (1.0, -0.2)]

    n_compared = 0
    max_err = 0.0
    for vx, vy, heading in poses:
        qw, qx, qy, qz = rdi.quat_from_heading(heading)
        data.qpos[qadr:qadr + 7] = [vx, vy, 0.075, qw, qx, qy, qz]
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)

        ch, sh = math.cos(heading), math.sin(heading)
        for dist, lateral in offsets:
            wx = vx + dist * ch - lateral * sh
            wy = vy + dist * sh + lateral * ch
            wz = 0.05

            u_pred, v_pred, depth, in_frame = project_to_pixel(wx, wy, wz, vx, vy, heading)
            if not in_frame or depth < 0.3:
                continue  # keep clear of the near plane / frame edge

            renderer.update_scene(data, camera="cam_front")
            rdi.insert_debug_marker(renderer.scene, (wx, wy, wz), MARKER_RGBA, radius=0.02)
            pixels = renderer.render()

            measured = measure_centroid(pixels)
            if measured is None:
                continue  # marker too close to the frame edge to register
            u_meas, v_meas, n_px = measured
            assert n_px < 500, "magenta mask too large — threshold picked up more than the marker"

            err = math.hypot(u_meas - u_pred, v_meas - v_pred)
            max_err = max(max_err, err)
            n_compared += 1

    assert n_compared >= 8, f"too few markers registered in-frame: {n_compared}"
    assert max_err < 2.0, (
        f"project_to_pixel diverged from the rendered marker centroid by "
        f"{max_err:.2f} px — check the row-flip convention (v measured from "
        f"top) or the aspect-derived horizontal FOV against what MuJoCo "
        f"actually rasterizes"
    )
