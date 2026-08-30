"""
Renders camera images for each pose in data/dataset_v0/labels.csv.

Run perception/dataset/generate_dataset.py FIRST (deterministic, headless,
already tested — see tests/test_generate_dataset.py). This script only
turns already-decided poses into pixels; it has no sampling logic of its
own on purpose, so a rendering bug can never be confused with a labeling
bug.

Mac-only: requires an OpenGL context for offscreen rendering (validated
since M0). Not runnable in the dev sandbox used to write this code.

Usage:
    python3 perception/dataset/generate_dataset.py   # writes labels.csv
    python3 perception/dataset/render_dataset_images.py
"""

import csv
import math
import os

import mujoco
import numpy as np
from PIL import Image

VEHICLE_XML = "sim/models/car.xml"
DATASET_DIR = "data/dataset_v0"
LABELS_CSV = f"{DATASET_DIR}/labels.csv"
IMG_DIR = f"{DATASET_DIR}/images"
IMG_WIDTH, IMG_HEIGHT = 320, 240   # matches the onboard camera / bridge protocol resolution


def quat_from_heading(heading: float):
    """MuJoCo quaternion convention: (w, x, y, z), rotation about world z.
    Matches perception/dataset/generate_dataset.py's convention exactly —
    if you ever change one, change the other."""
    return (math.cos(heading / 2.0), 0.0, 0.0, math.sin(heading / 2.0))


# Debug-only: inject bright, physically-impossible marker geoms into an
# already-updated MjvScene so a real MuJoCo render can be checked against
# camera_visibility.project_to_pixel (tests/test_camera_visibility.py,
# tools/visualize_camera_coverage.py). Visual only -- mjv_updateScene()
# resets scene.ngeom to the model's own geom count on every call, so markers
# never persist into a subsequent frame by themselves; this flag is a second,
# explicit guard so a debug run can never be mistaken for a real one. MUST
# stay False here -- these markers would appear in every training image.
DEBUG_INSERT_MARKERS = False


def insert_debug_marker(scene, pos, rgba, radius=0.02):
    """Append one sphere marker to an already-updated MjvScene. No physics,
    no effect on qpos or labels -- rendering-only, for camera-projection
    verification. Never call this from the real generation path (see
    DEBUG_INSERT_MARKERS above)."""
    i = scene.ngeom
    mujoco.mjv_initGeom(
        scene.geoms[i], type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([radius, 0, 0]), pos=np.array(pos, dtype=np.float64),
        mat=np.eye(3).flatten(), rgba=np.array(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def main():
    assert not DEBUG_INSERT_MARKERS, (
        "DEBUG_INSERT_MARKERS must be False for dataset generation -- "
        "markers would appear in every rendered training image"
    )
    os.makedirs(IMG_DIR, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(VEHICLE_XML)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=IMG_HEIGHT, width=IMG_WIDTH)

    # qpos[0:7] is the freejoint
    # current car.xml (see dev notes), but re-derive it defensively in
    # case the model changes later.
    joint_id = model.joint("root").id
    qadr = model.jnt_qposadr[joint_id]

    with open(LABELS_CSV) as f:
        rows = list(csv.DictReader(f))

    # mirror rows (see generate_dataset.mirror_row) are not re-rendered
    # through MuJoCo -- they're an exact horizontal flip of their source
    # image (cam_front has zero lateral offset, verified in
    # tests/test_generate_dataset.py), so the source must be rendered first.
    base_rows = [r for r in rows if r["mirrored"] != "True"]
    mirror_rows = [r for r in rows if r["mirrored"] == "True"]

    for i, row in enumerate(base_rows):
        x, y, heading = float(row["x"]), float(row["y"]), float(row["heading"])
        qw, qx, qy, qz = quat_from_heading(heading)

        data.qpos[qadr:qadr + 7] = [x, y, 0.075, qw, qx, qy, qz]
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)  # kinematics only, no physics stepping needed

        renderer.update_scene(data, camera="cam_front")
        pixels = renderer.render()
        Image.fromarray(pixels).save(os.path.join(IMG_DIR, row["filename"]))

        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(base_rows)} rendered")

    print(f"Rendered {len(base_rows)} images -> {IMG_DIR}")

    for i, row in enumerate(mirror_rows):
        src_path = os.path.join(IMG_DIR, row["source_filename"])
        mirrored = Image.open(src_path).transpose(Image.FLIP_LEFT_RIGHT)
        mirrored.save(os.path.join(IMG_DIR, row["filename"]))

        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(mirror_rows)} mirrored")

    print(f"Mirrored {len(mirror_rows)} images -> {IMG_DIR}")
    print(f"Done: {len(rows)} images written to {IMG_DIR}")


if __name__ == "__main__":
    main()
