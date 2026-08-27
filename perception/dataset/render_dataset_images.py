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


def main():
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

    for i, row in enumerate(rows):
        x, y, heading = float(row["x"]), float(row["y"]), float(row["heading"])
        qw, qx, qy, qz = quat_from_heading(heading)

        data.qpos[qadr:qadr + 7] = [x, y, 0.075, qw, qx, qy, qz]
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)  # kinematics only, no physics stepping needed

        renderer.update_scene(data, camera="cam_front")
        pixels = renderer.render()
        Image.fromarray(pixels).save(os.path.join(IMG_DIR, row["filename"]))

        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(rows)} rendered")

    print(f"Done: {len(rows)} images written to {IMG_DIR}")


if __name__ == "__main__":
    main()
