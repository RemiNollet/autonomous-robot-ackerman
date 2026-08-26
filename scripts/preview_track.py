"""
Render a snapshot of the generated track scene and save it as a PNG.

Run this on the Mac, where offscreen rendering is already validated
(M0). This sandbox cannot render (no OpenGL context headless), so this
script is untested end-to-end — the MJCF has only been structurally
validated (loads correctly, 451 geoms, 1 body). This is the step that
checks what actually matters: do the markings show up with enough
contrast against the road for the CNN to have something to learn from.

Usage:
    python3 scripts/preview_track.py
    # writes track_preview.png in the current directory
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mujoco
from perception.dataset.track_definitions import REFERENCE_TRACK, LANE_HALF_WIDTH
from perception.dataset.track_mjcf import generate_track_mjcf


def main():
    xml = generate_track_mjcf(REFERENCE_TRACK, LANE_HALF_WIDTH)

    with open("track_generated.xml", "w") as f:
        f.write(xml)
    print("Wrote track_generated.xml")

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=720, width=1280)

    # Top-down free camera, no camera body defined in the track-only scene —
    # this previews the track geometry, not the actual onboard camera view
    # (that check comes once the track is merged into the vehicle MJCF).
    cam = mujoco.MjvCamera()
    cam.lookat = [
        (REFERENCE_TRACK.point_at(0)[0] + REFERENCE_TRACK.point_at(REFERENCE_TRACK.total_length / 2)[0]) / 2,
        (REFERENCE_TRACK.point_at(0)[1] + REFERENCE_TRACK.point_at(REFERENCE_TRACK.total_length / 2)[1]) / 2,
        0,
    ]
    cam.distance = 30
    cam.elevation = -75
    cam.azimuth = 90

    renderer.update_scene(data, camera=cam)
    pixels = renderer.render()

    from PIL import Image
    Image.fromarray(pixels).save("track_preview.png")
    print("Wrote track_preview.png — open it and check marking contrast/visibility.")


if __name__ == "__main__":
    main()
