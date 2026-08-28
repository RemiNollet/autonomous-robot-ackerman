"""
Contact sheet: render actual camera frames and overlay what the visibility
filter (perception/dataset/camera_visibility.py) computes, so a human can
catch a filter that is wrong even though every unit test passes -- the same
failure mode ADR-8 documents (internally consistent maths, wrong geometry,
invisible without looking).

For each pose: lane boundary candidate points ahead, colour-coded by whether
point_visible() accepted them; the centreline; the horizon line; distance
ticks along the centreline; the CNN input crop rectangle (from
perception/dataset/cnn_input_config.json, not hard-coded here); and a text
verdict (visible-point count vs MIN_VISIBLE_POINTS, accept/reject).

Usage:
    python3 tools/visualize_camera_coverage.py
        -> docs/dataset/camera_coverage_contact_sheet.png, ~24 curated poses

    python3 tools/visualize_camera_coverage.py --pose 2.0 0.1 0.0
        -> renders and annotates a single pose to /tmp, opens nothing

    python3 tools/visualize_camera_coverage.py --from-dataset data/dataset_v0/labels.csv --n 24
        -> samples N poses from an existing dataset record file instead
"""

import argparse
import csv
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from perception.dataset.camera_visibility import (
    project_to_pixel,
    LOOKAHEAD_M, LOOKAHEAD_SAMPLES, MIN_VISIBLE_POINTS, IMG_WIDTH, IMG_HEIGHT,
)
from perception.dataset.track_definitions import REFERENCE_TRACK, LANE_HALF_WIDTH
from perception.dataset.generate_dataset import POS_LATERAL_RANGE, POS_HEADING_RANGE
from perception.dataset.render_dataset_images import VEHICLE_XML, quat_from_heading

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "perception", "dataset",
                            "cnn_input_config.json")
OUT_DIR = "docs/dataset"
OUT_PNG = f"{OUT_DIR}/camera_coverage_contact_sheet.png"
DISTANCE_TICKS_M = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0]

GREEN = (60, 220, 60)
RED = (230, 50, 50)
BLUE = (70, 140, 255)
YELLOW = (240, 210, 40)
WHITE = (255, 255, 255)


def load_crop_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def horizon_row(vehicle_x, vehicle_y, vehicle_heading):
    """Image row where a horizontal (zero-elevation) ray lands: the vanishing
    row for the direction of travel, independent of distance -- take a point
    far ahead at the camera's own height and project it; camera height
    cancels depth as distance -> infinity, so any large distance converges to
    the same row."""
    ch, sh = math.cos(vehicle_heading), math.sin(vehicle_heading)
    far = 5000.0
    fx = vehicle_x + far * ch
    fy = vehicle_y + far * sh
    from perception.dataset.camera_visibility import CAM_HEIGHT
    _, v, _, _ = project_to_pixel(fx, fy, CAM_HEIGHT, vehicle_x, vehicle_y, vehicle_heading)
    return v


def build_contact_sheet_poses():
    """~24 poses spanning the sampling envelope: straight, R=3m arc, R=5m arc
    primitives, each at envelope centre / lateral edges / heading edges / a
    near-rejection combo, plus a few realistic negatives and random-envelope
    samples for variety."""
    poses = []
    # (label, s at primitive midpoint, primitive curvature label)
    groups = [
        ("straight", 2.0),
        ("R=3m arc", 4.0 + 4.712 / 2),
        ("R=5m arc", 14.712 + 7.854 / 2),
    ]

    def pose_at(s, lateral, heading_off, label):
        cx, cy = REFERENCE_TRACK.point_at(s)
        h = REFERENCE_TRACK.heading_at(s)
        nx, ny = -math.sin(h), math.cos(h)
        x, y = cx + lateral * nx, cy + lateral * ny
        return (label, x, y, h + heading_off)

    for name, s in groups:
        poses.append(pose_at(s, 0.0, 0.0, f"{name} / centre"))
        poses.append(pose_at(s, +POS_LATERAL_RANGE, 0.0, f"{name} / lat+"))
        poses.append(pose_at(s, -POS_LATERAL_RANGE, 0.0, f"{name} / lat-"))
        poses.append(pose_at(s, 0.0, +POS_HEADING_RANGE, f"{name} / head+"))
        poses.append(pose_at(s, 0.0, -POS_HEADING_RANGE, f"{name} / head-"))
        # Genuine near-boundary pair, not just "0.9x the declared envelope"
        # (which turned out nowhere near the actual MIN_VISIBLE_POINTS cliff
        # -- see the resolvability/findings report). The visible-point count
        # does not decay gently with heading offset; it collapses sharply
        # around ~2.5-3.0x POS_HEADING_RANGE regardless of curvature group,
        # found by sweeping count_visible_lane_points directly.
        poses.append(pose_at(s, 0.0, 2.5 * POS_HEADING_RANGE, f"{name} / near-accept"))
        poses.append(pose_at(s, 0.0, 2.9 * POS_HEADING_RANGE, f"{name} / near-reject"))

    # Realistic negatives: reuse the actual rejection-sampled generator so
    # these are genuine dataset negatives, not hand-picked strawmen.
    from perception.dataset.generate_dataset import sample_pose
    rng = np.random.default_rng(2024)
    for i in range(3):
        x, y, heading, _ = sample_pose(rng, negative=True)
        poses.append((f"negative #{i+1}", x, y, heading))

    # Random-envelope positives for variety, same generator as real data.
    for i in range(3):
        x, y, heading, _ = sample_pose(rng, negative=False)
        poses.append((f"random positive #{i+1}", x, y, heading))

    return poses


def render_pose(model, data, renderer, qadr, x, y, heading):
    qw, qx, qy, qz = quat_from_heading(heading)
    data.qpos[qadr:qadr + 7] = [x, y, 0.075, qw, qx, qy, qz]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="cam_front")
    return renderer.render()


MARGIN = 40  # px of canvas beyond the true 320x240 frame, so a rejected
             # (out-of-frame) candidate that just missed can still be drawn
             # in red near the edge instead of being invisible -- point_visible
             # IS the in-frame test, so without this margin every point that
             # gets drawn at all is green by construction and "colour-coded
             # by whether point_visible accepted them" would be a no-op.


def annotate(pixels, x, y, heading, crop_cfg):
    base = Image.fromarray(pixels).convert("RGB")
    img = Image.new("RGB", (IMG_WIDTH + 2 * MARGIN, IMG_HEIGHT + 2 * MARGIN), (15, 15, 15))
    img.paste(base, (MARGIN, MARGIN))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    def to_canvas(u, v):
        return u + MARGIN, v + MARGIN

    def in_margin_canvas(u, v):
        return -MARGIN <= u <= IMG_WIDTH + MARGIN and -MARGIN <= v <= IMG_HEIGHT + MARGIN

    # True camera-frame boundary, so the margin area reads as "outside the
    # real image" rather than more valid frame.
    fx0, fy0 = to_canvas(0, 0)
    fx1, fy1 = to_canvas(IMG_WIDTH, IMG_HEIGHT)
    draw.rectangle([fx0, fy0, fx1, fy1], outline=(120, 120, 120), width=1)

    s0 = REFERENCE_TRACK.project(x, y)

    # Lane boundary candidates, colour-coded by point_visible -- exactly the
    # points count_visible_lane_points sums over. Drawn even when just
    # outside the true frame (within MARGIN) so a near-miss reject is
    # visible, not merely absent.
    n_visible = 0
    n_candidates = 0
    for i in range(LOOKAHEAD_SAMPLES):
        s = (s0 + LOOKAHEAD_M * i / LOOKAHEAD_SAMPLES) % REFERENCE_TRACK.total_length
        cx, cy = REFERENCE_TRACK.point_at(s)
        h = REFERENCE_TRACK.heading_at(s)
        nx, ny = -math.sin(h), math.cos(h)
        for side in (+1.0, -1.0):
            bx = cx + side * LANE_HALF_WIDTH * nx
            by = cy + side * LANE_HALF_WIDTH * ny
            n_candidates += 1
            u, v, depth, in_frame = project_to_pixel(bx, by, 0.02, x, y, heading)
            if in_frame:
                n_visible += 1
            if u is None or not in_margin_canvas(u, v):
                continue
            cu, cv = to_canvas(u, v)
            colour = GREEN if in_frame else RED
            r = 2
            draw.ellipse([cu - r, cv - r, cu + r, cv + r], fill=colour)

    # Centreline, thin blue dots.
    for i in range(LOOKAHEAD_SAMPLES):
        s = (s0 + LOOKAHEAD_M * i / LOOKAHEAD_SAMPLES) % REFERENCE_TRACK.total_length
        cx, cy = REFERENCE_TRACK.point_at(s)
        u, v, depth, in_frame = project_to_pixel(cx, cy, 0.02, x, y, heading)
        if u is not None and in_margin_canvas(u, v):
            cu, cv = to_canvas(u, v)
            draw.point([cu, cv], fill=BLUE)

    # Horizon line, full canvas width.
    hv = horizon_row(x, y, heading)
    if hv is not None and -MARGIN <= hv <= IMG_HEIGHT + MARGIN:
        _, chv = to_canvas(0, hv)
        draw.line([(0, chv), (img.width, chv)], fill=WHITE, width=1)

    # Distance ticks along the centreline.
    for d in DISTANCE_TICKS_M:
        s = (s0 + d) % REFERENCE_TRACK.total_length
        cx, cy = REFERENCE_TRACK.point_at(s)
        u, v, depth, in_frame = project_to_pixel(cx, cy, 0.02, x, y, heading)
        if u is None or not in_margin_canvas(u, v):
            continue
        cu, cv = to_canvas(u, v)
        draw.line([(cu - 4, cv), (cu + 4, cv)], fill=YELLOW, width=1)
        draw.text((cu + 5, cv - 10), f"{d:g}m", fill=YELLOW, font=font)

    # CNN input crop rectangle, from config -- currently a no-op (full frame)
    # since M2 hasn't decided a real crop yet; drawn anyway so this stays
    # accurate the day the config changes.
    c = crop_cfg["crop"]
    cx0, cy0 = to_canvas(c["left"], c["top"])
    cx1, cy1 = to_canvas(c["left"] + c["width"] - 1, c["top"] + c["height"] - 1)
    draw.rectangle([cx0, cy0, cx1, cy1], outline=(255, 140, 0), width=1)

    verdict = "ACCEPT" if n_visible >= MIN_VISIBLE_POINTS else "REJECT"
    text = f"visible={n_visible}/{n_candidates} min={MIN_VISIBLE_POINTS} {verdict}"
    draw.rectangle([0, img.height - 12, img.width, img.height], fill=(0, 0, 0))
    draw.text((2, img.height - 11), text, fill=WHITE, font=font)

    return img


def load_dataset_poses(csv_path, n, seed=0):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
    poses = []
    for idx in chosen:
        r = rows[idx]
        poses.append((r["filename"], float(r["x"]), float(r["y"]), float(r["heading"])))
    return poses


def build_contact_sheet(poses, out_path, cols=4):
    crop_cfg = load_crop_config()
    model = mujoco.MjModel.from_xml_path(VEHICLE_XML)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=IMG_HEIGHT, width=IMG_WIDTH)
    qadr = model.jnt_qposadr[model.joint("root").id]

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    tiles = []
    for label, x, y, heading in poses:
        pixels = render_pose(model, data, renderer, qadr, x, y, heading)
        img = annotate(pixels, x, y, heading, crop_cfg)
        caption_h = 14
        tile = Image.new("RGB", (img.width, img.height + caption_h), (20, 20, 20))
        tile.paste(img, (0, 0))
        d = ImageDraw.Draw(tile)
        d.text((2, img.height + 1), label, fill=WHITE, font=font)
        tiles.append(tile)

    rows = math.ceil(len(tiles) / cols)
    tw, th = tiles[0].size
    pad = 4
    sheet = Image.new("RGB", (cols * tw + (cols + 1) * pad, rows * th + (rows + 1) * pad),
                       (40, 40, 40))
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet.paste(tile, (pad + c * (tw + pad), pad + r * (th + pad)))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sheet.save(out_path)
    print(f"Contact sheet: {len(tiles)} poses -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", nargs=3, type=float, metavar=("X", "Y", "HEADING"))
    ap.add_argument("--from-dataset", type=str, default=None)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.pose:
        x, y, heading = args.pose
        poses = [(f"pose_{x:.2f}_{y:.2f}_{heading:.2f}", x, y, heading)]
        out = args.out or "/tmp/camera_coverage_single_pose.png"
        build_contact_sheet(poses, out, cols=1)
        return

    if args.from_dataset:
        poses = load_dataset_poses(args.from_dataset, args.n)
        out = args.out or OUT_PNG
        build_contact_sheet(poses, out)
        return

    poses = build_contact_sheet_poses()
    build_contact_sheet(poses, args.out or OUT_PNG)


if __name__ == "__main__":
    main()
