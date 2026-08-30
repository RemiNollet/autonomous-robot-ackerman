"""
How far ahead can this camera actually resolve the lane? Ground distance
compresses into fewer and fewer image rows as it recedes toward the horizon,
and the lane's projected width in pixels shrinks with it -- past some
distance, a point counted as "visible" by camera_visibility.py carries near
zero usable information: a handful of pixels the CNN cannot regress a lateral
offset from, or a few points too close together in image space to represent
a curve instead of noise.

This script measures three independent limits and reports the tightest:

  L_resolvable   -- projected lane width drops below 10 px at CNN input
                    resolution (after crop/resize, from cnn_input_config.json)
  L_separable    -- two points 0.5 m apart along the ground stop being
                    resolvable as more than 1 image row apart
  L_representable -- from track geometry alone: where the quadratic
                    y(x) = c0 + c1 x + c2 x^2 preview path (the /lane_state
                    contract's representation, see docs/lane-state-contract.md
                    section 1) stops being a good model of a circular arc of
                    radius R_min, taken as tan(s/R_min) > 1.0

  L_usable = min(L_resolvable, L_separable, L_representable)

Also sweeps camera mount height (0.125 m current, 0.25 m) x pitch (17.5 deg
current, 10 deg) to show the resolution/L_usable trade -- entirely local to
this script; camera_visibility.py's actual constants are never touched (see
the parent PR's non-goals: measure and report, decisions come after).

Usage:
    python3 tools/camera_resolvability.py
        -> docs/camera-resolvability.md, docs/camera-resolvability.png
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from perception.dataset.camera_visibility import (
    CAM_OFFSET_FORWARD, CAM_HEIGHT, CAM_PITCH_DOWN, CAM_FOVY_DEG,
    IMG_WIDTH, IMG_HEIGHT,
)
from perception.dataset.track_definitions import RADIUS_1, RADIUS_2, LANE_HALF_WIDTH

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "perception", "dataset",
                            "cnn_input_config.json")
OUT_MD = "docs/camera-resolvability.md"
OUT_PNG = "docs/camera-resolvability.png"

LANE_WIDTH_PX_THRESHOLD = 10.0
SEPARABILITY_GAP_M = 0.5
SEPARABILITY_ROWS = 2.0
D_GRID = np.arange(0.20, 20.0, 0.01)  # ground distance sweep, metres


def project_straight_ahead(dist, lateral, height, pitch_down, fovy_deg=CAM_FOVY_DEG,
                            width=IMG_WIDTH, img_height=IMG_HEIGHT):
    """Parametrized version of camera_visibility.project_to_pixel, for a
    vehicle at the world origin with heading 0 -- i.e. straight-ahead ground
    distance `dist`, lateral offset `lateral`, world z=0 (true ground, not
    the 0.02 m lane-marking height used elsewhere; irrelevant at these
    distances). height/pitch are swept locally; CAM_OFFSET_FORWARD and fovy
    are the fixed optics, not part of the mechanical-mount trade.
    """
    dx, dy, dz = dist - CAM_OFFSET_FORWARD, lateral, -height
    sp, cp = math.sin(pitch_down), math.cos(pitch_down)
    cx = -dy
    cy = sp * dx + cp * dz
    cz = -cp * dx + sp * dz
    depth = -cz
    if depth <= 1e-6:
        return None, None, depth

    tan_half_fovy = math.tan(math.radians(fovy_deg) / 2.0)
    tan_half_fovx = tan_half_fovy * (width / img_height)
    ndc_x = cx / (depth * tan_half_fovx)
    ndc_y = cy / (depth * tan_half_fovy)
    u = (ndc_x + 1.0) * 0.5 * width
    v = (1.0 - ndc_y) * 0.5 * img_height
    return u, v, depth


def compute_curves(height, pitch_down):
    v = np.array([project_straight_ahead(d, 0.0, height, pitch_down)[1] for d in D_GRID])
    u_left = np.array([project_straight_ahead(d, LANE_HALF_WIDTH, height, pitch_down)[0]
                        for d in D_GRID])
    u_right = np.array([project_straight_ahead(d, -LANE_HALF_WIDTH, height, pitch_down)[0]
                         for d in D_GRID])
    lane_width_px = np.abs(u_left - u_right)
    return v, lane_width_px


def rows_per_metre(v):
    # Central difference; D_GRID is uniform.
    dv = np.gradient(v, D_GRID)
    return np.abs(dv)


def first_crossing_below(d_grid, series, threshold):
    """First distance at which `series` drops below `threshold` and stays
    monotonically decreasing-ish (report the first crossing, not a later
    re-crossing from an out-of-frame NaN region)."""
    below = series < threshold
    idx = np.argmax(below) if below.any() else None
    if idx is None or not below[idx]:
        return None
    return float(d_grid[idx])


def resolvability_limits(height, pitch_down, crop_cfg):
    v, lane_width_px = compute_curves(height, pitch_down)
    rpm = rows_per_metre(v)

    scale_x = crop_cfg["resize_to"]["width"] / crop_cfg["crop"]["width"]
    lane_width_cnn_px = lane_width_px * scale_x

    valid = np.isfinite(v) & (v >= 0) & (v <= IMG_HEIGHT)
    l_resolvable = first_crossing_below(D_GRID[valid], lane_width_cnn_px[valid],
                                         LANE_WIDTH_PX_THRESHOLD)

    # rows_per_metre * GAP < ROWS  <=>  rows_per_metre < ROWS / GAP
    rpm_threshold = SEPARABILITY_ROWS / SEPARABILITY_GAP_M
    l_separable = first_crossing_below(D_GRID[valid], rpm[valid], rpm_threshold)

    return {
        "v": v, "lane_width_px": lane_width_px, "lane_width_cnn_px": lane_width_cnn_px,
        "rows_per_metre": rpm, "valid": valid,
        "L_resolvable": l_resolvable, "L_separable": l_separable,
    }


def main():
    with open(CONFIG_PATH) as f:
        crop_cfg = json.load(f)

    r_min = min(RADIUS_1, RADIUS_2)
    l_representable = r_min * math.atan(1.0)

    current = resolvability_limits(CAM_HEIGHT, CAM_PITCH_DOWN, crop_cfg)
    l_usable = min(x for x in
                    (current["L_resolvable"], current["L_separable"], l_representable)
                    if x is not None)

    # --- Sweep: height x pitch ------------------------------------------------
    sweep_points = [
        (0.125, math.radians(17.5), "0.125 m (current), 17.5 deg (current)"),
        (0.125, math.radians(10.0), "0.125 m (current), 10 deg"),
        (0.250, math.radians(17.5), "0.25 m, 17.5 deg (current)"),
        (0.250, math.radians(10.0), "0.25 m, 10 deg"),
    ]
    sweep_rows = []
    for height, pitch, label in sweep_points:
        res = resolvability_limits(height, pitch, crop_cfg)
        l_use = min(x for x in (res["L_resolvable"], res["L_separable"], l_representable)
                     if x is not None)
        sweep_rows.append((label, height, math.degrees(pitch),
                            res["L_resolvable"], res["L_separable"], l_representable, l_use))

    # --- Plots -----------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    v = current["v"]
    valid = current["valid"]
    ax = axes[0, 0]
    ax.plot(D_GRID[valid], v[valid])
    ax.set_xlabel("ground distance [m]")
    ax.set_ylabel("image row (0 = top)")
    ax.set_title("Ground distance vs. image row")
    ax.invert_yaxis()

    ax = axes[0, 1]
    ax.plot(D_GRID[valid], current["rows_per_metre"][valid])
    ax.axhline(SEPARABILITY_ROWS / SEPARABILITY_GAP_M, color="r", linestyle="--",
               label=f"{SEPARABILITY_ROWS:.0f} rows / {SEPARABILITY_GAP_M} m threshold")
    ax.set_xlabel("ground distance [m]")
    ax.set_ylabel("image rows per metre")
    ax.set_title("Depth resolution vs. distance")
    ax.set_yscale("log")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(D_GRID[valid], current["lane_width_px"][valid], label="320x240 (native)")
    ax.plot(D_GRID[valid], current["lane_width_cnn_px"][valid], label="CNN input (crop+resize)")
    ax.axhline(LANE_WIDTH_PX_THRESHOLD, color="r", linestyle="--",
               label=f"{LANE_WIDTH_PX_THRESHOLD:.0f} px threshold")
    ax.set_xlabel("ground distance [m]")
    ax.set_ylabel("projected lane width [px]")
    ax.set_title("Lane width in pixels vs. distance")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    labels = [row[0].replace(", ", "\n") for row in sweep_rows]
    l_res = [row[3] if row[3] is not None else np.nan for row in sweep_rows]
    l_sep = [row[4] if row[4] is not None else np.nan for row in sweep_rows]
    l_use = [row[6] for row in sweep_rows]
    x = np.arange(len(sweep_rows))
    w = 0.25
    ax.bar(x - w, l_res, w, label="L_resolvable")
    ax.bar(x, l_sep, w, label="L_separable")
    ax.bar(x + w, l_use, w, label="L_usable")
    ax.axhline(l_representable, color="k", linestyle=":", label="L_representable (fixed)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("distance [m]")
    ax.set_title("Height x pitch trade")
    ax.legend(fontsize=7)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130)

    # --- Markdown report ---------------------------------------------------
    lines = []
    lines.append("# Camera resolvability report\n")
    lines.append(
        "Generated by `tools/camera_resolvability.py`. Measures how far ahead the "
        "camera can actually resolve the lane, independent of whether "
        "`camera_visibility.py` counts a point as \"visible\" -- see the parent PR's "
        "finding #1 (`LOOKAHEAD_M` vs these limits).\n"
    )
    lines.append("## Current camera (height=0.125 m, pitch=17.5 deg)\n")
    lines.append(f"- **L_resolvable** (lane width < {LANE_WIDTH_PX_THRESHOLD:.0f} px at CNN "
                  f"input resolution): **{current['L_resolvable']:.2f} m**\n"
                  if current["L_resolvable"] is not None else
                  "- **L_resolvable**: not reached within the 20 m sweep\n")
    lines.append(f"- **L_separable** (two points {SEPARABILITY_GAP_M} m apart fall under "
                  f"{SEPARABILITY_ROWS:.0f} image rows apart): **{current['L_separable']:.2f} m**\n"
                  if current["L_separable"] is not None else
                  "- **L_separable**: not reached within the 20 m sweep\n")
    lines.append(f"- **L_representable** (tan(s / R_min) > 1, R_min = {r_min:.1f} m): "
                  f"**{l_representable:.2f} m**\n")
    lines.append(f"- **L_usable = min(...) = {l_usable:.2f} m**\n")

    lines.append("\n## Ground distance vs. image row (sample)\n")
    lines.append("| distance [m] | row |\n|---|---|\n")
    for d in [0.25, 0.5, 1, 2, 3, 5, 8, 12, 16]:
        idx = int(np.argmin(np.abs(D_GRID - d)))
        row_val = v[idx]
        lines.append(f"| {d:g} | {row_val:.1f} |\n" if valid[idx] else f"| {d:g} | (off-frame) |\n")

    lines.append("\n## Image rows per metre vs. distance (sample)\n")
    lines.append("| distance [m] | rows / m |\n|---|---|\n")
    for d in [0.25, 0.5, 1, 2, 3, 5, 8, 12, 16]:
        idx = int(np.argmin(np.abs(D_GRID - d)))
        rpm_val = current["rows_per_metre"][idx]
        lines.append(f"| {d:g} | {rpm_val:.2f} |\n" if valid[idx] else f"| {d:g} | (off-frame) |\n")

    lines.append("\n## Projected lane width vs. distance (sample)\n")
    lines.append("| distance [m] | width @320x240 [px] | width @CNN input [px] |\n"
                  "|---|---|---|\n")
    for d in [0.25, 0.5, 1, 2, 3, 5, 8, 12, 16]:
        idx = int(np.argmin(np.abs(D_GRID - d)))
        if valid[idx]:
            lines.append(f"| {d:g} | {current['lane_width_px'][idx]:.1f} | "
                          f"{current['lane_width_cnn_px'][idx]:.1f} |\n")
        else:
            lines.append(f"| {d:g} | (off-frame) | (off-frame) |\n")

    lines.append("\n## Height x pitch trade\n")
    lines.append("| mount | L_resolvable [m] | L_separable [m] | L_representable [m] | "
                  "L_usable [m] |\n|---|---|---|---|---|\n")
    for label, height, pitch_deg, lr, ls, lrep, lu in sweep_rows:
        lr_s = f"{lr:.2f}" if lr is not None else "n/a"
        ls_s = f"{ls:.2f}" if ls is not None else "n/a"
        lines.append(f"| {label} | {lr_s} | {ls_s} | {lrep:.2f} | {lu:.2f} |\n")

    lines.append(f"\n![resolvability plots]({os.path.basename(OUT_PNG)})\n")

    with open(OUT_MD, "w") as f:
        f.writelines(lines)

    print(f"L_resolvable   = {current['L_resolvable']}")
    print(f"L_separable    = {current['L_separable']}")
    print(f"L_representable= {l_representable:.3f}")
    print(f"L_usable       = {l_usable:.3f}")
    print(f"Written {OUT_MD} and {OUT_PNG}")


if __name__ == "__main__":
    main()
