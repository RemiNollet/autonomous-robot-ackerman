"""
Camera visibility check: does the onboard camera actually see lane markings
from a given vehicle pose?

Why this exists: the first dataset generation pass sampled positive
(valid=True) poses with lateral offsets up to 0.7 m against a lane
half-width of 0.4 m. Combined with heading errors up to 0.5 rad on the
tight (R=3 m) arcs, this produced "valid" samples where the lane had
left the camera frame entirely. The labels were arithmetically correct
— the sampling envelope was wrong. Rather than guess a narrower envelope,
this module answers the question directly and the generator uses it as a
rejection filter.

Camera parameters are hard-coded from sim/models/car.xml and verified
against MuJoCo's computed extrinsics in tests/test_camera_visibility.py.
If cam_front's pose or fovy changes in the MJCF, that test fails and
these constants must be updated to match.

Crop-visibility consistency (docs/decisions.md ADR-11 finding 7, closed
here): `lane_is_visible`/`any_lane_visible` used to count a point "visible"
if it fell anywhere in the full 320x240 render, but the CNN only ever sees
`perception/dataset/cnn_input_config.json`'s crop (rows 80:182 -- verified
to land almost exactly on L_usable at the top (row 80.23 at d=2.356 m) and
the near-clip artifact boundary at the bottom (ADR-12), i.e. the crop
already IS "the region that's both real ground and within L_usable" by
construction, not a coincidence). A point outside the crop but inside the
full frame was being counted toward MIN_VISIBLE_POINTS even though the
network never receives it. `point_visible_in_crop` closes that gap; the
generator's two visibility counts now use it instead of `point_visible`.
"""

import json
import math
import os

from perception.dataset.geometry import Track

_CROP_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "cnn_input_config.json")
with open(_CROP_CONFIG_PATH) as _f:
    _crop_cfg = json.load(_f)
CROP_TOP = _crop_cfg["crop"]["top"]
CROP_LEFT = _crop_cfg["crop"]["left"]
CROP_BOTTOM = CROP_TOP + _crop_cfg["crop"]["height"]
CROP_RIGHT = CROP_LEFT + _crop_cfg["crop"]["width"]

# --- Camera parameters, from sim/models/car.xml (cam_front) ---
CAM_OFFSET_FORWARD = 0.16    # m, along vehicle x, from chassis origin
CAM_HEIGHT = 0.125           # m, world z (chassis z=0.075 + camera z=0.05)
CAM_PITCH_DOWN = 0.30587887140485215  # rad, ~17.5 deg; exact value from
                                       # normalizing the MJCF xyaxes y-vector
                                       # (0.3, 0, 0.95). Verified against
                                       # MuJoCo's own extrinsics to <1e-9 m.
CAM_FOVY_DEG = 75.0
IMG_WIDTH, IMG_HEIGHT = 320, 240

_TAN_HALF_FOVY = math.tan(math.radians(CAM_FOVY_DEG) / 2.0)
_ASPECT = IMG_WIDTH / IMG_HEIGHT
_TAN_HALF_FOVX = _TAN_HALF_FOVY * _ASPECT

# How far ahead along the track to look for visible markings. Beyond the
# preview horizon the MPC cares about, distant markings do not make a
# sample useful even if technically in frame.
LOOKAHEAD_M = 6.0
LOOKAHEAD_SAMPLES = 60

# A sample counts as "lane visible" only if at least this many boundary
# points project inside the image. One stray pixel at the frame edge is
# not a usable perception target.
MIN_VISIBLE_POINTS = 6


def point_in_camera_frame(px, py, pz, vehicle_x, vehicle_y, vehicle_heading):
    """World point -> camera frame (OpenGL convention: x right, y up,
    -z forward). Returns (cx, cy, cz)."""
    ch, sh = math.cos(vehicle_heading), math.sin(vehicle_heading)

    cam_x = vehicle_x + CAM_OFFSET_FORWARD * ch
    cam_y = vehicle_y + CAM_OFFSET_FORWARD * sh
    cam_z = CAM_HEIGHT

    dx, dy, dz = px - cam_x, py - cam_y, pz - cam_z

    # Rotate world -> vehicle frame (undo heading)
    vx = dx * ch + dy * sh
    vy = -dx * sh + dy * ch
    vz = dz

    # Vehicle frame -> camera frame.
    # Camera axes in vehicle frame (from the MJCF xyaxes):
    #   x_cam = (0, -1, 0)
    #   y_cam = (sin(pitch), 0, cos(pitch))
    #   z_cam = (-cos(pitch), 0, sin(pitch))
    sp, cp = math.sin(CAM_PITCH_DOWN), math.cos(CAM_PITCH_DOWN)
    cx = -vy
    cy = sp * vx + cp * vz
    cz = -cp * vx + sp * vz
    return cx, cy, cz


def project_to_pixel(px, py, pz, vehicle_x, vehicle_y, vehicle_heading):
    """World point -> (u, v, depth, in_frame).

    Standard OpenGL/MuJoCo symmetric-frustum projection: camera-frame (x
    right, y up, -z forward) -> NDC in [-1, 1] via the fovy/aspect-derived
    half-angles -> viewport pixels.

    `u` in [0, IMG_WIDTH], increasing right, matching NDC x. `v` in
    [0, IMG_HEIGHT] is measured from the TOP row and increases downward --
    image-array convention (row 0 = top), NOT OpenGL's bottom-up NDC y, so
    that `v` lines up directly with a rendered frame's row index as read by
    PIL/numpy. This flip (v = (1 - ndc_y)/2 * H rather than (ndc_y + 1)/2 * H)
    was verified empirically against real MuJoCo renders in
    tests/test_camera_visibility.py::test_project_to_pixel_matches_rendered_markers
    -- get it backwards and every vertical position silently mirrors.

    `depth` is the distance in front of the camera along its viewing axis
    (positive = in front). `in_frame` is False for points behind the camera
    (depth <= 0) or outside the viewport, independent of `u`/`v`'s numeric
    value in that case (they are still returned, for debugging/plotting, but
    are meaningless off-frustum since depth could be near zero).
    """
    cx, cy, cz = point_in_camera_frame(px, py, pz, vehicle_x, vehicle_y, vehicle_heading)
    depth = -cz
    if depth <= 1e-6:
        return None, None, depth, False  # behind the camera, or at the camera

    ndc_x = cx / (depth * _TAN_HALF_FOVX)
    ndc_y = cy / (depth * _TAN_HALF_FOVY)
    u = (ndc_x + 1.0) * 0.5 * IMG_WIDTH
    v = (1.0 - ndc_y) * 0.5 * IMG_HEIGHT

    # Inclusive on both ends to exactly match the old angular-bounds check
    # (abs(cx/depth) <= tan_half_fovx etc.) that this replaces -- a boundary
    # sample landing exactly on an edge is accepted either way.
    in_frame = (0.0 <= u <= IMG_WIDTH) and (0.0 <= v <= IMG_HEIGHT)
    return u, v, depth, in_frame


def point_visible(px, py, pz, vehicle_x, vehicle_y, vehicle_heading) -> bool:
    _, _, _, in_frame = project_to_pixel(px, py, pz, vehicle_x, vehicle_y, vehicle_heading)
    return in_frame


def point_visible_in_crop(px, py, pz, vehicle_x, vehicle_y, vehicle_heading) -> bool:
    """Like point_visible, but also requires the pixel to fall within the
    CNN's actual input crop (CROP_TOP/BOTTOM/LEFT/RIGHT), not merely
    somewhere in the full render. This is what the dataset generator's
    visibility counts use -- a point outside the crop is invisible to the
    network regardless of whether MuJoCo would have rendered it."""
    u, v, _, in_frame = project_to_pixel(px, py, pz, vehicle_x, vehicle_y, vehicle_heading)
    if not in_frame:
        return False
    return (CROP_LEFT <= u <= CROP_RIGHT) and (CROP_TOP <= v <= CROP_BOTTOM)


def count_visible_lane_points(track: Track, vehicle_x, vehicle_y, vehicle_heading,
                               lane_half_width: float) -> int:
    """Count lane-boundary sample points ahead of the vehicle that fall
    inside the camera image."""
    s0 = track.project(vehicle_x, vehicle_y)
    count = 0
    for i in range(LOOKAHEAD_SAMPLES):
        s = s0 + LOOKAHEAD_M * i / LOOKAHEAD_SAMPLES
        s = s % track.total_length  # track is a closed loop
        cx, cy = track.point_at(s)
        h = track.heading_at(s)
        nx, ny = -math.sin(h), math.cos(h)
        for side in (+1.0, -1.0):
            bx = cx + side * lane_half_width * nx
            by = cy + side * lane_half_width * ny
            if point_visible_in_crop(bx, by, 0.02, vehicle_x, vehicle_y, vehicle_heading):
                count += 1
    return count


def lane_is_visible(track: Track, vehicle_x, vehicle_y, vehicle_heading,
                     lane_half_width: float) -> bool:
    n = count_visible_lane_points(track, vehicle_x, vehicle_y, vehicle_heading,
                                   lane_half_width)
    return n >= MIN_VISIBLE_POINTS


# Whole-track scan, used to qualify NEGATIVE samples. The forward-window
# check above answers "can the vehicle see the lane it should be following";
# this one answers "can the camera see ANY lane marking at all". On a closed
# 45 m loop the two differ a lot: a vehicle 4 m off course frequently has a
# different part of the loop in frame. Such a sample is not a usable negative
# — the CNN would see clear markings while the label says confidence=0.
WHOLE_TRACK_SAMPLES = 400


def count_visible_lane_points_whole_track(track: Track, vehicle_x, vehicle_y,
                                           vehicle_heading, lane_half_width: float) -> int:
    count = 0
    for i in range(WHOLE_TRACK_SAMPLES):
        s = track.total_length * i / WHOLE_TRACK_SAMPLES
        cx, cy = track.point_at(s)
        h = track.heading_at(s)
        nx, ny = -math.sin(h), math.cos(h)
        for side in (+1.0, -1.0):
            bx = cx + side * lane_half_width * nx
            by = cy + side * lane_half_width * ny
            if point_visible_in_crop(bx, by, 0.02, vehicle_x, vehicle_y, vehicle_heading):
                count += 1
    return count


def any_lane_visible(track: Track, vehicle_x, vehicle_y, vehicle_heading,
                      lane_half_width: float) -> bool:
    n = count_visible_lane_points_whole_track(track, vehicle_x, vehicle_y,
                                               vehicle_heading, lane_half_width)
    return n >= MIN_VISIBLE_POINTS