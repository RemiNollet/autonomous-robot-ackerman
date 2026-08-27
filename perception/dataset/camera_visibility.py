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
"""

import math
from perception.dataset.geometry import Track

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


def point_visible(px, py, pz, vehicle_x, vehicle_y, vehicle_heading) -> bool:
    cx, cy, cz = point_in_camera_frame(px, py, pz, vehicle_x, vehicle_y, vehicle_heading)
    if cz >= -1e-6:
        return False  # behind the camera
    depth = -cz
    return (abs(cy / depth) <= _TAN_HALF_FOVY) and (abs(cx / depth) <= _TAN_HALF_FOVX)


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
            if point_visible(bx, by, 0.02, vehicle_x, vehicle_y, vehicle_heading):
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
            if point_visible(bx, by, 0.02, vehicle_x, vehicle_y, vehicle_heading):
                count += 1
    return count


def any_lane_visible(track: Track, vehicle_x, vehicle_y, vehicle_heading,
                      lane_half_width: float) -> bool:
    n = count_visible_lane_points_whole_track(track, vehicle_x, vehicle_y,
                                               vehicle_heading, lane_half_width)
    return n >= MIN_VISIBLE_POINTS