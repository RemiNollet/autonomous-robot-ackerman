"""
Windowed curvature relabeling -- an alternative to compute_lane_state's
point-wise Frenet projection, specifically for curvature (ADR-14: the
point-wise kappa label is measurably wrong within L_usable of a curvature
transition, ~42% of the loop's arc-length, because it describes the
vehicle's exact position, not what the camera can see ahead of it).

Both methods below sample the same window: the camera's actual visible
ground distance under the current crop, [0.32, 2.42] m ahead -- not
[0, L_usable]. The vehicle cannot see the ground directly under or near
itself (ADR-12's near-clip); the window was computed in the M3 kickoff
investigation (item 4) by restricting camera_resolvability.py's row sweep
to the actual crop (perception/dataset/cnn_input_config.json, rows
80:182), not the theoretical L_usable bound. Built on the same primitives
compute_lane_state uses (Track.point_at/heading_at/curvature_at/project,
this module's only dependency beyond numpy) -- no MuJoCo dependency, no
new external library.

**Two mechanisms were tried; only one survived validation.**

`windowed_lane_state` fits y(x) = c0 + c1 x + c2 x^2 to the true
centerline in Cartesian vehicle-frame coordinates, kappa = 2*c2. Kept in
this module because it's a real, tested, documented negative result, not
because it's the recommended path: validated against compute_lane_state
away from any primitive join (where the two should agree closely, since
both describe the same constant-curvature ground truth there) and found
NOT to agree. Measured: R=3m arc, true kappa=0.333, fitted=0.471 (41%
relative error); R=5m arc, true kappa=0.200, fitted=0.226 (13%); a
dead-straight segment fits exactly (0% -- confirms the bias is specific
to true curvature, not a generic property of an off-center window). Cause:
degree-4 (quartic) content in a circular arc's y(x) expansion leaks into
the fitted quadratic coefficient over a window this size, worse for
tighter curves. Checked whether a weighting scheme rescues it: only by
concentrating weight so heavily near the window's near edge that the fit
degenerates into a point-curvature estimate there, at which point
`track.curvature_at` at a forward-shifted s is simpler and more accurate
than fitting anything.

`windowed_curvature_average` sidesteps the Cartesian approximation
entirely: kappa_representative = a weighted average of track.curvature_at(s)
itself, sampled along the arc-length actually visible in the window --
never approximating a circle's *position* with a polynomial, so it cannot
inherit the quartic-leakage failure mode above. Validated the same way:
exact agreement away from any join (0.00000000 diff, all three primitive
types, uniform weighting -- not approximate, exact, because curvature is
constant along a pure arc and averaging a constant recovers it exactly
regardless of weighting). This is the method used for the ADR-18
relabeling; `windowed_lane_state` stays for the record of what didn't work
and why.
"""
import math

import numpy as np

from perception.dataset.geometry import Track

# The camera's actual visible ground distance under the current crop
# (perception/dataset/cnn_input_config.json, rows 80:182) -- computed in
# the M3 kickoff investigation (item 4). NOT [0, L_usable]: L_usable is
# the FAR bound alone (2.36 m, matching WINDOW_FAR_M to within the crop's
# own row-to-distance rounding); the NEAR bound (0.32 m) exists because
# the near-clip artifact (ADR-12) means the vehicle cannot see the ground
# directly under or near itself at all.
WINDOW_NEAR_M = 0.32
WINDOW_FAR_M = 2.42
DEFAULT_N_SAMPLES = 60


def windowed_lane_state(track: Track, vehicle_x: float, vehicle_y: float,
                         vehicle_heading: float,
                         d_near: float = WINDOW_NEAR_M, d_far: float = WINDOW_FAR_M,
                         n_samples: int = DEFAULT_N_SAMPLES):
    """Fit y(x) = c0 + c1 x + c2 x^2 to the true centerline in the vehicle
    frame, sampled at n_samples points evenly spaced in arc length over
    [d_near, d_far] m ahead of the vehicle's projection onto the track.

    Returns (lateral_error, heading_error, curvature):

    - curvature = 2*c2 is the label this module exists for.
    - lateral_error = c0 and heading_error = atan(c1) are byproducts of
      the same fit, returned for completeness. They are NOT a proposed
      replacement for compute_lane_state's point-projection values (which
      already pass the sign-convention tests in
      tests/test_lane_state_geometry.py): x=0 (the vehicle's own
      position) sits outside the fitted window [d_near, d_far], so c0/c1
      here are a mild extrapolation of the fit, not an interpolated
      value the way curvature (evaluated across the fitted range itself)
      is.
    """
    s_vehicle = track.project(vehicle_x, vehicle_y)
    ch, sh = math.cos(vehicle_heading), math.sin(vehicle_heading)

    ds = np.linspace(d_near, d_far, n_samples)
    xs = np.empty(n_samples)
    ys = np.empty(n_samples)
    for i, d in enumerate(ds):
        s_global = (s_vehicle + d) % track.total_length
        px, py = track.point_at(s_global)
        dx, dy = px - vehicle_x, py - vehicle_y
        xs[i] = ch * dx + sh * dy
        ys[i] = -sh * dx + ch * dy

    c2, c1, c0 = np.polyfit(xs, ys, 2)

    lateral_error = float(c0)
    heading_error = float(math.atan(c1))
    curvature = float(2.0 * c2)
    return lateral_error, heading_error, curvature


def windowed_curvature_average(track: Track, vehicle_x: float, vehicle_y: float,
                                d_near: float = WINDOW_NEAR_M, d_far: float = WINDOW_FAR_M,
                                n_samples: int = DEFAULT_N_SAMPLES, weight_fn=None):
    """kappa_representative = a weighted average of track.curvature_at(s)
    over the arc-length actually visible in [d_near, d_far] m ahead of the
    vehicle's projection onto the track.

    No polynomial fit, no vehicle_heading dependence (curvature is a
    property of the track alone, not the vehicle's frame) -- this is the
    method validated for use, see the module docstring for why the
    Cartesian alternative (windowed_lane_state) was not: this one cannot
    inherit that bias because it never approximates the centerline's
    *position* with a low-degree polynomial in the first place.

    weight_fn(ds) -> weights, evaluated over the same arc-length samples
    the average is taken over. None (default) is uniform weighting in s --
    validated as sufficient (exact agreement away from any join, all three
    primitive types); a row-density weighting matching
    tools/camera_resolvability.py's rows-per-metre was the fallback plan
    if uniform hadn't been clean, and was not needed.
    """
    s_vehicle = track.project(vehicle_x, vehicle_y)
    ds = np.linspace(d_near, d_far, n_samples)
    kappas = np.array([
        track.curvature_at((s_vehicle + d) % track.total_length) for d in ds
    ])
    weights = np.ones_like(ds) if weight_fn is None else weight_fn(ds)
    return float(np.average(kappas, weights=weights))
