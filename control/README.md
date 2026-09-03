# Control

**Status: M3 kickoff.** Most of this page is pending (vehicle model, OCP
formulation, acados implementation, ROS2 control node, closed-loop
results — see `docs/portfolio-page-plan.md`). What's filled in below is
what the M3 kickoff investigation actually established: the hard
constraint the preview horizon inherits from perception, and the open
questions that came out of it. Everything else lands as M3 proceeds.

---

## Preview horizon: `L_usable` bounds it, not the other way round

Perception's usable preview distance is `L_usable = 2.36 m`
(`docs/decisions.md` ADR-11, re-derived against the actual crop in the M3
kickoff investigation rather than just the theoretical `R_min` bound —
see below). Whatever horizon the MPC picks has to fit inside that
distance at whatever speed the vehicle is actually moving; it isn't a
free parameter.

**`v_max = L_usable / T_preview_min`.** If the controller needs at least
`T_preview_min` seconds of lookahead to plan usefully (`N` prediction
steps at sample time `Ts`, so `T_preview_min = N * Ts`), the vehicle
cannot exceed:

```
v_max = L_usable / (N * Ts)
```

Equivalently, the **horizon validity constraint**: for any chosen `N`,
`Ts`, and operating speed `v`,

```
N * Ts * v <= L_usable
```

**Why this must hold, not just should:** the MPC's preview path is the
quadratic `y(x) = c0 + c1 x + c2 x^2` reconstructed from `/lane_state`
(`docs/lane-state-contract.md` §1). That quadratic is a fit to what the
camera actually saw — it has no validity, and no error bound, outside the
window it was fit over. Evaluating it at `x` beyond `L_usable` isn't a
degraded prediction, it's evaluating a polynomial at a point with no
information behind it at all. `N * Ts * v` is exactly the ground distance
the prediction horizon reaches at speed `v`; keeping it under `L_usable`
is what keeps every point the MPC evaluates inside the region the
quadratic was actually fit to.

**`L_usable = 2.36 m` still holds under the actual crop, not just the
theoretical bound** — checked directly in the M3 kickoff investigation,
not assumed to carry over from ADR-11 unchanged. Of the three limits ADR-11
defined (`L_resolvable`, `L_separable`, `L_representable`), only
`L_representable` (`R_min * atan(1) = 2.356 m`, from track geometry alone)
is even reachable within what the current crop (rows 80-182,
`perception/dataset/cnn_input_config.json`) actually shows: restricting
`L_resolvable`/`L_separable`'s distance sweep to that crop's real row range
(ground distance 0.32-2.42 m) rather than the full 320x240 frame, neither
threshold is crossed at all inside the visible window. `L_representable`
isn't the smallest of three competing numbers here — it's the only one of
the three that binds.

Once `N` and `Ts` are chosen (M3, not yet decided), plug them in above to
get the actual speed ceiling this perception stack imposes.

---

## Open questions

**Quadratic (kappa) relabeling of v0 — flagged, not decided.** The M3
kickoff investigation found that `compute_lane_state`
(`perception/dataset/geometry.py`) depends only on track geometry
(`REFERENCE_TRACK`, zero MuJoCo dependency) and the vehicle pose
`(x, y, heading)` already stored per row in `data/dataset_v0/labels.csv` —
a windowed quadratic fit could relabel all 4000 samples without
re-rendering a single image, potentially correcting the kappa labels that
are wrong within `L_usable` of a curvature transition (ADR-14, ~42% of
the loop's arc-length). This is a retraining-scale change to the M2
model, not something implemented or decided here. Not implemented, not
retrained, no ADR written deciding it — that's a scheduling call against
the rest of M3/M4, not an architecture decision this page or the ADR log
should preempt.

Separately and independently: a **cubic** term was investigated and
rejected (ADR-17) — the camera's visible window is permanently offset
from the vehicle's own origin (it can't see the ground under itself), so
a fit cubic coefficient carries a geometric noise floor comparable to any
real transition signal, regardless of track or sampling strategy. This
does not affect the quadratic relabeling question above; the two are
independent and shouldn't be conflated (ADR-17's own note).
