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

**`N = 30`, `Ts = 40 ms`** (`T_preview = 1.2 s`, ADR-19): `v_max =
L_usable / (N Ts) = 2.356 / 1.2 ≈ 1.96 m/s`. At the current reference
speed (`v = 1.0 m/s`), the horizon reaches `1.2 m` ahead — comfortably
under `L_usable`, not a tight fit.

---

## Resolved

**Quadratic (kappa) relabeling of v0 — done, ADR-18.** This was flagged
here as an open question during the M3 kickoff investigation; it has
since been implemented, retrained, measured, and decided (ADR-18,
`docs/decisions.md`): windowed-relabeled curvature is usable (81.3%
improvement over the published-zero baseline at the near-join zone, no
regression on the other three outputs), and the MPC formulation (ADR-19)
now treats curvature as a real, non-zero parameter rather than the
always-zero placeholder this section originally described.

**Cubic term — investigated and rejected, ADR-17.** The camera's visible
window is permanently offset from the vehicle's own origin (it can't see
the ground under itself), so a fit cubic coefficient carries a geometric
noise floor comparable to any real transition signal, regardless of
track or sampling strategy. Independent of the quadratic relabeling item
above; the two were never the same question (ADR-17's own note).

## Open questions

None currently flagged as formulation questions. Two pending
*measurements*, not decisions: `ddelta`'s hard constraint uses an
explicitly unmeasured placeholder (`control/mpc/params.py`,
`DELTA_DOT_MAX_PLACEHOLDER`) pending a bench test, and the acados RTI
preparation/feedback split needs its own bench measurement before it's
decided (ADR-19).
