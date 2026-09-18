# Control

A body-frame path-tracking MPC (acados, nonlinear-least-squares cost,
kinematic bicycle model) that turns `/lane_state`'s three scalars into a
steering-rate command, running as `mpc_node` in the ROS2 graph. This page
is the definitive write-up for M3: the formulation, why it's shaped the
way it is, what was measured, and what's still open. The full decision
trail is in [`docs/decisions.md`](../docs/decisions.md) (ADR-15 through
ADR-27); this consolidates it into one place and cites ADR numbers for
anything already decided there rather than re-arguing it.

---

## 1. Problem formulation

`/lane_state` ([`docs/lane-state-contract.md`](../docs/lane-state-contract.md)
§1) publishes three scalars — `lateral_error`, `heading_error`,
`curvature` — that reconstruct exactly into a quadratic reference path in
the vehicle's own body frame:

```
y(x) = c0 + c1 x + c2 x^2
c0 = lateral_error
c1 = tan(heading_error)
c2 = curvature / 2
```

That equivalence is the whole formulation question: does the controller
treat `(c0, c1, c2)` as three error terms to regulate directly
(**Frenet-frame regulation**), or as coefficients of a reference path to
track across the prediction horizon (**body-frame path tracking**)? The
two stop being the same problem the moment `c2` is a real, nonzero
number — Frenet regulation only ever sees curvature as a single
feedforward correction at the current instant; path tracking evaluates
the reference's actual shape at every predicted point ahead.

**The formulation flipped once, and the reason is entirely `curvature`'s
own status.** ADR-15 chose Frenet regulation, for a reason that later
stopped applying: `curvature` was hardcoded to `0.0` at the time (ADR-14
— an untrained kappa head, whose raw output would have been worse to
publish than nothing), so `c2 = 0` always and the two formulations were
numerically identical. With no performance difference to measure, ADR-15
decided on evolvability instead: Frenet regulation consumes the
contract's three scalars directly, with no path reconstruction needed
while `c2` stays zero, and needs no rework the day it doesn't. ADR-18
removed that premise — windowed-curvature-average relabeling and
retraining made kappa a real, accurate signal (81.3% improvement over
the zero baseline near curvature transitions) — and ADR-19 reversed
ADR-15's choice on exactly the grounds ADR-15 itself set up: an MPC with
a genuine multi-step preview should see the reference path's *shape*
across the whole horizon, which is what body-frame tracking gives and
Frenet regulation, by construction, cannot — a feedforward scalar isn't
a preview. **Decision: body-frame path tracking (ADR-19).**

**State, input, parameters** (`control/mpc/model.py`):

```
x = [X, Y, psi, delta]   -- body frame anchored at the vehicle's own pose
                             at the start of the horizon (X=Y=psi=0 at t=0)
u = [ddelta]              -- steering RATE, not angle
p = [c0, c1, c2, v]       -- one /lane_state measurement + speed,
                             identical at every shooting node
```

Anchoring the state at the vehicle's own pose, rather than world
coordinates, is what lets the predicted trajectory and the reference
`y(x)` share one frame with no transform between them — the entire point
of choosing this frame. Controlling `ddelta` rather than `delta` directly
gives the OCP authority over how fast steering moves and makes `delta` a
state the cost and constraints act on smoothly, rather than a
discontinuous control input.

---

## 2. Model

Kinematic bicycle, no tire slip:

```
Xdot     = v cos(psi)
Ydot     = v sin(psi)
psidot   = (v / L) tan(delta)
deltadot = u
```

**Kinematic, not dynamic, is a checked assumption, not a default
(ADR-16).** A kinematic model is only valid while lateral acceleration
stays well below where tire slip angles become significant. Measured at
the operating envelope actually being tested: `v = 1.0 m/s` at the
track's tightest turn (`R_min = 3 m`) gives `a_lat = v²/R_min = 0.333
m/s² = 0.034g`. The simulator's own tire friction (`mu = 1.5`,
`sim/models/car.xml`) puts the grip ceiling around `1.5g` — but that's
the wrong threshold to check against: the kinematic assumption protects
a geometric no-slip property, not a grip limit, so the relevant bound is
the standard rule of thumb for small slip angles, roughly `0.3-0.4g`,
independent of surface `mu`. At `R_min = 3 m`, `0.3g` isn't reached until
`v ≈ 2.97 m/s` — about **3x** the currently measured operating speed.
Kinematic is justified for what M3 actually tests; if a future run
pushes speed toward ~3 m/s on this turn (or a tighter one is added),
this needs re-measuring, not re-assuming.

**`L = 0.26 m`.** Wheelbase, read from `sim/models/car.xml`'s axle
spacing (front axle `x=0.13`, rear `x=-0.13`) and cross-checked against
`sim_server.py`'s own hardcoded `WHEELBASE = 0.26` — one number,
verified from two independent places in the codebase rather than
asserted once and reused.

---

## 3. Cost structure

`NONLINEAR_LS` (`control/mpc/ocp.py`). Stage residual, in raw physical
units:

```
y = [e_lat, e_psi, delta - atan(L*kappa), ddelta]     kappa = 2*c2

e_lat  = Y - (c0 + c1 X + c2 X^2)             -- evaluated at the predicted X
e_psi  = psi - atan(c1 + 2 c2 X)
```

The terminal residual is the same first three terms, with no `ddelta`
(there's no control at the terminal node). Weights are normalized
`W = diag(1/envelope²)` per term — the same normalization *philosophy*
`perception/model/targets.py`'s `*_SCALE` constants use for the CNN's
own targets, applied to the weight instead of the residual
(mathematically identical for a quadratic cost: `(r/s)² == r² (1/s²)`).
The terminal weight `W_e` comes from a DARE
(`scipy.linalg.solve_discrete_are`) on the linearized closed-loop error
dynamics, computed once at a nominal speed and kept curvature-agnostic —
re-linearizing about the true curved equilibrium instead of `delta=0`
changes `W_e` by at most 0.62% (max entry) at this track's tightest
curvature, smaller than the other approximations already baked into
this computation (Euler-vs-ERK4 discretization, a single nominal speed).
acados requires a fixed numeric `W_e`; a curvature- or speed-varying
terminal weight would need a materially bigger parametric-Riccati setup,
out of scope here.

**Two real bugs, both found by checking the resulting trajectory, not
just solver status (ADR-21).**

1. **Terminal target was `delta=0` unconditionally.** Correct only while
   curvature was always zero (ADR-15's premise) — on a curved reference
   the true terminal equilibrium is `delta_eq = atan(L kappa)`, from
   `e_psi_dot = (v/L) tan(delta) - v kappa = 0`. A fixed 8-case sanity
   check (`control/mpc/sanity_check.py`) showed predicted `delta` rising
   correctly mid-horizon on a curved case, then rolling back off toward
   0 by the terminal step — chasing the wrong target at the one node it
   mattered most. Fixed by folding the target into the residual itself
   (`delta - atan(L*kappa)`, `kappa` from the same `c2` parameter
   `e_lat`/`e_psi` already use) rather than a literal
   `yref_e = [0,0,atan(L kappa)]` — acados only accepts a *constant*
   numeric `yref`/`yref_e`, never a parameter-dependent one, so folding
   into the symbolic residual is the only way to express a
   curvature-varying target at all.
2. **Stage-cost consistency.** Fixing the terminal node alone wasn't
   enough: a follow-up closed-loop check found the other N-1 stage
   nodes were still targeting `delta=0` mid-horizon, fighting the
   (now-correct) terminal node the whole way there instead of
   converging smoothly. Fixed by applying the same `delta - delta_eq`
   residual at every stage node, not just the terminal one.

Both fixes were verified against all 8 `sanity_check.py` cases and a
standalone C++ executable built from the same generated solver, outside
Python — matching bit-for-bit, confirming neither fix was a Python-side
artifact.

**Final weights** (`control/mpc/params.py`, ADR-22): `q_y = 12.5`,
`q_psi = 8.0` — both `2.0x` their physically-normalized baseline
(`1/LANE_HALF_WIDTH² = 6.25`, `1/POS_HEADING_RANGE² = 4.0`). `r_delta`
(`1/DELTA_MAX² ≈ 2.78`) and `r_delta_dot` (`1/DELTA_DOT_MAX² ≈ 0.037`)
both stay at their normalized baseline, untouched.

**`q_y` and `q_psi` are coupled through the single steering DOF, not
independent knobs — this is the tuning finding worth stating plainly.**
Both errors are driven by the same control input (steering), so
weighting one harder doesn't just improve that term — it pulls control
effort away from the other. A `q_psi`-only sweep cut transition
overshoot cleanly (~24% on peak `e_lat`, ~11% on peak `e_psi`) but
*worsened* lateral tracking's own steady-state bias: `e_lat` bias grew
**+37%** at the R=3m join and **+44%** at R=5m over the same sweep.
Scaling `q_y` and `q_psi` together instead (holding their ratio fixed)
recovers most of that: the same bias growth roughly halves (+17.6% /
+25.8%) for a comparable overshoot reduction — confirming the coupling
is real, not a `q_psi`-specific artifact, and that isotropic scaling is
the right lever, not weighting one term in isolation.

`S=2` was chosen on that basis: cuts mean peak overshoot ~20% vs `S=1`
at both join sizes (R=3m: `0.0437 → 0.0349 m`; R=5m: `0.0246 → 0.0196
m`) with no saturation risk in ordinary driving (3-lap max `|ddelta|`
stays at 15.6% of `DELTA_DOT_MAX`) and no loss of settling speed — only
overshoot amplitude changes across the sweep. `r_delta_dot` was left
unchanged rather than loosened: loosening it does cut overshoot further,
but trades it for genuine sluggishness (steady-state variance grew up
to 10x at the tight end), and — checked in combination, not assumed
additive — `S=2` plus a loosened `r_delta_dot` lands *exactly* at 100%
of `DELTA_DOT_MAX` on the worst-case combined-disturbance scenario, no
margin left. `S=2` with baseline `r_delta_dot` was the point that
improved tracking without spending that margin.

---

## 4. Constraints

| | Bound | Type | Source |
|---|---|---|---|
| `delta` | `±0.6 rad` | Hard | `car.xml` steering joint range, measured |
| `ddelta` | `±5.2 rad/s` | Hard | `tools/steering_rate_step_response.py`, ADR-20 |
| `e_lat` | `±LANE_HALF_WIDTH` (`±0.4 m`) | Soft, slacked (linear `1e2`, quadratic `1e4`) | Lane geometry |

`e_lat` is slacked, not hard: a receding-horizon controller should be
able to command its way back toward the lane if briefly pushed past its
edge within the horizon, not report the problem infeasible and produce
nothing.

**`DELTA_DOT_MAX = 5.2 rad/s` needs its status stated explicitly, not
just its value.** No hardware exists to bench-test against (ADR-1:
sim-only on the current setup), and `car.xml` has no rate limit modeled
anywhere — the actuators declare PD servo gains (`kp=15`, `kv=0.5`), not
a hard cap. `5.2 rad/s` comes from the simulated position actuator's own
step response (2% settling-time criterion, full `±0.6 rad` sweep) — but
that same actuator has **no `forcerange`** (unlimited torque/current),
which inflates the settling time this figure is built from, not only
the transient peak (`18.95 rad/s`, an obvious artifact of the same
idealization, not used for anything). A torque-limited real servo would
settle *more slowly* than this, not just peak lower. **`5.2 rad/s` is
therefore an upper bound from an idealized actuator, not a conservative
estimate, and not a validated hardware bound** — it stays in the OCP as
measured (a `3.1 rad/s` margined alternative was considered and rejected,
ADR-22, since that concern is about the measurement's own idealization,
not about how hard any particular scenario drives it; the worst standard
sanity-check case reaches 83.9% of the raw bound at the chosen weights,
not actually saturating). Explicitly flagged for re-characterization
against a torque-limited model or real hardware once it exists (M4 or
later).

---

## 5. Horizon

`N = 30`, `Ts = 40 ms` → `T_preview = N·Ts = 1.2 s` (ADR-19).

Two independent bounds set this, not one:

**Upper bound on reach — horizon validity.** The reference `y(x)` is a
fit to what the camera actually saw; it has no validity, and no error
bound, past the edge of that window. `L_usable = 2.356 m` (ADR-11,
re-derived against the actual crop) is the ground distance the camera's
fit is trustworthy over, so the horizon's own reach at speed `v` —
`N · Ts · v` — must stay under it:

```
N * Ts * v <= L_usable
v_max = L_usable / (N * Ts) = 2.356 / 1.2 ≈ 1.96 m/s
```

At the current reference speed (`v = 1.0 m/s`), the horizon reaches
`1.2 m` ahead — about 51% of `L_usable`, margin rather than a tight fit.

**Lower bound on preview length — dynamic response.** `Ts` has to
resolve the vehicle's own actuator dynamics, and `N` has to span enough
of them to matter: the steering actuator's own measured settling time
is `~0.23 s` (ADR-20) against a `40 ms` sample time — about 5-6 samples
across one settling transient, and a `1.2 s` horizon is over 5x that
settling time, not a single step of it. Too coarse a `Ts` or too short
an `N` would have the OCP planning against dynamics it can't actually
resolve or react across.

**Why `N` wasn't increased despite the headroom in the upper bound — a
real, counterintuitive result worth keeping visible.** More preview
sounds strictly better, and there's clearly room (`1.2 m` used of `2.356
m` available). It isn't, because of the limitation in §6: the reference
is a quadratic truncation of the true circular arc, and that mismatch
grows with how far ahead it's evaluated. Checked directly: at `N=40`
(`x_max = 1.6 m`) vs `N=30` (`x_max = 1.2 m`), the closed-loop
steady-state `e_lat` bias at `R=3m` *grew*, `-0.0208 → -0.0244 m`, not
shrank. A longer horizon here makes the known formulation limit worse,
not better — increasing `N` was tried, measured, and rejected on that
basis, not left unconsidered.

---

## 6. Known limitations

**Residual steady-state bias on sustained curvature — bounded and
documented, not an open bug.** `ocp.py`'s reference `y(x) = c0 + c1x +
c2x²` is a quadratic *truncation* of the true circular arc (`y = R -
sqrt(R² - x²)`) — it drops the quartic term, which the terminal-cost fix
(ADR-21) cannot correct, since that fix addresses what `delta`'s
*target* is, not the shape of the path being tracked at all. Verified
directly, not inferred: solving from exactly the mathematically
"correct" equilibrium (`e_lat=0, e_psi=0, delta=delta_eq`) still returns
a nonzero optimal `ddelta` (`+0.0189 rad/s` at `R=3m`) — the solver's
own residual reads as growing across the horizon even from a point
exactly on the true track, which is what a truncated reference, not a
bug, looks like. Confirmed to scale with horizon reach rather than
shrink with more preview (§5's `N=30` vs `N=40` result) — the signature
of a reference-shape mismatch, not a tuning artifact.

At the final `S=2` weights, this shows up as a bounded bias — `~2.1 cm`
lateral offset at `R=3m` (`e_lat = -0.0208 m`, `N=30`) — on the order of
"a few cm / hundredths of a rad" across this track's curvatures.
Open-loop, evaluated right after the terminal/stage-cost fix (ADR-21,
before weight tuning): the equilibrium-tracking gap was `0.03%`/`0.26%`
off the true equilibrium on the two clean curvature cases, `14.7%` on a
case with a large initial disturbance competing for the same horizon's
correction budget. Removing this needs a higher-order or exact-geometry
reference — a materially bigger formulation change than a target or
weight adjustment, out of scope for M3.

---

## 7. Solve time

Three measurement contexts, each a step closer to production, all
comfortably inside the `20 ms` / `50 Hz` budget:

| Context | Mean | p95 | Notes |
|---|---|---|---|
| Standalone acados solve | `~0.4-0.5 ms` | — | Solver alone, `sanity_check.py`/`closed_loop_sim.py`, no ROS2 |
| Node-cycle harness | `0.367 ms` | `0.722 ms` | Real `on_lane_state()`, real `Twist` publish through rclpy/DDS serialization, no live subscriber (ADR-24, 1200 cycles; `p99=0.876 ms`, `max=1.602 ms`) |
| Live full-chain / sustained run | `~0.9-1.1 ms` | `~1.5 ms` | Real subscriber, real perception messages at ~25 Hz, real ROS2 executor contention |

Each step adds real overhead the previous one didn't have to pay — DDS
serialization, then a live executor sharing CPU with the rest of the
running graph — and the number moves accordingly, but stays under 8% of
budget even at its worst observed point. The 18-minute, 24-lap sustained
run's solve-time trend stayed flat throughout, no drift and zero solve
failures — evidence this holds up over time under real load, not just
in a short diagnostic run.

---

## 8. Closed-loop validation

Four runs, each testing a different slice of the system, cited rather
than re-derived here:

- **Perfect-state baseline** (`control/mpc/closed_loop_sim.py`,
  simulator-truth state, no perception in the loop) — 3-lap confirmation
  at the final `S=2` weights: zero solve failures, numbers matching the
  tuning sweep's own `S=2` point exactly (ADR-22).
- **Full-chain with real perception** (`control/mpc/chain_monitor.py`,
  the whole graph, ground truth from odometry compared against
  `/lane_state`'s own reported values) — near/far kappa accuracy ratio
  ~1.7-1.8x, reproducing ADR-18's own offline measurement (`1.80`) live,
  under real perception noise rather than only against held-out test
  data.
- **Sustained stability** — an 18-minute, 24-lap run: flat error RMS and
  solve times throughout (no degradation over time), zero solve
  failures, near/far perception accuracy stable at ~1.7-1.8x for the
  full duration. The perception-health fallback triggered 32 times
  total, staleness-only, clustered into two isolated VM-side jitter
  bursts uncorrelated with `perception_node`'s own report cycle (ruling
  out a recurrence of an earlier, already-fixed report-blocking issue)
  — caught and recovered correctly every time, with no special-cased
  recovery logic (ADR-27).
- **Forced perception-dropout test** (`control/mpc/dropout_test.py`, 27
  scripted trials — staleness at 100/140/160ms/1.5s/7.5s and invalidity
  at N=1/2/3/10, three reps each) — 26/27 matched expectations exactly.
  The one mismatch (a 140ms staleness trial triggering the fallback) is
  explained by the injection method's own ~20-30ms baseline latency —
  the frozen stamp already carried perception's real pipeline lag, so
  the actually-tested age was 166.5ms, past the 150ms threshold — not a
  threshold bug. All 27 trials recovered cleanly with no residual
  state, zero solve failures.

---

## Open for M4

Two items remain genuinely undecided, not just untuned: the acados RTI
preparation/feedback split needs its own bench measurement before it's a
decision, not a default (ADR-19); and `DELTA_DOT_MAX = 5.2 rad/s` needs
re-characterizing against a torque-limited model or real hardware before
M4 relies on it as a safety bound rather than a sim ceiling (ADR-20,
ADR-22). `STALE_AGE_THRESHOLD_S` and `N_INVALID_CONSECUTIVE_THRESHOLD`
are first-guess constants (ADR-27) — functionally correct per §8's
dropout results, but not yet tuned against a dedicated sweep the way the
cost weights were.
