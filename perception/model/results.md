# M2 results

config_hash=4ba7402eaef9  (perception/model/training_config.yaml, epochs=60 lr=0.001 batch_size=64 seed=42)

No early stopping on val loss: v0's val split does not measure generalisation (ADR-11 finding 5 -- every arc is geometrically identical to its twin), so stopping on it would halt on noise and make runs non-reproducible. Epoch count is fixed, chosen from where perception/model/loss_curves.png actually plateaus.

kappa's loss weight is 1.0 (ADR-18): the point-wise Frenet label's near-transition problem (ADR-14) was addressed by relabeling data/dataset_v0/labels.csv with perception/dataset/windowed_relabel.windowed_curvature_average -- a weighted average of true track curvature over the camera's actual visible window, not a Cartesian polynomial fit (which was tried first and rejected: 13-41% biased even on a pure arc with zero discontinuity present, see windowed_relabel.py's module docstring). Near-join vs away-from-join kappa MAE, and the check that e_y/e_psi/confidence didn't regress from retraining, are in perception/model/analyze_adr18_retrain.py's output and ADR-18 (docs/decisions.md) -- not in the per-sample table below, which has no notion of distance-to-transition.

Plateau epoch per component (val, main model): {'e_y': 60, 'e_psi': 55, 'kappa': 58, 'confidence': 60}

| Run | Params | MACs | val loss | normalized test MAE e_y | normalized test MAE e_psi | conf acc | train time |
|---|---|---|---|---|---|---|---|
| main (width=1.0) | 417,940 | 33,229,312 | 0.0122 | 0.0216 | 0.0579 | 1.000 | 504.5s |

## Test-set error in physical units

See `perception/model/physical_metrics.py`. "Normalized" above is the same unit the loss is computed in (dimensionless, scaled by envelope width); the tables below are the physical-unit numbers a controller would actually see.

### main (width=1.0), test split
**In-distribution error, not a generalisation measurement.** Per ADR-11 finding 5, no partition of v0 measures generalisation -- every arc is geometrically identical to its twins under identical lighting and texture. These are upper bounds on a track the model has effectively memorised, reported because they're still the honest description of what was measured, not because they say whether the model can drive on a track it hasn't seen.
| Output | Units | MAE | RMSE | p50 | p95 | max | n |
|---|---|---|---|---|---|---|---|
| e_y | m | 0.0216 | 0.0290 | 0.0176 | 0.0530 | 0.1688 | 348 |
| e_psi | rad | 0.0579 | 0.0701 | 0.0517 | 0.1316 | 0.1782 | 348 |
| e_psi | deg | 3.32 | 4.02 | 2.96 | 7.54 | 10.21 | 348 |

e_y MAE is 5.4% of the lane half-width (0.4 m); p95 is 13.3%. e_psi MAE is 11.6% of the sampling envelope (+/-0.5 rad = +/-28.6 deg); p95 is 26.3%.

Confidence: accuracy 1.000 over n_valid=348, n_invalid=36 (imbalanced ~90/10 -- accuracy alone hides class performance). Per-class: valid recall 1.000, invalid recall 1.000.

**kappa: loss weight 1.0, trained.** Not evaluated in this per-sample/per-bin table -- its meaningful comparison is near-join vs away-from-join MAE (distance to the next curvature transition, not available to this generic evaluator); see perception/model/analyze_adr18_retrain.py and docs/decisions.md ADR-18.

| Curvature bin | e_y MAE (m) | e_y p95 (m) | e_psi MAE (deg) | e_psi p95 (deg) | n |
|---|---|---|---|---|---|
| straight | 0.0208 | 0.0571 | 2.38 | 5.22 | 124 |
| R=5m arc | 0.0188 | 0.0419 | 1.66 | 4.25 | 54 |
| R=3m arc | 0.0169 | 0.0455 | 2.48 | 5.47 | 48 |

## ADR-18: kappa, near-join vs away-from-join (windowed-relabeled dataset)

Full dataset (all splits, not test alone -- ADR-9's per-primitive
stratification puts every test sample within ~0.6 m of a transition by
construction, so test alone cannot represent "away from any join" at all).
`perception/model/analyze_adr18_retrain.py`. n=3600 valid samples (1508
near-join, `dist_to_next_transition <= L_usable=2.356 m`; 2092
away-from-join).

| Method | Near-join MAE (1/m) | Away-from-join MAE (1/m) | Near/away ratio |
|---|---|---|---|
| Published today (hardcoded 0.0) | 0.1317 | 0.1379 | 0.96 |
| ADR-14 checkpoint (untrained kappa) | 0.7228 | 0.7120 | 1.02 |
| **ADR-18 retrain (kappa weight 1.0)** | **0.0247** | **0.0137** | **1.80** |

**81.3% improvement over the published-today baseline at the near-join
zone**; near/away ratio 1.80, under the ~2x usability bar. Both criteria
from the M3 "quadratic curvature relabeling" brief are met, clearly, not
marginally. The ADR-14 checkpoint's own untrained kappa output (~0.72 MAE,
both zones) is included only to make the point explicit: an untrained head
is not a fair "before" baseline for this comparison -- it is worse than
just publishing zero, which is exactly why ADR-14 hardcoded zero in the
first place.

**Do the other three outputs regress?** No -- test split, in-distribution/
interpolation only (ADR-11 finding 5), same caveat as every other
test-split number in this project:

| Output | ADR-14 (kappa=0) MAE | ADR-18 (kappa=1) MAE | Delta |
|---|---|---|---|
| e_y (m) | 0.0219 | 0.0216 | -0.0003 |
| e_psi (deg) | 3.41 | 3.32 | -0.09 |
| confidence accuracy | 0.9948 | 1.0000 | +0.0052 |
| confidence invalid recall | 0.9444 | 1.0000 | +0.0556 |

No regression on any of the three; all four numbers moved slightly in the
improving direction (plausibly noise from a different training run rather
than a real effect of the added loss term -- not claimed as significant,
only as "not worse").

## VM inference frequency -- measured

Measured live on the VM via SSH (`bridge_node` + `perception_node` running
together, against a real `sim_server.py` on the Mac at 50 Hz state / 25 Hz
camera), after fixing the `bridge_node` message-drop bug below -- numbers
taken before that fix would have been measuring a starved, irregular image
stream, not the perception path itself.

**`/lane_state` publish interval** (4 windows of 300 frames each, ~65 s
total; `perception_node`'s own `stats_window_frames` instrumentation,
`perf_counter`, single VM clock):

| Window | mean (ms) | p50 | p95 | max | std |
|---|---|---|---|---|---|
| 1 (includes node startup) | 40.02 | 39.79 | 47.07 | 284.47 | 15.81 |
| 2 | 39.98 | 39.93 | 47.20 | 74.22 | 5.38 |
| 3 | 40.01 | 40.28 | 47.85 | 52.96 | 4.95 |
| 4 | 39.96 | 39.97 | 48.92 | 77.22 | 5.99 |

Mean interval ~40.0 ms = **~25.0 Hz achieved**, matching the camera rate
exactly -- once `bridge_node` isn't dropping images, `perception_node`
keeps up with every single frame. Window 1's 284 ms max is a one-time
startup transient (first inference after model load, page faults, no
steady-state meaning); windows 2-4 settle to a max of 53-77 ms and a much
tighter std (5-6 ms) -- that's the number to trust for steady-state jitter.

**Per-stage cost** (same windows, VM CPU): preprocess mean 0.83-0.88 ms,
forward mean 5.3-5.7 ms (p95 8.7-9.6 ms), publish (msg build) mean ~0.13 ms
-- forward still dominates preprocess here too, consistent with the Mac
measurement, though VM forward cost (~5.5 ms) is roughly 5x the Mac's
(~1.1 ms): expected for virtualized CPU, not a regression.

**Against the 20 Hz / 50 ms control-loop target:** perception's own compute
(preprocess + forward + publish) costs ~6.5 ms mean, ~10-11 ms at p95 --
13-22% of the 50 ms budget, leaving 78-87% headroom for acados. Separately,
worth flagging for M3: perception updates at ~25 Hz (40 ms) while the
control loop target is 20 Hz (50 ms) -- the two rates aren't a clean
multiple of each other, so a control tick will not always have a
brand-new `/lane_state` waiting; the controller needs to tolerate reusing
the previous frame's estimate on some ticks, not assume a fresh one every
time.

**End-to-end age (`header.stamp` -> publish):** mean 7.8-8.2 ms, p95
12.0-12.8 ms, max 15-49 ms (again, the 49 ms max is window 1's startup
transient). Single-clock, VM-only -- see the correction below for why this
does not include the Mac->VM ZeroMQ hop, and how to get the full figure.

**Correction, now fixed (ADR-13) -- this section describes what was true
when these numbers were measured, not current behavior.** At the time of
this measurement, `header.stamp` was not the Mac's camera-render time.
`bridge_node.py`'s `poll()`/`_process_frame()` stamped every image with
`self.get_clock().now()` -- the VM's own clock, at the moment the VM
received the ZeroMQ frame -- not a converted Mac timestamp. ADR-13 fixed
this: `bridge_node.py` now converts the decoded `t_sim` (MuJoCo's own
simulation clock) into the ROS stamp, per `docs/lane-state-contract.md`
section 3. The "end-to-end age" figures above were therefore graph-internal
latency only (executor dispatch + preprocess + forward + publish) at
measurement time; post-fix, the same measurement would also include the
Mac->VM ZeroMQ transit, since `header.stamp` now starts at true render
time rather than VM-receipt time -- re-measure after ADR-13 rather than
reusing these numbers as the current age figure.

The Mac->VM hop itself is measured separately, already clock-skew-corrected:
`bridge_node.py` publishes it on `/carsim/latency_ms` using ADR-4's
round-trip-sum method (one-way Mac/VM timestamp deltas were the ~30 ms
figure ADR-3/ADR-4 found to be predominantly clock skew, not real transit
time) -- measured at 2.5-6.1 ms during this same session, drifting upward
over the ~2 min run (the same clock-skew drift ADR-3/ADR-4 already
documented, not a new finding). The full Mac-render-to-`/lane_state` age is
the sum of that figure and the `header.stamp -> publish` number above --
never subtract a Mac timestamp from a VM one directly.

