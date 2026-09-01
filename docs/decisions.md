# Architecture Decision Log

Recorded at the time each decision was made, not reconstructed afterward. Format: context, options considered, decision, consequence.

---

## ADR-1 — Simulator: Gazebo → MuJoCo

**Context:** Gazebo was the initial choice for physics and camera simulation, run inside the Ubuntu VM alongside ROS2.

**Problem:** Gazebo's camera rendering requires OpenGL 3.3, which proved unworkable in the UTM/virgl graphics stack without GPU passthrough — not available on this hardware setup.

**Options considered:**
- Force software rendering in Gazebo (too slow, unreliable in early testing)
- Move the whole stack to a different VM/hypervisor with passthrough support (large scope increase, uncertain payoff)
- Split the simulator out from ROS2 entirely, run it natively on macOS instead

**Decision:** Run MuJoCo natively on macOS (Apple Silicon, no rendering constraint), ROS2 stays in the VM, connected by a network bridge.

**Consequence:** Forces a simulator/control-stack separation that mirrors real robot architecture — the simulator is a "plant" the control stack talks to over a hardware-like interface, not a component inside the ROS2 graph. This became a structural strength (see ADR-2) rather than a workaround. Cost: a bridge had to be designed and validated (M0), and macOS/VM clock skew became a real problem to solve (ADR-4).

---

## ADR-2 — Bridge transport: ZeroMQ over ROS2-on-macOS or raw sockets

**Context:** With MuJoCo on macOS and ROS2 in the VM, something has to move sensor state and commands across that boundary.

**Options considered:**
- Run ROS2 on macOS too (re-creates the exact problem the split was meant to avoid)
- Raw TCP sockets (works, but message framing — knowing where one message ends and the next begins — is left entirely to hand-rolled code)
- ZeroMQ (message-oriented, not stream-oriented; PUB/SUB pattern fits a one-to-one state/command link)

**Decision:** ZeroMQ, one PUB socket for state (Mac → VM), one SUB for commands (VM → Mac).

**Consequence:** No message-framing code to maintain. Automatic reconnection. The PUB/SUB pattern immediately raised the "slow joiner" problem (a subscriber's connection isn't instantaneous, so frames queue before it's ready) — surfaced early during M0 latency testing rather than as a mystery bug later.

---

## ADR-3 — Wire serialization: JSON + raw buffer, not pickle

**Context:** The bridge crosses a Python runtime boundary — historically also a version boundary (3.10 on macOS, 3.12 in the VM), though the environments were later aligned.

**Decision:** Serialize scalar state as JSON, image data as a raw byte buffer with a small header. No `pickle`, no Python-object-specific format.

**Consequence:** The protocol makes no assumption about what's on the other end of the wire. This matters beyond the Python-version issue that originally motivated it: it's what makes the bridge a genuine hardware-interface contract (FR-7) rather than an RPC mechanism tied to this specific codebase — the same property that makes a future physical-robot port a substitution of the plant, not a rewrite of the protocol.

---

## ADR-4 — Latency measurement: two-way sum, not one-way delta

**Context:** Initial one-way latency measurements (VM-side, comparing a Mac-stamped publish time to VM receive time) showed ~30–40 ms, with a slow but steady upward drift over each run.

**Investigation:** Three signals pointed away from "real network latency": the standard deviation (~0.5 ms) was far too tight for genuine network jitter over a VM boundary; the drift was monotonic over tens of seconds, the signature of two independent clocks slowly diverging rather than of network conditions changing; and a single-process reference test (no VM boundary at all) measured sub-millisecond latency with the same code path.

**Decision:** Measure latency as the sum of both transmission directions (VM→Mac plus Mac→VM), which cancels clock offset exactly regardless of its magnitude, rather than as a one-way difference (which requires synchronized clocks to be valid at all).

**Consequence:** True one-way latency characterized honestly at ~5 ms, not the ~30 ms artifact. A real, separate bug was found alongside this investigation: the camera publish rate was quantized to 25 Hz instead of the intended 30 Hz, because a render check was only evaluated on discretized control ticks. Fixed with an exact integer tick divisor rather than a floating-point rate threshold.

---

## ADR-5 — Freshness over completeness on the bridge

**Context:** ZeroMQ sockets buffer internally. If a consumer is briefly slow or just connecting, frames queue up and are served oldest-first by default.

**Decision:** Drain the socket to the newest frame only, every cycle, rather than processing every frame in order.

**Consequence:** A closed-loop controller acting on a 400ms-old state is acting on a state the simulation has already moved past — a pure delay injected into the loop, which eats stability margin. Dropped-frame counts in the bridge logs are expected and intentional, not a sign of a problem.

---

## ADR-6 — Milestone ordering: control validated before quantization

**Context:** Original milestone order had quantization (INT8) before MPC/control integration.

**Reasoning:** Developing and tuning a controller against a quantized perception model risks conflating two different error sources — control tuning error and quantization-induced perception error — with no clean way to separate them if something behaves badly in closed loop.

**Decision:** Validate the full closed loop (perception + MPC) against the FP32 model first. Quantize afterward, and benchmark INT8 against that known-good FP32 baseline, including closed-loop tracking behavior, not offline accuracy alone.

**Consequence:** A functional unquantized system is worth more than a quantized system that doesn't drive. If INT8 degrades tracking, that's a reportable result against a trusted reference, not a confound.

---

## ADR-7 — `/lane_state` sign convention, and a bug it caught

**Context:** Designing the perception→control contract (see `docs/lane-state-contract.md`), the sign conventions for `lateral_error`, `heading_error`, and `curvature` were fixed before any dataset generation, specifically to avoid a class of bug that's hard to spot once a controller is tuned around it.

**What happened:** Writing the acceptance tests for the projection routine caught two separate issues before any code shipped. First, the design doc's own written acceptance-test description was ambiguous about which of the vehicle or the lane tangent was being rotated — worth fixing in the doc even though the formal sign table was correct throughout. Second, and more substantively, the arc-projection implementation had a genuine sign bug: an angle-wrapping calculation applied the direction-of-travel sign twice for clockwise arcs, which silently zeroed the projection instead of raising an error. A left-turning-arc test passed by coincidence; a right-turning-arc test caught it immediately.

**Decision:** Keep the convention as specified in `docs/lane-state-contract.md` §2 (`lateral_error` centerline-relative-to-vehicle, `heading_error = tangent − vehicle heading`, both positive-left). Fix the projection code, not the convention — the bug was in the arc math, not in the sign choice.

**Consequence:** This is the reason the SRS risk register flags dataset labeling as the milestone-1 risk with the highest downstream cost (invisible until closed-loop testing at M3). It also argues for the value of writing the acceptance tests before the dataset generator, not after: this exact bug would have silently mislabeled every clockwise-turn sample in the dataset.

---

## ADR-8 — Dataset sample validity is defined by camera geometry, not by envelope tuning

**Context:** The first dataset generation pass sampled poses from hand-chosen envelopes: positives within ±0.7 m lateral of the centerline, negatives 1–4 m off. Both numbers were picked by judgement, not derived from anything.

**What visual inspection caught:** rendering ~50 sample images (rather than the full 2000) surfaced that several `valid=True` samples had the vehicle sitting fully outside the painted lane. The lane half-width is 0.4 m; the positive envelope allowed 0.7 m. The labels themselves were arithmetically correct — verified independently, since on a straight segment the projection reduces to `s = x`, `lateral_error = −y`, `heading_error = −heading`, which the data matched exactly. The defect was in the sampling envelope, not the labeling maths.

**What the first fix attempt got wrong:** the initial hypothesis was that the camera had lost sight of the lane. Building a geometric visibility check and running it against the flagged samples disproved this — all of them had 90+ lane boundary points inside the image. Visibility was never the issue; the vehicle simply being outside the lane was.

**What the visibility check found instead:** turning the same tool on the *negative* samples showed that roughly half of them still had lane markings clearly in frame. On a 45 m closed loop, a vehicle even 5–12 m off course usually has some other part of the loop in view — sweeping the envelope showed no offset/heading combination that reliably hides the track. Those samples would have trained the confidence head to output zero on perfectly readable images. A 50-image spot check would not have caught this: only ~5 of those images are negatives.

**Decision:**
1. The positive lateral envelope is derived from `LANE_HALF_WIDTH`, not set independently, so the two cannot drift apart.
2. Sample validity is decided by rejection sampling against the actual camera frustum: a positive must have the lane it should follow visible ahead; a negative must have no lane marking anywhere in frame. Rejection sampling replaces envelope tuning as the mechanism for correctness.
3. The analytic camera model used by the filter is pinned to MuJoCo's own computed extrinsics by a unit test, agreeing to 1e-9 m. This keeps the generator headless and fast (1.7 s for 2000 labels including rejection) without the model silently diverging from the MJCF.

**Consequence:** Negatives on this track are necessarily "vehicle far off in empty space" cases (mean 6.1 m off centerline). That is a somewhat easy negative — the confidence head will learn "no markings visible" rather than anything subtler about ambiguous perception. Worth stating plainly in the perception write-up rather than implying the confidence signal is more sophisticated than it is. The alternative, negatives with markings visible but mislabeled, would be actively harmful.

**Method note:** both defects were found by checking properties over the whole dataset programmatically, not by looking at rendered images. Visual inspection found the first one and pointed in the wrong direction as to its cause; the sweep found the second, which was the more damaging and was invisible to sampling by eye.

---

## ADR-9 — Train/val/test split: per-primitive stratification, not a single loop-wide cut

**Context:** The M1 no-leakage requirement was implemented as a single 70/20/10 cut across 10 equal-length zones spanning the whole 45 m loop (`SPLIT_ASSIGNMENT`, contiguous arc-length blocks — see the design note in `generate_dataset.py`). This correctly prevents near-duplicate poses from straddling splits, which was the property it was built to guarantee.

**What checking split composition programmatically caught:** writing `perception/dataset/plot_label_distributions.py` (the M1 histogram deliverable) and printing split × curvature-bin counts, rather than just plotting them, showed the test split contained **zero** straight-line samples and **zero** R=3 m samples — every test sample was an R=5 m left turn. The val split had zero R=3 m samples. Only train ever saw the tight radius.

**Root cause:** `REFERENCE_TRACK` is built from 8 primitives (2 identical halves, each straight–arc(R=3)–straight–arc(R=5)). A single global 70/20/10 cut puts val and test entirely in whichever primitives happen to fall at the end of the loop — on this track, that window landed inside one straight and one R=5 m arc, and never touched the R=3 m arcs at all. The zone width (4.5 m) was large enough to prevent leakage but was never checked against "does every split see every curvature value," which is a different property than leakage.

**Consequence if shipped as-is:** M2's exit criterion is validation error "documented in physical units" — but a test set containing only one curvature value cannot measure tracking error on straights or tight turns at all. This would have been invisible until M3 closed-loop testing exposed a model that never learned to generalize past R=5 m curves, exactly the failure mode the SRS risk register flags dataset imbalance for.

**Decision:** Apply the 70/20/10 split independently **within each track primitive** (`zone_for_s` in `generate_dataset.py`), rather than once across the whole loop. Every primitive — and therefore every curvature value on the track — now contributes its own 70/20/10 to train/val/test, guaranteed by construction rather than by luck of where the loop-wide cut happens to fall.

**Trade-off accepted:** this raises the number of train/val/test boundary points around the loop from 2 to 16 (one pair per primitive instead of one pair total), which is more opportunities for a train sample and a val/test sample to land close together in arc length `s`. Checked directly on the generated seed-42 dataset: the closest cross-split pair in `s` is ~6 mm, but every such close-in-`s` pair differs substantially in `lateral_error` and/or `heading_error` (sampled independently per pose), so none are near-duplicate images — the leakage property the original design protected still holds in practice, not just in theory. Regression tests: `test_split_has_no_leakage_across_track_zones` (split is a pure function of `zone_for_s(s)`) and `test_every_curvature_bin_present_in_every_split` (the defect this ADR documents, made permanent).

---

## ADR-10 — Mirror augmentation: the dataset had zero right turns

**Context:** `REFERENCE_TRACK` is a closed loop traversed in one direction, built entirely from LEFT turns — two identical halves, each a straight followed by a 90° left arc (`track_definitions.py`). This is a property of the track geometry, not of the sampling code.

**What looking at `label_distributions.png` caught:** the curvature-bin bar chart (`perception/dataset/plot_label_distributions.py`, the M1 histogram deliverable) has three bins — `κ ∈ {0, 1/5, 1/3}` — and every one of them is non-negative. All 1800 valid samples in the 2000-sample dataset had `curvature >= 0`. Every check written so far (ADR-8's visibility filter, ADR-9's per-split curvature coverage) operates on curvature *magnitude* or on visibility, both of which hold identically whether curvature can go negative or not — so nothing in the existing test suite could have caught this. It took literally looking at the bar chart to see that three bins meant three *magnitudes*, not five signed values.

**Consequence if shipped as-is:** a CNN trained on this data would learn "curvature is non-negative" as a hard prior, not something inferred from the image, and would never predict a right turn regardless of what the camera sees. Worse, this would be invisible in validation, since val/test are drawn from the exact same left-only distribution as train (per ADR-9, correctly stratified — but stratified over a distribution that itself only contains one sign).

**Decision:** Mirror-augment every sample. For each rendered pose, add a second row (`mirror_row()` in `generate_dataset.py`) with the image horizontally flipped and `lateral_error`, `heading_error`, `curvature` all negated; `x`/`y`/`heading`/`s`/`confidence`/`valid`/`split` are copied unchanged from the source (there is no "mirrored world pose" to invent — mirroring is a label-and-image-space transformation applied to the same rendered pose, not a new place to put the vehicle). A mirror is never assigned a different split than its source, since split is a pure function of the unchanged `s`.

**Why the mirror is geometrically exact, not approximate:** `cam_front` has zero lateral offset in `sim/models/car.xml` (`pos="0.16 0 0.05"`, forward-only) and zero yaw/roll in its mount. Verified directly against `point_in_camera_frame`: negating `(vehicle_y, vehicle_heading)` negates the camera-frame `x`-coordinate exactly and leaves camera-frame `y` (image row) and depth exactly unchanged — i.e. mirroring the world across the vehicle's forward axis is *identical* to flipping the rendered image left-right, to floating-point precision, not an approximation that degrades off-axis.

**Rendering:** mirrors are never re-rendered through MuJoCo — `render_dataset_images.py` renders every base pose once, then produces each mirror by loading the corresponding source PNG and flipping it (`Image.FLIP_LEFT_RIGHT`). Cheaper than re-rendering and removes any risk of the analytic mirror and the renderer's own left/right convention silently disagreeing.

**Consequence:** dataset size doubles, 2000 → 4000 (1800 → 3600 valid), with curvature now exactly symmetric: 403 samples at each of `κ=±1/3`, 599 at each of `κ=±1/5`, per split as well as overall (`test_dataset_doubles_and_covers_both_curvature_signs`, `test_mirror_row_negates_lane_state_and_preserves_everything_else`). Per-channel normalization statistics (`perception/dataset/normalization_stats.json`) are numerically unchanged by this — a horizontal flip doesn't change a pixel value's distribution, only its position — which is itself a useful sanity check that mirroring didn't silently corrupt anything.

**Method note:** same pattern as ADR-8 and ADR-9 — a property of the *whole dataset* (every sample sharing one sign) that is invisible in any single sample, any per-sample unit test, or any check phrased in terms of magnitude. The histogram deliverable that M1's own risk register calls for (SRS: "Mandatory visual inspection of labels before training") is what caught it, not the rejection-sampling tests, which is exactly the failure mode that requirement exists for.

---

## ADR-11 — Camera resolvability limits, and what v0 does and doesn't support

**Context:** `camera_visibility.py`'s extrinsics were verified against MuJoCo to 1e-9, but its projection — fovy-to-pixel mapping, aspect handling, axis conventions — had never been checked against an actual render, and `LOOKAHEAD_M = 6.0` was set by intuition with no measurement behind it. A verification pass added `project_to_pixel` (exposing pixel coordinates the module was already computing internally), an independent numeric test against MuJoCo's own scene-camera frustum (agrees to <4e-5 px), an end-to-end render check using injected colour markers (agrees to <1.2 px, against a 2 px budget), a coverage contact sheet, and a resolvability report (`tools/camera_resolvability.py`, `docs/camera-resolvability.md`).

**What was measured — the camera can usefully see 2.36 m ahead, not 6 m:**

| Limit | Value | Meaning |
|---|---|---|
| `L_resolvable` | 13.25 m | projected lane width falls below 10 px at CNN input resolution |
| `L_separable` | 2.44 m | two points 0.5 m apart stop being more than 2 image rows apart |
| `L_representable` | 2.36 m | `tan(s/R_min) > 1` at `R_min = 3 m` — where the quadratic preview path stops fitting a circular arc |

`L_usable = min(...) = 2.36 m`. The binding constraints are depth resolution and path-representation validity, not pixel width — lane width stays resolvable out to 13 m, so that was never the bottleneck. Depth resolution collapses fast: by `docs/camera-resolvability.md`'s own table, a ground point at row 236.7 (0.25 m out) is at row 75.0 by 5 m and row 72.0 (essentially the 71.7-row horizon) by 16 m — almost the entire usable row range is spent on the first couple of metres.

**Findings:**

1. **`LOOKAHEAD_M = 6.0` exceeds `L_usable` by 2.5x.** Points counted as visible beyond 2.36 m carry little to no usable information, so the rejection filter is weaker than its constants suggest. Left unchanged for v0 — tightening it now would invalidate the already-generated dataset — revisit when the track is regenerated (see finding 5).

2. **`MIN_VISIBLE_POINTS = 6` is not satisfied by far-field clusters, but the boundary is visually fragile.** The "far-field sub-pixel points" concern is refuted on real data: at the acceptance boundary, surviving points are near-field (0–0.5 m) and 29–64 px apart, because heading error displaces distant points more than near ones, so far points leave the frame first. But the contact sheet (`docs/dataset/camera_coverage_contact_sheet.png`) shows `straight/near-reject` (4/120) and `R=5m/near-reject` (5/120) both still show a clearly visible road band, not meaningfully different by eye from `straight/near-accept` (6/120) — confirmed by direct inspection of those three tiles. The point count is a proxy for usability, not a measurement of it. Distance-weighting the count is a reasonable future robustness improvement; not implemented.

3. **Labels are exact at the projection point, and invalid across a curvature transition inside the preview window.** `compute_lane_state` evaluates `(lateral_error, heading_error, curvature)` exactly at the single projection point — there is no windowed fit anywhere in the labeling pipeline (confirmed by reading the function), so this is not the ADR-8/ADR-9 failure mode again. The real gap: when the vehicle is within `L_usable` of a curvature change, the true path over the preview window spans two curvatures while the label carries only the one at the projection point. `REFERENCE_TRACK` has exactly 8 primitive boundaries (every straight↔arc join, all 8 confirmed to be genuine curvature discontinuities), so `T=8` transitions around the 45.13 m loop — the affected fraction is `T · L_usable / total_length = 8 × 2.36 / 45.13 ≈ 42%` of the loop's arc-length. Not a corner case. Not fixable by augmentation; needs either a windowed fit over the visible arc or continuous track curvature (clothoid transitions). Deferred to a v1 track.

4. **Curvature is 5 discrete signed values (0, ±1/5, ±1/3), not a continuous target.** Regressing curvature on v0 is a 5-class classification problem in disguise: near-zero error is achievable by memorizing five constants, and no metric distinguishes that from reading road geometry. This is unrelated to, and not fixed by, ADR-10's mirror augmentation (which fixed the *sign* imbalance, not the discreteness). Consistent with the contract's existing decision to omit a cubic (`dκ/ds`) term (`docs/lane-state-contract.md` §1) — `dκ/ds` is zero everywhere on this track except at the same 8 discontinuities, so a cubic would be over-parameterized here regardless.

5. **No split of v0 measures generalization.** Both R=3 m arcs are geometrically identical to each other (same radius, same construction), and likewise both R=5 m arcs — under identical lighting and texture. A random split measures interpolation between near neighbours in arc-length; ADR-9's per-primitive split (correctly) reserves val/test poses from the *same* physical arcs the model trains on, since there are no other arcs to hold out. No partition of v0 tests whether the model generalizes to unseen geometry, only whether it interpolates within seen geometry. Validation/test metrics on v0 must be reported as measuring interpolation, not generalization, wherever they're written up.

6. **Negative samples are structurally easy, and the mechanism needs correcting from what was assumed.** ADR-8 already flagged that v0 negatives are "somewhat easy" (mean 6.1 m off centerline, markings absent by construction). Measured directly here: 100 sampled negative images have mean luminance 119.1 (± 1.0, a strikingly tight range) versus 122.7 (± 5.6) for positives — off-track ground is unlit and untextured, exactly as uniform as ADR-8 predicted. But a single global mean-luminance threshold does *not* cleanly separate them on this data (positive minimum 119.1 falls inside the negative range) — "solvable by a luminance threshold" is not quite right; "structurally homogeneous regardless of where off-track the vehicle is" is the accurate framing, and still means the confidence head can learn a shortcut rather than genuine perceptual ambiguity.

7. **The visibility filter runs on the full 320×240 frame; the CNN-input-crop gap is currently zero, but latent.** `perception/dataset/cnn_input_config.json` is an explicit no-op placeholder (full frame) since M2 hasn't designed the input pipeline yet. The moment a real crop is defined, `lane_is_visible`/`any_lane_visible` need to be recomputed against it, or visibility counts will be systematically inflated relative to what the network actually receives.

**Decision:** Proceed to M2 on the v0 track and dataset, with these limitations recorded rather than corrected now — the first priority is a working perception→control loop end to end, not track complexity. `L_usable = 2.36 m` is treated as a hard constraint on the M3 MPC preview horizon. The CNN is dimensioned against the eventual degraded-domain case (varied lighting, worn markings, real texture), not against v0 — on v0 it is deliberately over-capacity, so that adding domain variation later means regenerating data and retraining, not redesigning the architecture.

**Consequence:** `perception/README.md` and `control/README.md` (per the naming already set in `docs/portfolio-page-plan.md`) must carry findings 3, 4, 5, and 6 when written, and `control/README.md` inherits `L_usable = 2.36 m` as a measured, justified horizon bound rather than a tuned one. A v1 track — continuous curvature via clothoid transitions, varied radii, both turn directions built into the geometry itself rather than mirrored — is what addresses findings 3, 4, and 5; domain variation (lighting, texture) is what addresses finding 6. Both out of scope for M2, both scoped here for later.

**Artifacts:** `docs/camera-resolvability.md` / `.png`, `docs/dataset/camera_coverage_contact_sheet.png`, `tools/camera_resolvability.py`, `tools/visualize_camera_coverage.py`, `tests/test_camera_visibility.py` (`test_project_to_pixel_matches_mujoco_frustum`, `test_project_to_pixel_matches_rendered_markers`).

---

## ADR-12 — The bottom of every render is skybox, not ground: MuJoCo's near-clipping plane, not the vehicle or the floor's extent

**Context:** Building the M2 preprocessing contact sheet (`perception/model/preprocessing_contact_sheet.png`) surfaced a light-blue band filling the bottom of every cropped training image. Two hypotheses were raised and checked directly rather than assumed:

1. *The vehicle's own roof/hood.* Checked `sim/models/car.xml`: the chassis body has exactly one geom (`type="box"`, red `rgba="0.75 0.2 0.2 1"`) — no roof, hood, or windshield geom exists. `cam_front` sits at local `x=0.16`, almost exactly the chassis box's front face (`half-length 0.17`), facing forward — geometrically there is nothing of the vehicle's own body ahead of it to see. Ruled out.
2. *The 80x80 m floor plane doesn't extend far enough.* Ray-cast from the camera through the artifact rows (`mujoco.mj_ray`) hit geom `"floor"` at every single row tested, including the closest (row 239, ray distance 0.153 m) — the floor geometry genuinely extends there. Ruled out.

A third hypothesis (low `reflectance="0.05"` on the floor's `grid` material picking up the sky via a Fresnel-like effect at the shallow near-camera viewing angle) was tested directly: re-rendering with reflectance forced to 0 left the band completely unchanged. Ruled out.

**Diagnosis:** MuJoCo's near-clipping plane. `model.stat.extent` (auto-computed from the scene's overall bounding size, dominated by the floor's `size="40 40"`, i.e. 80x80 m against a ~45 m track) = 18.51. `vis.map.znear` (unset in the MJCF, so MuJoCo's default) = 0.01. The absolute near-clip distance is their product: `18.51 x 0.01 = 0.1851 m`, matching `frustum_near` read directly from `mjvGLCamera` to 8 significant figures. Any ground point closer than this along the camera's view axis cannot be rasterized and falls through to the skybox. Swept the ray-cast row by row: the crossing from real-ground to past-near-clip lands at row 182, one pixel from the empirically measured artifact boundary (row 181, identical across 25 poses spanning every curvature bin and lateral/heading offset in the sampling envelope — a fixed camera-geometry effect, not pose-dependent). Rows 182-239 — **37% of the M1 crop's kept height (rows 80-240)** — are skybox in every one of the 4000 rendered training images.

**Why the fix is not the floor's `size`:** shrinking the floor to something proportionate to the track (e.g. `15 15`) would shrink `stat.extent` and the near-clip proportionally, appearing to fix this — but only as a side effect of the floor's size feeding MuJoCo's auto-computed extent. It would silently regress the moment the floor is resized again for any other reason (a wider track, added scenery, a different loop), with no error or warning, since nothing ties the fix to the actual cause. The direct fix is `<visual><map znear="..."/></visual>` (or an explicit `<statistic extent="..."/>`), which decouples the near-clip from scene scale entirely and states the real constraint (how close the camera needs to see) rather than an indirect one (how big the floor happens to be).

**Verified, not assumed, how much this actually buys back:** rendering with `znear="0.002"` (`near = 0.002 x 18.51 = 0.037 m`) removes the artifact completely within the visible frame, not partially — real, lit ground renders all the way to the bottom row (row 239, ray distance 0.153 m, comfortably beyond the new 0.037 m near-clip). This is a stronger result than initially assumed when this ADR was scoped (a rough estimate of "~5 cm of near-field ground recovered" turned out to undersell it substantially once actually measured) — a one-line MJCF change fully eliminates a defect present in 100% of rendered images, not a partial mitigation.

**Decision:** Do not apply the fix now. `data/dataset_v0`'s 4000 images are already rendered; changing the MJCF and re-rendering is worth bundling with the v1 track work (ADR-11's finding 5 already calls for a v1 track with continuous curvature), not spending a second full render pass on the v0 geometry that's being replaced anyway. M2's crop was moved to rows 80:182 instead (`perception/dataset/cnn_input_config.json`) to exclude the artifact from the current dataset without touching the MJCF or re-rendering.

**Consequence:** the crop fix (rows 80:182, not 80:240) is a stopgap specific to v0; it does not carry forward automatically if v0 is regenerated with different camera parameters. When the v1 track forces a re-render, apply `<visual><map znear="0.002"/></visual>` (or a value derived the same way, verified the same way — don't reuse this number without re-measuring against v1's own `stat.extent`, which will differ from 18.51 once the floor/track size changes) to the MJCF directly, and the crop can revert to using the full available row range.