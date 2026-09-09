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

---

## ADR-13 — `bridge_node.py` stamps with wall-clock time, not simulation render time: a contract §3 conformance fix

**Context:** `docs/lane-state-contract.md` §3 specified render-time semantics for `header.stamp` at M1, before any dataset generation started — *"header.stamp is the simulation time at which the camera frame was rendered, propagated end to end... Not the publish time, and not VM wall-clock time"*, precisely to make message age computable without the Mac/VM clock-skew error ADR-3/ADR-4 already found once. The M3 kickoff investigation (contract audit + loop timing prerequisites) checked whether the implementation actually does this and found it does not: `bridge_node.py`'s `_process_frame` stamped every outgoing message with `self.get_clock().now().to_msg()` — the VM's own wall clock at ZeroMQ-receipt time — while `protocol.py`'s wire format already carries `t_sim` (`sim_server.py`'s `data.time`, MuJoCo's own simulation clock, encoded via `encode_state(seq, t_sim, ...)`), decoded into `header['t_sim']` on the VM side but never read for the stamp. This was not a new design question; it was un-implemented §3.

**Diagnosis, why this stayed unnoticed:** the bug doesn't produce an obviously wrong age — VM-receipt time is only a few milliseconds behind true render time under normal operation (the Mac→VM transit is short), so age computed from the wrong clock still *looks* small and plausible. It systematically drops the Mac→VM ZeroMQ transit time from every age estimate, silently, with no error — exactly the failure mode §4's staleness fallback exists to prevent, and exactly the kind of clock confusion §3 was written to rule out by construction.

**Decision:** Convert `t_sim` into a ROS `Time` and use it as `msg.header.stamp` in both `make_odom` and `make_image` (`carsim_bridge/carsim_bridge/bridge_node.py`), via a shared module-level `sim_time_to_stamp(t_sim)` — pure function of the decoded value, independently testable without a live node. `_process_frame` no longer computes a stamp itself; both message builders derive it from the `header` dict they already receive. Added `tests/test_bridge_node.py` asserting `make_odom`/`make_image`'s stamps derive from `header['t_sim']` and are *not* wall-clock-shaped (an epoch-time regression would produce a `sec` field north of 1.7 billion; a sim-time-correct one stays under 10,000 for any realistic sim run), so a future regression back to `self.get_clock().now()` fails a test rather than silently drifting the age computation again.

**Consequence:** every downstream consumer of `header.stamp` — `/carsim/odom`, `/carsim/image_raw`, and transitively `/lane_state` (`perception_node.py` propagates the image's stamp unchanged) — now carries true render time, matching §3 for the first time. `/carsim/latency_ms` (ADR-4's round-trip-sum figure) is unaffected — it was already computed from `t_pub`, not the message stamp. M3's delay-compensation and staleness-fallback logic can now trust `header.stamp` age as specified, not as approximated by VM-receipt time. Not verified live on the VM this session (no SSH access at the time of this fix — verified independently earlier in the M2 closing work, but that VM session has since ended); the regression test and a direct read of the fixed code are what this decision rests on until the next live run confirms it.

---

## ADR-14 — kappa's loss weight is 0, and `curvature` publishes as a hardcoded `0.0`

**Context:** this decision was made and shipped during M2 (the CNN training loss config, and `perception_node.py`'s publish logic), but was never recorded as an ADR — `results.md` and code comments cite "ADR-12" for it, and ADR-12 is entirely about the near-clipping/skybox crop artifact and never mentions kappa. Caught during the M3 kickoff investigation (`grep -n "kappa" docs/decisions.md` returns nothing) and corrected here: this entry documents the decision that already exists in the code, timestamped at the point the gap was found and closed, not backdated to when the decision was originally made.

**Reason (ADR-11 finding 3, `perception/model/kappa_transition_proximity.png`):** the point-wise Frenet `curvature` label is exact only at the vehicle's projection point; the preview window the camera can actually see (`L_usable = 2.36 m`, ADR-11) frequently spans a curvature discontinuity the label doesn't know about. `REFERENCE_TRACK` has 8 primitive joins around its 45.13 m loop, so `T · L_usable / total_length ≈ 42%` of the loop's arc-length carries a label that describes the vehicle's exact position, not what the camera sees ahead of it. Measured, not inferred: on straight samples, mean `|kappa_pred|` correlates at `r = -0.67` with distance to the next transition (0.023 beyond `L_usable`, 0.115 within it — a 5x gap), and per-component loss curves show kappa's *training* loss converging normally while its *validation* loss flatlines from ~epoch 5 — the network reads the road correctly and memorizes training-specific mappings that don't generalize, not a failure to learn at all.

**Options considered:**
1. *Train kappa anyway, publish its raw output.* Rejected: a kappa output that predicts curvature correctly on straights approaching a bend and near-zero once the bend actually starts would actively steer an MPC feedforward term off a straight line at exactly the point curvature is most needed — worse than publishing nothing, not merely unusable.
2. *Zero the loss weight, drop the kappa head entirely.* Rejected: the head costs nothing to keep in the architecture (the flatten→fc layers already exist for e_y/e_psi/confidence), and a future windowed-fit relabeling (item 3 of the M3 kickoff investigation — feasible without re-rendering, flagged in `control/README.md` open questions, not decided here) could retrain the same head without an architecture change.
3. *Zero the loss weight, publish a hardcoded `0.0`.* **Chosen.**

**Decision:** `training_config.yaml`'s `component_weights.kappa = 0.0` (the head trains on nothing, `component_losses` still logs its raw unweighted loss for monitoring only). `perception_node.py`'s `on_image` publishes `out.curvature = 0.0` unconditionally, never the model's raw head output, regardless of what that output happens to be for a given frame.

**Consequence:** `/lane_state`'s `curvature` field is a known constant on v0, not a signal — every downstream consumer (control, ADR-15) must treat it as absent rather than zero-valued-but-real, which is exactly what ADR-15 records for the MPC formulation. `perception_node.py`'s comments, `perception/model/results.md`, and `perception/README.md` cited "ADR-12" for this decision; corrected to cite ADR-14 (this entry) in the same pass that added it, since leaving a live mis-citation next to a newly-written correct one would be worse than the original gap.

---

## ADR-15 — MPC problem formulation: Frenet vs. body-frame tracking, and the cost of no curvature feedforward

**Context:** `/lane_state` publishes three scalars — `lateral_error`, `heading_error`, `curvature` — reconstructed by the contract (`docs/lane-state-contract.md` §1) into `y(x) = c0 + c1 x + c2 x^2`, `c0 = lateral_error`, `c1 = tan(heading_error)`, `c2 = curvature/2`. With `curvature` hardcoded to `0.0` (ADR-14), `c2 = 0` always: the reconstructed preview path is a straight line regardless of the vehicle's true position on the track.

**Options considered — Frenet error regulation vs. body-frame path tracking:**
1. *Frenet-frame error regulation* (drive `lateral_error`/`heading_error` to zero directly, curvature entering as a feedforward correction to the steering command).
2. *Body-frame path tracking* (reconstruct `y(x)` and track the quadratic as an explicit reference path in vehicle coordinates, curvature entering as a term in the path itself).

With `c2 = 0` on v0, these two formulations are **numerically identical** — the quadratic degenerates to `y(x) = c0 + c1 x`, a straight preview line either representation reduces to the same tracking problem. This is therefore not a performance decision (there is no performance difference to measure on v0) but an **evolvability** one: which formulation is cheaper to extend if a real curvature signal arrives later (ADR-14's item-3-flagged windowed relabeling, or a v1 continuous-curvature track).

**Decision:** Frenet-frame error regulation. Reason: it consumes `lateral_error`/`heading_error`/`curvature` directly as the three physical scalars the contract already guarantees (§1, §6 — "the message is always in SI units," no lookahead-point convention needed), with curvature entering the cost/dynamics as a feedforward term that's a no-op exactly when it's `0.0`. Body-frame path tracking would require re-deriving the quadratic's coefficients back out of the three scalars inside the controller (undoing the contract's own §1 "equivalence" transform) for no benefit while curvature stays zero, and would need re-deriving again the day it stops being zero. Frenet regulation changes by exactly one term (curvature feedforward going from a hardcoded no-op to a real value) when that day comes; body-frame tracking would need its reference-path reconstruction revisited.

**Cost of running without feedforward, computed:** the standard steady-state cross-track error of a proportional-only lateral steering law `delta = -K1 * e_y` (no curvature feedforward) tracking a path of constant curvature `kappa`, kinematic bicycle, wheelbase `L`: `e_y_ss = -L * kappa / K1`. At `R_min = 3 m` (`kappa = 1/3 = 0.333 1/m`) and `L = 0.26 m` (`sim/models/car.xml`, item 6 of the M3 kickoff investigation — front axle at `x=0.13`, rear at `x=-0.13`, cross-checked against `sim_server.py`'s hardcoded `WHEELBASE = 0.26`): `L * kappa = 0.0867 m`. No lateral-error gain has been tuned yet (M3 controller design not started), so `K1` is not a real number yet either — stating the formula with an illustrative `K1 = 1.0 (rad/m)` as a placeholder gives `e_y_ss ≈ -8.7 cm`, comparable to the lane's own half-width (`LANE_HALF_WIDTH = 0.4 m`, i.e. ~22% of it) on the tightest turn on this track. Because `e_y_ss` scales as `1/K1`, this number moves once a real gain exists; the formula and the `L * kappa = 0.0867 m` product are what should be recomputed against the tuned `K1`, not this illustrative figure re-cited as final.

**Consequence:** the MPC cost function and constraints should be written against Frenet-frame errors directly. `control/README.md` inherits `L_usable = 2.36 m` (ADR-11, re-derived against the actual crop in the M3 kickoff investigation) as the preview horizon bound this formulation's curvature feedforward term must never be evaluated beyond.

---

## ADR-16 — Kinematic, not dynamic, bicycle model

**Context:** the MPC's prediction model needs a vehicle model. A kinematic bicycle (no tire slip, geometric Ackermann relation between steering angle and turn radius) is simpler to formulate and solve in real time than a dynamic bicycle model (tire forces, slip angles, requires a tire friction model). The kinematic assumption is valid only while lateral acceleration stays well below the point tire slip angles become significant — an assumption that needs justifying against this vehicle's actual operating envelope, not assumed by default.

**Measured:** at `v = 1.0 m/s`, `R_min = 3 m` (the tightest turn on the track): `a_lat = v^2 / R_min = 1.0^2 / 3.0 = 0.333 m/s^2 = 0.034 g`. `sim/models/car.xml`'s wheel geoms declare `friction="1.5 0.005 0.0001"` (sliding coefficient `mu = 1.5`), giving a theoretical grip ceiling around `1.5 g` in the simulator — `0.034 g` is roughly 2% of that ceiling, not a binding constraint at this operating point.

**Decision:** kinematic bicycle model, for the operating envelope measured so far (`v <= 1.0 m/s` at `R_min = 3 m`). The relevant threshold for when this stops holding is not the simulator's tire-friction ceiling (`1.5 g` is far above any speed this vehicle scale would actually reach) but the standard engineering rule of thumb for kinematic-model validity — small tire slip angles, conventionally taken as holding up to roughly `0.3-0.4 g` lateral acceleration for a road vehicle, independent of surface `mu`, since the assumption being protected is geometric (no-slip steering) rather than a grip-limit test. At `R_min = 3 m`, `0.3 g` is reached at `v = sqrt(0.3 * 9.81 * 3) ≈ 2.97 m/s` — **roughly 3x the currently measured operating speed.**

**Consequence:** kinematic bicycle is justified for the speeds M3 is targeting. If later testing pushes speeds toward ~3 m/s on the `R_min = 3 m` turn (or a tighter turn is added), re-measure `a_lat` against this same `0.3-0.4 g` threshold before continuing to assume no-slip steering — this ADR's justification is speed-and-radius-specific, not a permanent property of the vehicle.

---

## ADR-17 — Cubic term rejected: window asymmetry makes `c3` unreliable on v0, independent of the kappa relabeling question

**Context:** `docs/lane-state-contract.md` §1 already rejected a cubic (`dkappa/ds`) term at M1, on the grounds that it "adds a noisy, low-value term to the path the MPC sees." The M3 kickoff investigation ran the numeric experiment that decision anticipated but hadn't yet measured: does `c3`, fit over the camera's actual visible window, carry real transition geometry, or is it a fit artifact?

**Measured (`REFERENCE_TRACK`, no CNN, no rendering):** fit quadratic and cubic `y(x)` in the vehicle frame over the item-4 visible window (ground distance 0.32-2.42 m, the actual crop's near/far bound, not a theoretical symmetric window) at 96 poses positioned so a primitive join falls inside the window, and 48 poses positioned so the window stays within a single primitive. Near-join mean `|c3| = 0.0232`; away-from-join baseline mean `|c3| = 0.0126` — a real difference (~1.8x), and the absolute RMSE improvement from adding `c3` is ~2x larger near joins (2.8 mm vs 1.4 mm).

**That baseline is not clean, and this is the deciding evidence:** an isolating check on a single pure arc (constant curvature, zero discontinuity anywhere in range) gives `c3 = 0.0439` fit over the actual camera window — comparable to or larger than several near-join samples — versus `c3 ≈ 0.0001` (numerically zero) for the identical pose fit over a window centered on the vehicle. The cause is geometric, not a bug: a circular arc's `y(x)` is an even function (zero odd-order coefficients) only when the fit window straddles the frame's own origin. The camera's actual visible window never does — the vehicle cannot see the ground directly under or near itself (ADR-12's near-clip), so the window is permanently offset from `x = 0`. Any smooth arc, join or no join, produces a nonzero `c3` under this windowing; the join signal (1.8x) sits inside a noise floor of comparable magnitude that is a fixed property of the camera geometry, not something a different track or sampling strategy on v0 can avoid.

**Decision:** do not add a cubic term to the `/lane_state` contract, and do not attempt cubic relabeling of the existing dataset. This is independent of, and should not be conflated with, the separate question of whether a *quadratic* (kappa) windowed relabeling is worth doing (item 3 of the M3 kickoff investigation, feasible without re-rendering) — that decision is flagged, not made, in `control/README.md`'s open questions.

**Consequence:** confirms `docs/lane-state-contract.md` §1's original quadratic-only design under a direct measurement, rather than leaving it as an untested assumption. If a v1 track someday changes the camera mount (height/pitch) such that the visible window can be made more symmetric about the vehicle origin, this specific finding would need re-measuring against the new geometry — it is a property of this camera's window, not a permanent property of cubic fits in general.

---

## ADR-18 — Windowed-relabeled curvature is usable; supersedes ADR-15's "curvature unavailable" premise

**Context:** item 3 of the M3 kickoff investigation found `compute_lane_state`'s dependencies (`Track.point_at`/`heading_at`/`project`, `perception/dataset/geometry.py`, zero MuJoCo dependency) plus each sample's already-stored `x`/`y` are sufficient to relabel curvature for all 4000 samples without re-rendering. This ADR is that experiment, run to completion: relabel, retrain, measure, decide.

**Two mechanisms were tried; the task's own validation gate rejected the first before any dataset was touched.**

1. *Cartesian windowed quadratic fit* — `y(x) = c0 + c1 x + c2 x^2` fit to the true centerline in vehicle-frame coordinates over the camera's actual visible window (`[0.32, 2.42]` m ahead, item 4's re-derivation, not the theoretical `L_usable` bound). Validated away from any primitive join, where it should agree closely with `compute_lane_state`'s point-projection kappa (both describe the same constant-curvature ground truth there) — and did not: `41%` relative error on the `R=3m` arc, `13%` on `R=5m`, both with zero discontinuity present (a dead-straight segment fit exactly, `0%` — confirming the bias is specific to true curvature, not the off-center window in general). Cause: degree-4 (quartic) content in a circular arc's `y(x)` expansion leaks into the fitted quadratic coefficient over a window this size, worse for tighter curves. A weighting scheme was checked before concluding: it reduces the bias only by concentrating weight so heavily near the window's near edge that the fit degenerates into a point-curvature estimate there, at which point evaluating `track.curvature_at` directly at a forward-shifted `s` is simpler and more accurate than fitting anything. Per the investigation's own stop condition ("if they don't [agree], stop and report rather than proceeding, since it means the fit itself is wrong"), this was not carried further. Kept in `perception/dataset/windowed_relabel.py` as `windowed_lane_state`, a real, tested, documented negative result (`tests/test_windowed_relabel.py`, `xfail(strict=True)` on the two arc cases, not silently dropped) — not the method used.

2. *Curvature-space windowed averaging* — `windowed_curvature_average`: a weighted average of `track.curvature_at(s)` itself over the arc-length actually visible in the same window, no polynomial fit on position at all. This mechanism cannot inherit the failure above by construction: curvature is exactly constant along a pure arc, so averaging a constant (any weighting) recovers it exactly, not approximately. Validated: `0.00000000` diff away from any join, all three primitive types (straight, `R=3m`, `R=5m`) — a mathematical certainty for this method on a piecewise-constant-curvature track, not a favorable measurement. Near a join, the average is a convex combination of the two adjacent curvatures, and is therefore bounded by them by construction — checked at 4 offsets across all 8 joins (32 cases), holds unconditionally. This is the method used.

**Relabeled:** `perception/dataset/relabel_curvature.py` — all 4000 rows of `data/dataset_v0/labels.csv`. Base rows relabeled directly from their stored `x`/`y`; mirrored rows derived by negating the *new* source row's value, matching `generate_dataset.mirror_row`'s existing sign convention — not recomputed independently, since mirrored rows share `x`/`y`/`heading` exactly with their source (`mirror_row`'s own docstring), so an independent recomputation would silently drop the sign flip. Original point-wise labels preserved at `data/dataset_v0/labels_v0_pointwise.csv` before any overwrite. Verified: mirror sign relationship exact on all 2000 mirrored rows (`|new + source| < 1e-9`); all 11 non-curvature fields byte-identical on a 200-row spot check; kappa distribution went from 5 discrete values (ADR-11 finding 4) to 217 unique values as a side effect, still bounded within `[-1/R_1, 1/R_1]`.

**Retrained:** kappa's loss weight restored to `1.0` — the original, pre-ADR-14 value, confirmed via git history (`loss.py`'s first version, commit `d2d81b5`, had no per-component weighting at all: an unweighted mean over `e_y`/`e_psi`/`kappa`, i.e. implicit weight `1.0` each) rather than guessed. Otherwise identical config to the ADR-14 baseline (60 epochs, `training_config.yaml` `config_hash=4ba7402eaef9`).

**Measured** (`perception/model/analyze_adr18_retrain.py`, full dataset — not test alone; ADR-9's per-primitive stratification puts every test sample within ~0.6 m of a transition by construction, so test alone cannot represent "away from any join" — `n=3600` valid samples, `1508` near-join / `2092` away-from-join, `dist_to_next_transition <= L_usable=2.356 m`):

| Method | Near-join MAE (1/m) | Away-from-join MAE (1/m) | Ratio |
|---|---|---|---|
| Published today (hardcoded `0.0`) | `0.1317` | `0.1379` | `0.96` |
| ADR-14 checkpoint's own raw kappa (untrained) | `0.7228` | `0.7120` | `1.02` |
| **ADR-18 retrain (kappa weight `1.0`)** | **`0.0247`** | **`0.0137`** | **`1.80`** |

`81.3%` improvement over the published-today (zero) baseline at the near-join zone; near/away ratio `1.80`, under the `~2x` usability bar this investigation set going in. Both criteria met clearly, not marginally. (The ADR-14 checkpoint's own raw kappa is shown only to make explicit why it isn't a fair "before" — an untrained head is worse than publishing zero, which is exactly why ADR-14 hardcoded zero in the first place, not a baseline to beat.)

**Corroborated by training dynamics, independent of the final-MAE comparison:** kappa's validation loss converges smoothly across all 60 epochs (`0.1097` at epoch 1 to `0.0049-0.0051` by epochs 50-60, plateaued) — no flatline, unlike the original point-wise-label training's overfitting signature that motivated ADR-14 (`perception/model/loss_curves_ADR14_pointwise_kappa0.png`, frozen copy; the live `perception/model/loss_curves.png` now reflects this retrain instead).

**No regression on the other three outputs** (test split, in-distribution/interpolation only — ADR-11 finding 5, same caveat as every other test-split number in this project): `e_y` MAE `0.0219` → `0.0216` m, `e_psi` MAE `3.41` → `3.32` deg, confidence accuracy `0.9948` → `1.0000`, confidence invalid recall `0.9444` → `1.0000`. All four moved in the improving direction — plausibly training-run noise rather than a real effect of the added loss term, not claimed as significant, only as "not worse."

**Decision:** kappa is usable via windowed-relabeled curvature. `training_config.yaml`'s default is restored to `kappa: 1.0`. **This supersedes ADR-15's premise that curvature is unavailable.** ADR-15 is not edited (append-only) — its Frenet-vs-body-frame evolvability framing and its no-feedforward cost calculation (`e_y_ss = -L kappa / K1`) should be read as historical context for a premise that no longer holds, not current guidance: curvature feedforward is now available, at the accuracy measured above.

**Not done, deliberately, deferred:** `perception_node.py` still publishes `curvature=0.0` unconditionally — ADR-14's hardcoded output is untouched — and the MPC feedforward term and every file under `control/` are untouched. Flipping the published output and wiring a real feedforward term into the controller is a control-integration decision for a separate, later task, once this ADR has been reviewed. This ADR establishes that the retrained signal is accurate enough to justify that integration work; it does not perform the integration.

**Consequence:** the retrained checkpoint is at `perception/model/checkpoints/lane_cnn_width1.0_best.pt`; the ADR-14 baseline is preserved at `lane_cnn_width1.0_best_ADR14_kappa0_baseline.pt` for comparison. `data/dataset_v0/labels.csv` carries windowed-relabeled curvature going forward; `labels_v0_pointwise.csv` preserves the original point-wise labels. `perception/README.md`'s kappa sections were updated to point at this ADR and at the frozen pre-ADR-18 loss-curve evidence rather than the live `loss_curves.png`, which now shows a different run. `physical_metrics.py`'s `format_physical_report` and `train.py`'s `results.md` boilerplate, both previously hardcoded to assert "kappa untrained, loss weight 0" unconditionally, were made conditional on the run's actual `component_weights` — caught while writing this ADR's results section, not a pre-existing test failure.

**Artifacts:** `perception/dataset/windowed_relabel.py`, `perception/dataset/relabel_curvature.py`, `perception/model/analyze_adr18_retrain.py`, `tests/test_windowed_relabel.py`, `perception/model/loss_curves_ADR14_pointwise_kappa0.png`, `data/dataset_v0/labels_v0_pointwise.csv`, `perception/model/checkpoints/lane_cnn_width1.0_best_ADR14_kappa0_baseline.pt`.