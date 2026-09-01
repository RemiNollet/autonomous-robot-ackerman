# Perception

A custom CNN that turns a single 320x240 camera frame into `/lane_state` --
lateral offset, heading offset, and (nominally) curvature -- the entire
interface between what the camera sees and what the controller does about
it. This page is a portfolio page: what was measured, what it means, and
what it doesn't mean. The full decision trail is in
[`docs/decisions.md`](../docs/decisions.md); this condenses it, it doesn't
replace it.

## The problem, and the contract

The vehicle needs three numbers from a camera frame: how far off centerline
it is (`lateral_error`), which way it's pointed relative to the lane
(`heading_error`), and how the lane curves ahead (`curvature`) -- plus a
confidence signal so a stale or ambiguous frame doesn't get blindly trusted.
That's the whole output space; no lane geometry, no object detection, no
occupancy grid. The full sign-convention and units contract is
[`docs/lane-state-contract.md`](../docs/lane-state-contract.md) (FR-8,
FR-18) -- REP-103 frame, `+left` on every signed quantity, curvature signed
by turn direction. It was fixed before any dataset generation, specifically
so a sign bug couldn't hide inside a tuned controller later (it nearly did
once, caught by the acceptance tests -- ADR-7).

## Architecture

`perception/model/lane_cnn.py`. Input `3x80x160` (a crop of the 320x240
render -- see below). Every conv is `BatchNorm2d + ReLU`, no bias (the BN
shift makes a conv bias redundant):

**Input geometry: `80x160`, not the crop's native `102x320`.** The crop
(rows 80:182, full width) is resized anisotropically -- width scales by
0.5, height by ~0.78 -- specifically so four stride-2 halvings land on
integers with no rounding or padding ambiguity: `80x160 -> 40x80 -> 20x40
-> 10x20 -> 5x10`. A size that didn't divide cleanly by 16 would force an
odd/even mismatch at some stage, handled either by asymmetric padding
(which shifts the effective receptive-field center) or a crop, neither of
which has a reason to exist here. The horizontal squeeze is accepted
because lane width, not its aspect ratio, is what the downstream regression
needs (see Preprocessing: `L_resolvable` depends on horizontal scale, which
this resize halves uniformly, same as every column in the image).

```
stem   conv 3->16   k5 s2 p2   -> 16 x 40x80
b1     conv 16->32  k3 s2 p1   -> 32 x 20x40
       conv 32->32  k3 s1 p1
b2     conv 32->64  k3 s2 p1   -> 64 x 10x20
       conv 64->64  k3 s1 p1
b3     conv 64->96  k3 s2 p1   -> 96 x  5x10
       conv 96->96  k3 s1 p1
red    conv 96->32  k1         -> 32 x  5x10
flatten -> fc 1600->128, ReLU, dropout 0.2
-> fc 128->4   (e_y, e_psi, kappa, confidence_logit)
```

**Downsampling is strided convs, not pooling.** Every spatial halving
(`stem`, `b1a`, `b2a`, `b3a`) is a stride-2 conv, not a pool followed by a
conv. A pooling layer throws away everything except one statistic per
window before the next layer ever sees it; a strided conv lets the network
learn *which* combination of the window's pixels matters before discarding
the rest. On an input this small (already downsampled from 320x240 to
160x80 by the crop), that headroom is worth keeping.

**No global average pooling, no depthwise-separable convs.** GAP before the
head would collapse the `5x10` spatial map to a single vector per channel --
fine for classification, wrong here: `e_y` and `heading_error` are position-
and-orientation-dependent quantities, and GAP is invariant to *where* in the
feature map the signal sits, which is exactly the information a regression
head over a spatial layout needs. Depthwise-separable convs (MobileNet-style)
trade accuracy for parameter count at a ratio that matters at hundreds of
thousands to low millions of parameters; at 418k, the full convs stay well
under the embedded budget below without needing that trade.

**A 1x1 reduction before flatten, not a flatten straight from `b3`.**
Flattening `96x5x10` directly would make `fc1` a `4800->128` layer --
`614,400` weights, larger than the entire rest of the network combined. The
`1x1` conv projects `96` channels down to `32` first (`cred` in the code,
scales with `width_mult`), keeping the encoder's channel width where the
convs need it while keeping the FC layer's input small.

**`forward` returns `(output, b3_features)`.** The raw `96`-channel feature
map before the `1x1` reduction is exposed so a different head -- a
per-column head, for instance -- can be attached later without retraining
the encoder. It isn't used for anything on v0: the labels are a single
Frenet-frame evaluation at the vehicle's position, not per-column ground
truth (see Limitations), so there's nothing for a per-column head to be
supervised against on this dataset.

`width_mult` scales every conv channel count (not the FC layers), for the
capacity ablation below.

## Preprocessing

`perception/model/preprocess.py`, imported directly by both training
(`perception/model/dataset.py`) and inference
(`carsim_bridge/perception_inference.py`) -- never reimplemented, so a
preprocessing mismatch between the two paths, which is silent and very hard
to diagnose, isn't a risk that exists here (NFR-8, and see the perception_node
section below).

Crop rows **80:182** of the 320x240 render (`perception/dataset/cnn_input_config.json`),
resized to `160x80`. Neither bound is arbitrary, both were verified rather
than assumed:

- **Row 80 (top)** lands at ground distance **2.36 m** -- `L_usable`, the
  point where the camera resolution report says the road stops being
  reliably informative (see Limitations). Row 80.23, to be precise.
- **Row 182 (bottom)** is the boundary of a rendering artifact: MuJoCo's
  near-clipping plane, not the vehicle's own geometry or an
  insufficient ground plane (both hypotheses were checked directly and
  ruled out -- full diagnosis in `docs/decisions.md` ADR-12). Below this
  row, every rendered frame shows skybox, not ground.

So the crop, top to bottom, **is** "the region that is both real ground and
within `L_usable`" by construction, not a coincidence discovered afterward.

Per-image standardization (`(x - mean) / (std + 1e-6)`, computed per image),
not the dataset-wide constants in `normalization_stats.json`. MuJoCo's
directional lighting makes the same scene read very differently depending
on which way the vehicle is facing -- measured directly in
`tests/test_camera_visibility.py`, where one rendered marker read anywhere
from RGB `(105, 0, 105)` to `(230, 5, 230)` depending only on heading. A
fixed dataset-wide mean/std would leave that lighting variation sitting in
the input; per-image standardization removes it.

Augmentation (train split only, pixel-space, `training_config.yaml`):
brightness/contrast/gamma jitter, Gaussian noise, random erasing. No
horizontal flip -- the dataset already contains mirror pairs (ADR-10), and
a mirror shares its source's split, so flipping again at train time would
pair a sample with its own twin.

## Embedded budget

| | This model |
|---|---|
| Parameters | 417,940 |
| MACs | 33,229,312 (33.2 M) |
| FLOPs (2x MACs) | ~66.5 M |
| Pi 5 ceiling (given) | ~100 MFLOP |

Both counts come from a graph-walking counter (`count_params`/`count_macs`
in `lane_cnn.py`, hooked to every `Conv2d`/`Linear` at runtime), not a
hand-derived number -- it tracks the architecture (and `width_mult`)
automatically if either changes, and was cross-checked against a hand
computation from the layer spec before being trusted (agreement to 0.2%).

66.5 MFLOP against a ~100 MFLOP ceiling leaves headroom, deliberately. INT8
quantization (M4) typically buys 2-4x throughput, not accuracy -- spending
the entire FP32 budget now would mean the quantization pass either has no
room left to spend on the eventual degraded-domain retrain (varied
lighting, worn markings, real camera noise) or has to shrink the *architecture*
in M4 instead of just the *numeric precision*, which defeats the point of
having a separate quantization milestone. The margin is intentional, not
unspent optimization.

## Results

**All numbers below are on v0 -- read the Limitations section before
trusting any of them as "can this drive."** Full breakdown:
[`perception/model/results.md`](model/results.md), generated by
`perception/model/train.py` (config hash `d5832461132b`, `training_config.yaml`).

### Main model, test split (in-distribution -- see Limitations)

| Output | Units | MAE | RMSE | p95 | max |
|---|---|---|---|---|---|
| e_y | m | 0.0219 | 0.0286 | 0.0581 | 0.1058 |
| e_psi | deg | 3.41 | 4.14 | 7.78 | 10.69 |

e_y MAE is 5.5% of the lane half-width (0.4 m), p95 is 14.5%. e_psi MAE is
11.9% of the sampling envelope (+/-28.6 deg), p95 is 27.2%. Confidence:
99.5% accuracy, but that's on an imbalanced 90/10 valid/invalid split --
per-class, valid recall is 100.0%, invalid recall 94.4%.

**kappa is not reported as a result anywhere on this page.** Its loss
weight is 0 -- see Limitations.

### Per-curvature-bin (main model, test split)

| Bin | e_y MAE (m) | e_psi MAE (deg) | n |
|---|---|---|---|
| straight | 0.0191 | 2.92 | 138 |
| R=5m arc | 0.0232 | 3.54 | 126 |
| R=3m arc | 0.0247 | 4.02 | 84 |

Error rises with tighter curvature -- about 30% worse on R=3m than on
straight, for both outputs. Not dramatic in-distribution. It is dramatic
under the one measurement that isn't in-distribution -- next.

### Mirror-generalization probe (the one number that isn't pure interpolation)

Trained on source (non-mirrored) renders only, evaluated on their never-seen
mirror twins -- right turns from a track that only physically contains left
turns (ADR-10). Geometrically distinct enough from training to measure
something closer to generalization than the table above does.

| Output | Units | MAE | RMSE | p95 | max |
|---|---|---|---|---|---|
| e_y | m | 0.0440 | 0.0583 | 0.1144 | 0.4211 |
| e_psi | deg | 6.49 | 8.36 | 16.25 | 21.50 |

Roughly 2x the in-distribution error on both outputs -- a real,
measured generalization gap, not assumed to be zero.

| Bin | e_y MAE (m) | e_psi MAE (deg) | n |
|---|---|---|---|
| straight | 0.0266 | 2.21 | 601 |
| R=5m arc | 0.0499 | 8.88 | 416 |
| R=3m arc | 0.0751 | 12.66 | 256 |

Here the curvature effect is not subtle: e_y MAE nearly 3x worse on R=3m
than straight, e_psi MAE almost 6x worse. Under the one measurement that
approximates generalization, tight-curve performance degrades far more than
the aggregate number suggests -- this constrains M3 more than the
in-distribution table does, and should be read as the more honest estimate
of what a genuinely new track segment would look like.

### Constant predictor and width ablation

Constant predictor (train-set mean, ignores the image): e_y MAE 0.183
(normalized), e_psi MAE 0.233 (normalized), confidence accuracy 0.906 --
the CNN clearly beats it on e_y/e_psi (normalized MAE 0.022 / 0.060). This
is the check that matters most for kappa specifically: on the pre-fix
checkpoint, the CNN did **not** clearly beat this baseline on kappa, which
is direct evidence toward the transition-mislabeling finding below, not
just consistent with it.

Width ablation (`width_mult=0.25`) was run once, earlier in the M2 work,
under a materially different setup than everything else on this page:
40 epochs (not 60), the pre-crop-fix dataset (no rows-80:182 crop, no
crop-visibility-consistency check), and a nonzero kappa loss weight. Its
numbers are reproduced below **for reference only** -- they are not
comparable to the main-model table above, which was regenerated this
session under the current dataset, crop, and `component_weights`:

| Run | Params | MACs | val loss | MAE e_y (norm.) | MAE e_psi (norm.) | conf acc | train time |
|---|---|---|---|---|---|---|---|
| width=0.25 (**stale, pre-crop-fix**) | 65,512 | 2,835,712 | 0.0550 | 0.0350 | 0.0720 | 1.000 | 276.4s |

Not re-run this session -- the finding it was collected for (capacity isn't
saturated at full width) doesn't depend on the crop/kappa changes, so
re-running it wasn't prioritized over the physical-unit evaluation above.
The qualitative result held under that older setup: e_y/e_psi degrade
noticeably at 0.25x width (MAE roughly double the width=1.0 run under the
*same* old setup) while confidence does not, evidence that v0 does not
saturate the model's capacity on the outputs that actually work. Treat the
specific numbers as historical, not as a current capacity measurement.

## Limitations, plainly

**No partition of v0 measures generalization.** ADR-11 finding 5: every
R=3m arc is geometrically identical to the other, same for R=5m, under
identical lighting and texture. A random split measures interpolation
between near neighbors in arc-length; the per-primitive stratified split
used here (ADR-9) correctly prevents leakage but still draws val/test from
the same physical geometry as train, because there is no other geometry on
this track to hold out. The mirror-generalization probe above is the
exception, and its ~2x-worse (up to ~6x on tight curves) numbers are the
more honest estimate of what a new track would look like.

**Confidence is structurally easy, not a nuanced signal.** Negative
(off-track) samples are near-uniformly dark regardless of exactly where the
vehicle is (ADR-8: mean luminance 119.1 +/- 1.0 over 100 sampled negatives,
vs 122.7 +/- 5.6 for positives) -- off-track ground in this scene is unlit
and untextured. The 99%+ confidence accuracy reported above reflects "is
this frame structurally empty," which is genuinely useful, but it is not
evidence the model has learned anything about ambiguous or partially-occluded
perception. Don't read it as more sophisticated than it is.

**kappa is untrained, and here is the actual mechanism, measured, not
inferred.** REFERENCE_TRACK has 8 curvature transitions around its loop;
`L_usable` (2.36 m -- see Preprocessing) means the camera's reliably
informative range ends before many of those transitions are behind the
vehicle. A sample labeled "straight, kappa=0" one meter before an R=3m arc
shows a visible curve in frame -- the point-wise Frenet label describes the
vehicle's exact position, not what the camera can see ahead of it.
Measured directly (`perception/model/kappa_transition_proximity.png`,
`analyze_kappa_transitions.py`): on straight samples, mean `|kappa_pred|`
correlates at **r=-0.67** with distance to the next transition, plateauing
almost exactly at `L_usable` -- **0.023** beyond it vs **0.115** within it,
a 5x gap. Per-component loss curves confirm the mechanism further: kappa's
*training* loss converges as well as e_y's and e_psi's (the network can fit
it on seen examples), while its *validation* loss flatlines from ~epoch 5
onward (`loss_curves.png`) -- the network was reading the road correctly
and memorizing training-specific mappings that don't transfer, not failing
to learn at all. This affects roughly 42% of the loop's arc-length
(`T=8` transitions x `L_usable` / `total_length`). kappa's loss weight is
set to 0 as a result (`training_config.yaml`) -- not merely small: a kappa
output that predicts curvature on straights and near-zero in curves would
actively steer an MPC feedforward term off a straight line approaching a
bend, which is worse than publishing nothing. `perception_node` always
publishes `curvature=0.0` (ADR-12), never the network's raw output.

**`L_usable = 2.36 m` caps the M3 preview horizon.** Three independent
limits were measured (`docs/camera-resolvability.md`): `L_resolvable`
(13.25 m at native resolution, 6.69 m after the CNN crop -- lane width
falls below 10 px), `L_separable` (2.44 m -- two points 0.5 m apart stop
being resolvable by more than 2 image rows), `L_representable` (2.36 m --
where the quadratic preview path stops fitting a circular arc of the
track's tightest radius). The binding constraint is depth resolution and
path-representation validity, not pixel width -- lane width stays resolvable
well past where the other two limits bind. Any MPC preview horizon beyond
2.36 m evaluates a reference the perception layer never reliably observed.

**Classical CV is not being out-competed here, on purpose.** ADR-11's
decision, stated before this measurement, not after: on clean v0, a
classical lane-detection pipeline (Hough transform / color threshold on a
track with painted white markings against dark, untextured ground) is
expected to be competitive with or better than this CNN, and this page does
not claim otherwise. The CNN is dimensioned against the eventual
degraded-domain case -- varied lighting, worn markings, real camera noise --
where a hand-tuned classical pipeline is far more brittle. On v0 the model
is deliberately over-capacity for exactly that reason (see Embedded
budget); the results above should be read as "does the pipeline work
end-to-end and is the architecture sized sensibly," not as "this CNN beats
simpler methods on this specific track."

## `perception_node`

`carsim_bridge/perception_node.py`, running in the VM alongside the rest of
the ROS2 graph. Subscribes to `carsim/image_raw`, publishes
`carsim_msgs/LaneState` on `lane_state`. The model/preprocessing logic
lives in a separate, rclpy-independent module
(`carsim_bridge/perception_inference.py`) specifically so it's unit
testable on a machine with no ROS2 install (this project's ROS2 stack is
VM-only, ADR-1) -- `perception_node.py` itself is a thin wrapper around it.

- **Preprocessing is the training path, not a second copy of it** --
  `perception_inference.py` imports `perception/model/preprocess.py`
  directly. Tested byte-for-byte: a real dataset image, decoded through the
  node's ROS-message path and then `preprocess()`, produces an array
  `numpy.array_equal` to loading the same image directly through the
  training dataset's path (`tests/test_perception_inference.py`).
- **`header.stamp` is propagated, not re-stamped** -- copied from the
  incoming image message straight onto the outgoing `LaneState`, closing
  the M1 timestamp item (`docs/lane-state-contract.md` section 3: the age
  computation downstream is only exact if every hop forwards the original
  render time instead of re-stamping with its own receipt time).
- **`curvature` is always published as exactly `0.0`**, with a comment
  pointing at ADR-12 -- never the network's raw kappa output, which is
  untrained initialization drift that could vary unpredictably between
  checkpoints.
- **`confidence`** is the model's sigmoid output; `valid` is
  `confidence >= confidence_threshold` (parameter, default 0.5).

**Deployment note:** the trained checkpoint
(`perception/model/checkpoints/lane_cnn_width1.0_best.pt`) is produced on
the Mac (training happens there, alongside MuJoCo) and is git-ignored --
getting it onto the VM for the node to load is a manual step (scp / shared
volume) not yet automated, since M2 hasn't needed it until now.

### Latency

Measured on the **Mac host's CPU** (`torch.device('cpu')`, not MPS), over
200 real dataset images, timing preprocessing/forward/publish separately.
This is compute in isolation -- a lower bound, not the rate achievable
inside the actual ROS2 graph (executor overhead, subscription queueing, the
ZeroMQ hop none of that touches).

**Not the VM.** The VM was booted and network-reachable this session
(`192.168.64.4` answered ICMP) but had no open SSH or other execution
channel available, so nothing could actually be run inside it.
`perception_node.py` carries the instrumentation for when it can be:
`on_image` now accumulates raw preprocess/forward/publish samples, the
wall-clock interval between consecutive `/lane_state` publications
(actual achieved rate, not derived from the stage timings), and the
end-to-end age from `header.stamp` to publish. Every `stats_window_frames`
frames (500 by default, so the report is never based on fewer than the
task's own floor) it logs the full mean/p50/p95/max/std distribution for
each series and writes the same table to `stats_output_path` (default
`/tmp/perception_node_vm_stats.md`) -- see
[`perception/model/results.md`](model/results.md)'s "VM inference
frequency" section for the exact collection procedure and, notably, a
correction to this project's own assumption about what `header.stamp`
measures: it's stamped by the VM's own clock at ZeroMQ-receipt time, not
converted from the Mac's render time, so `header.stamp -> publish` is a
single-clock, graph-internal number -- it does not by itself span the
Mac->VM hop, which is measured separately and already clock-skew-corrected
(`/carsim/latency_ms`, ADR-4).

| Stage | mean | p50 | p95 | max |
|---|---|---|---|---|
| preprocess | 0.26 ms | 0.26 ms | 0.31 ms | 0.37 ms |
| forward | 1.11 ms | 1.06 ms | 1.46 ms | 1.69 ms |
| publish (message build) | ~0.00 ms | ~0.00 ms | ~0.00 ms | ~0.00 ms |
| **total** | **1.37 ms** | **1.32 ms** | **1.73 ms** | **1.99 ms** |

Implied max rate ~728 Hz, far above the ~30 Hz camera rate -- on this
hardware, latency is not the bottleneck.

**A prediction that did not hold: forward pass dominates preprocessing here
(1.11 ms vs 0.26 ms), not the reverse.** The going-in assumption --
"preprocessing dominates on small models" -- is real, but it's a fact about
*depthwise-separable* architectures (MobileNet-style): when every conv is
decomposed into a near-free depthwise pass plus a small pointwise one, the
fixed crop/resize/standardize cost on the input pixels stops being small by
comparison. This network was deliberately built the other way -- full
`3x3`/`5x5` convs throughout, no depthwise separation (Architecture, above)
-- specifically because 418k params has room to spend on full convs without
threatening the embedded budget. That same decision is why the assumption
doesn't transfer: full convs cost enough per layer, even at `33.2 M` total
MACs, that PyTorch's per-op eager-mode dispatch overhead across ~8 conv
layers outweighs the crop/resize/standardize cost on a `320x240` source
image. Reported as measured, not as the expected result; may shift on the
VM's CPU (different architecture, virtualized) or after ONNX export removes
the eager-mode dispatch overhead (M4).

## What v1 changes, and why

- **Continuous curvature (clothoid transitions, varied radii, both turn
  directions built into the track geometry itself)** -- addresses the
  kappa-label problem directly (no more 8 discrete transitions each
  invalidating a point-wise label within 2.36 m) and the discrete-5-value
  target problem (ADR-11 finding 4) at the same time. Would let kappa's
  loss weight be restored to nonzero for the first time since ADR-12.
- **Domain variation (lighting, texture, worn markings)** -- addresses the
  confidence head being structurally easy (this Limitations section) and
  is the actual justification for this CNN's parameter budget over a
  classical pipeline (Embedded budget, and ADR-11's decision).
- Both are scoped in ADR-11, out of scope for M2.

## Artifacts

- [`perception/model/kappa_transition_proximity.png`](model/kappa_transition_proximity.png) --
  the r=-0.67 evidence
- [`perception/model/loss_curves.png`](model/loss_curves.png) -- per-component
  train/val loss, the overfitting signature on kappa
- [`perception/model/kappa_scatter.png`](model/kappa_scatter.png) -- predicted
  vs true kappa, pre-fix checkpoint
- [`docs/camera-resolvability.md`](../docs/camera-resolvability.md) /
  [`.png`](../docs/camera-resolvability.png) -- L_resolvable/L_separable/L_representable
- [`docs/dataset/camera_coverage_contact_sheet.png`](../docs/dataset/camera_coverage_contact_sheet.png) --
  visibility filter coverage, 27 poses
- [`perception/model/results.md`](model/results.md) -- full numeric results
- [`docs/decisions.md`](../docs/decisions.md) -- ADR-7 through ADR-12
