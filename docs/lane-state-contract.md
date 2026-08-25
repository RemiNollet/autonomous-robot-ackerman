# `/lane_state` Contract — Design

**Milestone:** M1
**Requirements:** FR-8, FR-18
**Status:** Validated. Implemented in [`perception/dataset/geometry.py`](../perception/dataset/geometry.py), tested in [`tests/test_lane_state_geometry.py`](../tests/test_lane_state_geometry.py) (9/9 passing).

This is the interface between perception and control. It is the tightest coupling point in the system: changing it later means regenerating the dataset, retraining the CNN, and reworking the MPC. Everything below is decided once, before any dataset generation starts.

---

## 1. What the message carries

Three scalars describing the vehicle's pose relative to the lane centerline, plus metadata.

```
# LaneState.msg

std_msgs/Header header

float32 lateral_error     # m,    signed offset of lane centerline in vehicle frame
float32 heading_error     # rad,  signed angle from vehicle heading to lane tangent
float32 curvature         # 1/m,  signed curvature of centerline at the projection point

float32 confidence        # [0,1] perception confidence
bool    valid             # producer-side threshold on confidence
```

### Why these three scalars

`lateral_error` and `heading_error` are the standard cross-track and heading error pair used in path tracking. They are what an MPC cost function penalizes directly.

`curvature` is the one that deserves justification. Without it, the MPC predicts over an implicitly straight lane and lags in curves — a steady-state lateral offset that grows with speed and tightness. With it, the controller has feedforward information and can anticipate. Since the track geometry is known in simulation, labeling curvature is nearly free at dataset generation time.

### Equivalence to a path fit

These three scalars are the same information as a quadratic fit of the lane centerline in the vehicle frame:

```
y(x) = c₀ + c₁·x + c₂·x²

c₀ = lateral_error
c₁ = tan(heading_error)
c₂ = curvature / 2
```

The MPC reconstructs the preview path from the three scalars rather than consuming coefficients directly. Same content, but the scalar form validates in interpretable units — an error of 4 cm and 2° means something immediately, whereas an error on `c₂` does not. Given that FR-8's verification criterion is stated in physical units, the scalar parameterization is the better fit.

*Alternative considered:* cubic polynomial coefficients (comma.ai / production lane-keeping style). Richer preview, but adds a fourth output that is hard to validate intuitively, and a noisy cubic term feeds the MPC a bad path. Not worth the fragility on a track this size.

---

## 2. Reference frame and sign conventions

Frame: `base_link`, following REP-103 — **x forward, y left, z up, right-handed.** Angles counter-clockwise positive.

Sign definitions, stated once so they are never ambiguous:

| Field | Definition | Positive means |
|-------|-----------|----------------|
| `lateral_error` | Lateral offset of the lane centerline from the vehicle origin, measured along the vehicle y-axis | Centerline is **to the left** of the vehicle → steer left to correct |
| `heading_error` | Angle from vehicle heading to the lane tangent, wrapped to (−π, π] | Lane tangent points **left** of current heading → steer left to correct |
| `curvature` | Signed curvature of the centerline at the projection point | Lane curves **left** |

Note the convention choice on `lateral_error`: it is the offset of the *centerline relative to the vehicle*, not of the vehicle relative to the centerline. The two differ by a sign. This form was chosen because positive error then calls for positive (left) steering, which keeps the MPC coupling sign intuitive and avoids a class of sign bug that is genuinely hard to spot in a tuned controller.

**Acceptance test for the convention** (implemented in `tests/test_lane_state_geometry.py`):

1. Vehicle to the right of the centerline, heading aligned with the lane → `lateral_error > 0`, `heading_error ≈ 0`.
2. Vehicle heading rotated clockwise relative to the lane tangent (so the tangent now points to the vehicle's left) → `heading_error > 0`, calling for a left correction.

*Revision note: the original wording of case 2 here ("rotate it to point left of the tangent") was ambiguous about which of the vehicle or the tangent was rotating, and read backwards during implementation — the sign definition in the table above is the authoritative one and was unaffected. Restated unambiguously once the test suite caught the ambiguity.*

---

## 3. Timestamp semantics

`header.stamp` is **the simulation time at which the camera frame was rendered**, propagated end to end. Not the publish time, and not VM wall-clock time.

Two reasons, both learned at M0.

The MPC needs the *age* of the perception it is acting on, and that age includes camera-to-inference latency, not just transport. Stamping at publish time hides the inference cost.

More importantly, the image originates on the macOS host and `/lane_state` is published inside the VM. Any timestamp comparison across that boundary using local wall clocks measures clock skew, not elapsed time — exactly the 30 ms artifact found at M0. Propagating the simulator's own clock gives a single time base and makes the age computation exact.

The bridge already carries a publish timestamp. This requires that it also carry the render time as a distinct field, and that `perception_node` copies it into the header rather than restamping. Small change, worth making explicit now.

`header.frame_id` is `base_link`.

---

## 4. Invalid and degraded perception (FR-18)

Two distinct failure modes, handled separately because they fail differently.

**Perceptual failure** — the lane is not visible, occluded, or the vehicle is off track. Signalled by `confidence`, produced by an auxiliary head on the CNN. `valid` is set producer-side as `confidence >= threshold`, with the threshold a node parameter so the decision lives in one place rather than being re-derived by each consumer.

**Pipeline failure** — the perception node crashed, lagged, or the bridge dropped out. Detected consumer-side by comparing `header.stamp` against current simulation time. No message content is involved; a perfectly valid message that is 300 ms old is still unusable.

The MPC's fallback logic keys on both: act only if `valid` and the message age is under a defined bound. Fallback behavior itself is specified at M3, but the contract must expose enough for that decision to be made, which is what these fields do.

**Training implication:** a confidence head needs negative examples. Roughly 10% of the dataset should be poses where the lane is not properly visible — vehicle well off track, or heading away from the lane. This is cheap to generate but must be planned at generation time, not retrofitted. If the timeline gets tight, the fallback position is to drop `confidence` to a constant and rely on staleness alone, documented as a limitation rather than left silently broken.

---

## 5. Consequences for other M1 tasks

Two constraints flow backwards from this design into tasks not yet started.

**Track geometry must be parametric.** Curvature and the centerline projection both need ground truth that is exact and cheap. If the track is built as arcs and straight segments (or a spline), curvature at the projection point is a closed-form lookup. If it is built as a texture on a mesh, both become an estimation problem, and the labels inherit that error. Build the track as parametric geometry with a rendered representation derived from it.

**The dataset generator needs a projection routine.** Given a vehicle pose, find the nearest point on the centerline and evaluate position, tangent, and curvature there. This is the core of the labeling logic and is where sign errors hide. Worth unit-testing against hand-computed cases on a simple arc before generating anything at scale.

---

## 6. What is not in the contract

**Normalization.** The CNN may internally regress normalized targets, but the message is always in SI units. Normalization statistics are an implementation detail of `perception_node`, shared with training per NFR-8.

**Lookahead point.** Some formulations report error at a fixed distance ahead rather than at the vehicle. That is a pure-pursuit convention; an MPC with a proper kinematic model and curvature preview does not need it.

**Lane width, lane count, absolute position.** Out of scope per SRS §1.5. Single-lane tracking only.

---

## Confirmed decisions

1. **Curvature included from the start.** One extra output dimension, near-zero labeling effort, avoids a full retrain if the MPC later needs feedforward.

2. **Quadratic path only — no cubic term.** The MPC preview reconstructs `y(x) = c₀ + c₁·x + c₂·x²` from the three physical scalars (§1). A cubic (`dκ/ds`) term was considered and dropped: on this track's scale and preview horizon it adds a noisy, low-value term to the path the MPC sees. The message never carried more than `lateral_error`, `heading_error`, `curvature` — this was the design from the first draft, not a change.

3. **Confidence head kept, cheap version.** No separate head or dedicated loss — a single sigmoid output added to the existing regression head, trained with BCE against ~10% deliberately off-track poses in the dataset. This is what makes FR-18 (fallback on invalid perception) real rather than only covering pipeline dropout via staleness. If the schedule gets tight, the documented fallback is to drop this to a constant and rely on staleness alone — not to leave it silently unused.

4. **Sign convention on `lateral_error` and `heading_error`, fixed as in §2.** `lateral_error` is centerline-relative-to-vehicle (positive = centerline to the vehicle's left). `heading_error = wrap(tangent_heading − vehicle_heading)`, so `c₁ = tan(heading_error)` is positive, consistent with the polynomial form in §1. Verified against hand-computed cases in `tests/test_lane_state_geometry.py`, including a genuine sign bug caught in the arc-projection code for clockwise turns — see the fix commit for the affected function.

5. **Parametric track geometry** (arcs and line segments), per §5. Confirmed before the track-building task starts — curvature ground truth is exact under this construction, not estimated.
