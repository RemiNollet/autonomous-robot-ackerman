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
