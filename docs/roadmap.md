# Roadmap — Ackerman Robot Project

**Window:** August 1 – September 15, 2026 (~6.5 weeks)
**Companion to:** SRS v1.1 (requirements and acceptance criteria), Asana board (live task tracking)

## Purpose of This Document

This roadmap works at the milestone level: what each phase accomplishes, why it comes in this order, what can run in parallel, and what could derail it. Task-by-task detail, estimates, and day-to-day status live in Asana, which is the single source of truth for execution. This document is the narrative layer — it explains the plan Asana tracks, and it's meant to be readable on its own, without board access.

---

## Milestone 0 — Environment and Closed Loop (Complete)

Gazebo was abandoned after camera rendering proved unworkable without GPU passthrough in the VM. The project moved to MuJoCo running natively on macOS, with ROS2 in an Ubuntu VM and a ZeroMQ bridge between them.

What's validated: an Ackermann vehicle model that drives and steers correctly, a bridge that closes the loop end-to-end (odometry ~50 Hz, camera 25 Hz deterministic), and a honestly characterized latency of roughly 5 ms per direction — measured by summing both transmission directions to cancel out clock skew between the two machines.

What carries forward as known debt: jitter on the odometry topic (4–40 ms) is the first suspect if the MPC becomes unstable later, and any future latency measurement needs the same two-way summing approach.

---

## Milestone 1 — Lane State Contract and Dataset

**Goal:** define the `/lane_state` interface between perception and control, and produce a labeled synthetic dataset from MuJoCo.

This is the highest-leverage early decision in the project. The perception/control contract is the tightest coupling point in the system — get it wrong and either the CNN gets retrained or the MPC gets rewritten later, at a much higher cost than fixing it now. The dataset conditions everything downstream in perception, so it's worth getting right before writing a single line of model code.

Two things matter more here than anywhere else in the project: verifying the labels visually before trusting them, and keeping the split clean (no leakage between passes over the same track segment). Both are cheap to skip and expensive to have skipped — a mislabeled dataset stays invisible until the closed-loop test in M3, by which point the debugging cost is much higher.

---

## Milestone 2 — Perception CNN

**Goal:** a compact, custom-designed CNN that consumes camera images and outputs lane state, trained, evaluated, and running in a ROS2 node publishing `/lane_state`.

The architecture is deliberately small and justified by an embedded deployment target — that framing is what makes "custom CNN" a defensible engineering choice rather than a reinvention of something better solved by a pretrained model. Evaluation happens in physical units (meters, degrees), not abstract loss, because a validation loss curve doesn't tell you whether the vehicle can actually drive.

MPS training on Apple Silicon is a soft risk here — mostly reliable, occasionally an unsupported op away from a frustrating afternoon. A CPU fallback path is worth having ready rather than discovering the need for one mid-training run.

---

## Milestone 3 — MPC Control and Closed Loop

**Goal:** an MPC controller built on acados, consuming `/lane_state`, producing steering and acceleration commands, validated in full closed loop using the FP32 model.

This is the project's critical path. It's the largest single time investment, the least familiar toolchain (acados has real learning curve on top of an already-solid MPC theory background), and the milestone most likely to overrun. Two disciplines keep it contained: get a minimal acados example running before attempting the real formulation, and validate the solver standalone in C++ before wiring it into ROS2. Skipping either step tends to produce debugging sessions where it's unclear whether the bug is in the model, the solver, or the integration.

Closed-loop validation happens here, against the FP32 perception model — deliberately before quantization. A functional unquantized system is worth more than a quantized system that doesn't drive, and validating control against a known-good baseline means any problems found during quantization (M4) can be attributed to quantization, not tangled up with control tuning.

---

## Milestone 4 — Quantization and Optimization

**Goal:** the FP32 → ONNX → INT8 pipeline, benchmarked against the M3 baseline — including closed-loop tracking performance, not just offline accuracy.

This milestone only depends on M2 (the trained CNN), not on M3, so it can shift earlier if M3 runs long and a change of pace is useful, or run in parallel once the CNN is stable. It's also what turns "I trained a CNN" into "I designed a model for embedded deployment and proved it" — the benchmark table comparing FP32 and INT8 on accuracy, latency, and size is one of the strongest single artifacts in the portfolio.

The honest framing matters here: if INT8 degrades tracking meaningfully, that's a valid and reportable result, not a failure to hide. FP32 remains the reference system regardless of what quantization shows.

---

## Milestone 5 — Docker and CI/CD

**Goal:** a working Dockerfile for the ROS2 stack and a GitHub Actions pipeline that builds, tests, and passes reliably.

This comes after the code is stable, not before — dockerizing a moving target means maintaining a Dockerfile that breaks every other day. The main technical risk is ARM: building on Apple Silicon has its own set of base-image and dependency quirks worth reading up on before diving in, rather than discovering them one failed build at a time.

---

## Milestone 6 — Portfolio and Documentation

**Goal:** a GitHub repository and README that a recruiter can read and understand in about five minutes, with a visual demo, quantified results, and explicit architectural decisions.

The portfolio is the actual deliverable — the code is the evidence supporting it. Per-component documentation written incrementally at M1–M5 makes this milestone assembly work rather than a writing sprint from a blank page. What doesn't compress well: producing a working demo video, and a genuine final read-through with a recruiter's eye rather than the author's.

---

## Ordering and Parallelization

Three sequencing decisions aren't negotiable: M1 before M2 (no CNN without a dataset), M2 before M3 (the MPC needs `/lane_state`, though its theoretical formulation can start earlier), and M3 before M4 (closed-loop validation against a known-good baseline, ahead of the confound quantization would introduce).

Two things can shift for schedule flexibility. M4 depends only on M2, not M3 — it's a legitimate parallel track if M3 stalls on acados and a change of pace is useful. And the acados competency ramp should start during M2, not at the beginning of M3 — it's the longest single learning curve in the project and the one most worth de-risking early rather than discovering cold.

## Indicative Calendar

- **Week 1 (Aug 1–7):** M1. Begin acados theory in parallel.
- **Week 2 (Aug 8–14):** M2 — architecture and training.
- **Week 3 (Aug 15–21):** M2 close-out (ROS2 node) + M3 start (formulation, minimal acados example).
- **Week 4 (Aug 22–28):** M3 — solver and MPC node. M4 in parallel if blocked.
- **Week 5 (Aug 29–Sep 4):** M3 close-out (closed loop + tuning) + M4.
- **Week 6 (Sep 5–11):** M5.
- **Half-week 7 (Sep 12–15):** M6 + buffer.

The final half-week is the buffer. If acados overruns — the likely scenario — it gets absorbed there. If the schedule holds, that time goes to the hardware bonus or to polishing the portfolio.

## An Honest Note on the Plan

The critical risk isn't any single milestone — it's M3 eating into everyone else's time. If something has to give, the hardware bonus goes first, then CI/CD polish (a working Dockerfile without a fully tuned pipeline still demonstrates the competency). What should never get cut: dataset inspection at M1, closed-loop validation at M3, and the final portfolio pass at M6. Each protects against a failure mode that costs far more time to fix later than it costs to prevent now.

For granular task lists, hour estimates, requirement traceability, and live status, see the Asana board.
