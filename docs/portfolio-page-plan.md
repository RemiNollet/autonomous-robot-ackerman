# Portfolio Page Plan

Landing page is written. This is the plan for the remaining pages: what each contains, where the content comes from, and when it can be filled in.

Pages sit as `README.md` inside each component folder so GitHub renders them when the folder is opened. No GitHub Pages site needed — a recruiter browsing the repo hits each page naturally. (If you later want a hosted site, the same markdown feeds GitHub Pages with minimal change.)

---

## 1. `perception/README.md`

**Fillable now:** contract design, architecture rationale, planned pipeline
**Pending:** all results, training curves, benchmark numbers

| Section | Source | Milestone |
|---------|--------|-----------|
| The problem | What lane state means, why lateral + heading error is the right output | Now |
| `/lane_state` contract | SRS FR-8, FR-18 — representation, units, frame, invalid handling | M1 |
| Dataset generation | How poses are sampled, how ground truth is computed from MuJoCo state, distribution rationale | M1 |
| Dataset validation | Label overlay inspection, distribution histograms, leakage check | M1 |
| CNN architecture | Layer structure, parameter count, and the embedded budget that drove the sizing | M2 |
| Training | Loss, augmentation, hyperparameters, training curves | M2 |
| Results | Test error in meters and degrees, failure case analysis, inference frequency | M2 |
| ONNX export | Export process, numerical equivalence verification | M4 |
| INT8 quantization | PTQ approach, calibration set, why INT8 on embedded targets | M4 |
| Benchmark | FP32 vs INT8 table: accuracy, latency, size, and closed-loop tracking | M4 |

The benchmark table is the strongest single artifact in this section. It is what turns "trained a CNN" into "designed a model for deployment and proved the tradeoff."

---

## 2. `control/README.md`

**Fillable now:** why MPC over PID, vehicle model
**Pending:** solver details, tuning, closed-loop results

| Section | Source | Milestone |
|---------|--------|-----------|
| Why MPC | Non-holonomic constraint, why a PID is insufficient here, what MPC buys | Now |
| Ackermann geometry | Steering geometry, inner/outer wheel angles, the kinematic consequence | Now |
| Kinematic bicycle model | State, inputs, equations the MPC predicts over | M3 |
| OCP formulation | Cost function terms, constraints, horizon and discretization choice | M3 |
| acados implementation | Solver generation, standalone C++ validation before ROS2 integration | M3 |
| ROS2 control node | Node structure, control frequency, fallback on stale perception (FR-18) | M3 |
| Closed-loop results | Tracking behavior on straights and curves, solve time per cycle, tuning process | M3 |

Worth including here: the tuning process itself, not just final weights. What oscillated, what was tried, how the cause was isolated between perception, MPC, and jitter. That reads as engineering rather than a lucky configuration.

---

## 3. `infra/README.md`

**Fillable now:** bridge design and protocol, latency methodology — this section is largely complete already
**Pending:** Docker, CI, tests

| Section | Source | Milestone |
|---------|--------|-----------|
| Bridge architecture | Why ZeroMQ, PUB/SUB pattern, simulator-as-plant rationale (SRS §2.1, FR-7) | Now |
| Wire protocol | Message structure, language-neutral serialization, why not pickle | Now |
| Freshness over completeness | Queue draining, the slow-joiner problem, why dropped frames are intentional | Now |
| Latency methodology | Clock skew between host and VM, why two-way sum, the 30 ms that was not latency | Now |
| Docker | ROS2 stack containerization, ARM specifics on Apple Silicon, why MuJoCo stays native | M5 |
| CI/CD | Pipeline structure, what it checks | M5 |
| Tests | What is tested and the reasoning for targeted coverage over exhaustive | M5 |

The latency writeup is a good story and mostly already told in your logs. A measurement that looked like a problem, three converging clues that it was not, a corrected methodology, and a real bug found alongside it (25 Hz instead of 30, from tick quantization).

---

## 4. `docs/project-management.md`

**Fillable now:** everything except the retrospective

| Section | Source | Milestone |
|---------|--------|-----------|
| Approach | Why a solo project was specified and tracked formally | Now |
| Requirements | Link to SRS, explanation of the FR/NFR traceability scheme | Now |
| Planning | Link to roadmap, milestone ordering rationale | Now |
| Tracking | Link to Asana board, custom field scheme (Milestone / Estimate / Requirement) | Now |
| Estimate vs actual | Where estimates held, where they did not, by milestone | M6 |
| Retrospective | What went wrong, what was replanned, what would be done differently | M6 |

Keep this page factual and short. The retrospective is the part that carries weight; the rest is context for it.

---

## 5. `docs/decisions.md`

An architecture decision log, appended to as decisions happen rather than written at the end. Each entry: context, options considered, decision, consequence.

**Already decidable now:**

- Gazebo to MuJoCo (context: VM rendering; consequence: bridge architecture, better sim-to-real story)
- ZeroMQ over raw TCP or running ROS2 on macOS
- Language-neutral serialization over pickle
- Queue draining over ZeroMQ CONFLATE (CONFLATE does not support multipart)
- Control before quantization in milestone ordering
- Custom CNN over fine-tuned detector

**To append as they arise:** lane state representation, CNN architecture sizing, MPC horizon and cost structure, Docker scope boundary.

This is the page that ages best. Decisions written down at the time they were made are visibly different from decisions reconstructed afterward, and the difference is legible to anyone who has done both.

---

## Writing Order

Three of these can be substantially written now, before more code exists:

1. `infra/README.md` — the bridge work is done and the latency story is complete
2. `docs/decisions.md` — six decisions are already made and fresh
3. `docs/project-management.md` — everything but the retrospective

Then `perception/` and `control/` fill in as their milestones complete. Writing them at the time rather than at M6 is the difference between assembly and a writing sprint from nothing.
