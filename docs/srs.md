# Software Requirements Specification — Ackerman Robot Project

**Author:** Rémi
**Version:** 1.1
**Date:** August 1, 2026
**Status:** Approved for Development

*Changes from v1.0: control milestone now precedes model optimization (closed-loop validation in FP32 before quantization); hardware deployment reclassified from out-of-scope to optional bonus objective; bridge serialization rationale reframed as a design choice rather than an environment constraint.*

---

## 1. Context and Objectives

### 1.1 Context

This project is developed as part of a career transition into robotics. It serves as a technical portfolio for applying to a first robotics engineer position, targeting the South Korean market. The project's goal is not to produce a commercially deployable system but to demonstrate, across a complete and realistic pipeline, mastery of the core competencies expected in such a role: neural network-based perception, advanced control, ROS2 integration, and software engineering practices.

### 1.2 System Objective

Develop a miniature autonomous vehicle with Ackermann steering, simulated, capable of autonomously following a lane. The system integrates visual perception via CNN, a model predictive controller (MPC), and a distributed software architecture representative of industrial practices.

### 1.3 Educational and Demonstration Objectives

The project must demonstrate:

- Design of a complete perception pipeline, from dataset to optimized deployable model.
- Formulation and implementation of an MPC controller under constraints on a non-holonomic system.
- A realistic ROS2 architecture where intelligence is decoupled from the simulated "plant."
- Industrialization practices: containerization, continuous integration, testing, documentation.
- End-to-end technical project execution, from planning to delivery.

### 1.4 Optional Objectives (Bonus)

**Physical hardware deployment.** Porting the perception and control stack to a Raspberry Pi, with the physical vehicle replacing the simulated plant, is an optional extension. The architecture is designed to make this substitution straightforward: the simulator is treated as a hardware interface throughout, so perception and control nodes require no modification.

This extension is not scheduled within the primary timeline and conditions no primary deliverable. It is documented as a designed-for capability and pursued if time permits or after the initial delivery. The INT8 quantized model (FR-14) exists specifically to support this path.

### 1.5 Out of Scope

- Global navigation, long-range trajectory planning, and dynamic obstacle avoidance.
- Production-grade robustness. The system demonstrates competence across a coherent pipeline, not deployment readiness.

---

## 2. General Description

### 2.1 Overall Architecture

The system relies on a strict separation between the simulator, treated as a "plant" (hardware), and the ROS2 control stack. The physics simulator (MuJoCo) runs natively on the development workstation (macOS, Apple Silicon); it exposes sensor state and accepts actuator commands. All intelligence—perception, estimation, control—resides in a ROS2 graph hosted in an Ubuntu virtual machine. The two worlds communicate via a network bridge.

This separation mirrors the real architecture of robots, where the ROS2 stack never runs physics in its nodes but instead interfaces with a hardware abstraction layer. It also makes porting to a physical robot conceptually straightforward: the simulated "plant" is replaced by the real "plant," with no modifications to perception and control nodes.

### 2.2 Functional Chain

```
[macOS]  MuJoCo (physics + camera)
              ↕  network bridge (ZeroMQ)
[VM]     bridge_node (rclpy)
              ↓ /carsim/image_raw, /carsim/odom
         perception_node (Python, CNN)
              ↓ /lane_state
         mpc_node (C++, acados)
              ↓ /cmd (steering, acceleration)
         bridge_node → MuJoCo
```

### 2.3 Development Environment

- Development workstation: MacBook Air M1, 32 GB RAM.
- Simulator: MuJoCo native on macOS.
- ROS2 stack: Ubuntu in a UTM VM.
- Training: PyTorch native Apple Silicon (MPS).
- The software environment is frozen (MuJoCo 3.11.0, numpy 2.2.6) for the project duration. Dependency changes require explicit revalidation of physics and offscreen rendering.

---

## 3. Functional Requirements

Each requirement is identified (FR-n) and associated with a verification criterion.

### 3.1 Simulation and Vehicle Model

**FR-1.** The system must simulate an Ackermann steering vehicle (four-wheel bicycle model extension) equipped with a forward-facing onboard camera.
*Verification:* the vehicle drives and steers with correct differential angles between inner and outer wheels in turns.

**FR-2.** The simulator must expose sensor state: camera image, position, velocity, heading.
*Verification:* data is published and readable on the ROS2 side via dedicated topics.

**FR-3.** The simulator must accept actuator commands: steering angle and acceleration (or torque).
*Verification:* a command sent modifies vehicle behavior coherently.

**FR-4.** The scene must contain a track with lane markings visible to the camera, covering straight segments and curves of varying radii.
*Verification:* markings are detectable in captured images across varied position and heading conditions.

### 3.2 Communication Bridge

**FR-5.** A bridge must transport simulator state to ROS2 and ROS2 commands back to the simulator, with distinct frequencies for image and odometric state.
*Verification:* state at ~50 Hz, image at ~25 Hz, commands at ~50 Hz, measured on topics.

**FR-6.** The bridge must prioritize data freshness over completeness, processing only the most recent frame per cycle.
*Verification:* stale frame rejection behavior is observable and documented.

**FR-7.** The bridge must use language-neutral serialization (structured text for scalar state, raw buffer with header for images), independent of any runtime-specific object format.
*Verification:* the wire protocol is documented and parseable without reference to the producing language or runtime.

*Rationale: the bridge represents a hardware interface. A physical robot will not share the control stack's runtime, so the protocol must not assume one. This requirement is what makes the hardware substitution in §1.4 a swap rather than a rewrite.*

### 3.3 Perception

**FR-8.** The system must provide a CNN that ingests camera images and outputs lane state: lateral error (signed distance from lane center) and heading error.
*Verification:* the model produces outputs conforming to the contract on test images.

**FR-9.** The CNN must be a custom, compact architecture dimensioned for embedded deployment, not a reused pre-trained generic model.
*Verification:* parameter count and structure are documented and justified by hardware constraints.

**FR-10.** Perception must be encapsulated in a ROS2 node that subscribes to images and publishes lane state on `/lane_state`.
*Verification:* the node runs in the graph and publishes at a measured frequency.

### 3.4 Control

**FR-11.** The system must implement an MPC controller based on the vehicle kinematic model, consuming lane state and producing steering and acceleration commands.
*Verification:* the controller produces sensible commands on fixed lane states before loop closure.

**FR-12.** The MPC must respect physical constraints: maximum steering angle, maximum acceleration.
*Verification:* produced commands remain within defined bounds.

**FR-13.** Control must be encapsulated in a C++ ROS2 node (rclcpp) using acados for solving.
*Verification:* the node runs at the target control frequency and publishes on `/cmd`.

### 3.5 Model Optimization

Optimization is performed after closed-loop validation (FR-17), against a functional FP32 baseline.

**FR-14.** The model must be exported to ONNX format with numerical equivalence verification against the source PyTorch model.
*Verification:* PyTorch and ONNX Runtime outputs coincide within tolerance.

**FR-15.** The system must produce an INT8-quantized version of the model via post-training quantization with calibration.
*Verification:* the INT8 model is functional and produces usable outputs.

**FR-16.** A benchmark must compare FP32 and INT8 versions on accuracy, inference latency, and model size. The comparison must include closed-loop behavior, not offline accuracy alone.
*Verification:* a quantified comparison is produced, including tracking performance of the vehicle under each variant.

### 3.6 System Behavior

**FR-17.** In closed-loop operation, the vehicle must follow the lane stably, both on straight segments and curves, using the FP32 model.
*Verification:* the vehicle traverses the track without leaving the lane over a significant duration.

**FR-18.** The system must define fallback behavior when perception is invalid or stale.
*Verification:* MPC behavior under missing or stale lane state is defined and tested.

---

## 4. Non-Functional Requirements

**NFR-1. Real-Time Performance.** The control loop must maintain its target frequency (~50 Hz) and MPC solve time per cycle must be measured and compatible with this frequency.

**NFR-2. Latency.** Communication latency from perception to command must be measured honestly (accounting for clock skew between machines) and documented.

**NFR-3. Reproducibility.** The environment must be frozen and the end-to-end pipeline (dataset generation, training, export, quantization) reproducible.

**NFR-4. Portability.** The ROS2 stack must be containerized (Docker), accounting for the ARM architecture of the development workstation.

**NFR-5. Continuous Integration.** A CI/CD pipeline (GitHub Actions) must build the workspace, run tests, build the Docker image, and pass reliably.

**NFR-6. Testability.** The project must include targeted high-signal tests: bridge protocol, CNN preprocessing, and at least one lightweight integration test.

**NFR-7. Documentation.** Each component must be documented, and the project must present a main README readable by a recruiter in minutes, with architecture, quantified results, and justified decisions.

**NFR-8. Train/Inference Consistency.** Image preprocessing must be identical between training and inference, factored into shared code.

---

## 5. Technical Constraints

- **ROS2** is mandatory: it is a requirement of the target market and the architectural foundation.
- **Simulator / control stack separation:** MuJoCo outside the container (treated as hardware), ROS2 stack containerized.
- **Python / C++ hybrid:** perception in Python (rclpy), control in C++ (rclcpp), a deliberate choice to reflect industry real-time practices.
- **MPC via acados:** no solver implementation from scratch; the contribution is formulation and integration.
- **GitHub** for version control and CI/CD (not GitLab).

---

## 6. Deliverables

- Structured and documented GitHub repository.
- MJCF vehicle model and track scene.
- ZeroMQ bridge and ROS2 bridge node.
- Labeled synthetic dataset and its generation pipeline.
- Perception CNN: training code, trained model, ROS2 node.
- MPC controller: formulation, acados solver, ROS2 node.
- Optimization pipeline: ONNX export, INT8 model, FP32/INT8 benchmark.
- Docker containerization and GitHub Actions CI/CD pipeline.
- Test suite.
- Documentation: main README, per-component docs, visual demo (video/gif), results figures.
- Project management documents: this SRS, roadmap, and tracking.

---

## 7. Acceptance Criteria by Milestone

The project is divided into milestones. A milestone is accepted when its exit criterion is verified.

| Milestone | Objective | Acceptance Criterion |
|-----------|-----------|----------------------|
| M0 | Environment + closed loop | Stable end-to-end closed loop, latency characterized. **(complete)** |
| M1 | `/lane_state` contract + dataset | Dataset inspected and labels verified, split without leakage, contract frozen and documented. |
| M2 | Perception CNN + ROS2 node | `perception_node` publishes `/lane_state` at measured frequency, validation error documented in physical units. |
| M3 | MPC acados + complete loop | Autonomous vehicle stable on track in FP32, perception + control in closed loop, MPC solve time measured. |
| M4 | Optimization (ONNX + INT8) | Functional INT8 model, FP32/INT8 benchmark quantified including closed-loop behavior, reproducible pipeline. |
| M5 | Docker + CI/CD | Functional Dockerfile, CI pipeline green, relevant tests. |
| M6 | Portfolio + documentation | Repository readable by recruiter, visual demo, explicit results and decisions. |

*Ordering rationale: closed-loop validation (M3) precedes quantization (M4) so that optimization is measured against a known-good FP32 baseline. Developing the controller against a quantized model would conflate control errors with quantization errors, and a functional unquantized system is worth more than a quantized system that does not drive.*

---

## 8. Risks and Mitigations

| Risk | Milestone | Impact | Mitigation |
|------|-----------|--------|------------|
| Mislabeled or imbalanced dataset | M1 | CNN learns wrong signal, invisible until M3 | Mandatory visual inspection of labels before training |
| CNN insufficiently accurate for control | M2 | Feedback loop to dataset or architecture | Evaluate in physical units, not abstract loss |
| PyTorch MPS surprises (Apple Silicon) | M2 | Time lost on unsupported ops | CPU fallback for debugging, test early |
| acados learning curve | M3 | Main risk of project timeline overrun | Run minimal example first; standalone MPC before ROS2 integration; start competency ramp at M2 |
| Closed-loop instability | M3 | Behavior non-compliant with FR-17 | Instrument each component, isolate root cause (perception / MPC / jitter) before tuning |
| INT8 precision degradation too severe | M4 | Quantized model degrades tracking | FP32 remains the reference system; INT8 is measured against it and reported honestly, including negative results |
| Docker multi-arch ARM friction | M5 | Build broken on Apple Silicon | Document ARM base images, test build early on simple component |
| Portfolio rushed due to time | M6 | Technical work doesn't sell itself | Write docs incrementally, reserve dedicated time for assembly |

---

## 9. References

- Detailed project roadmap (milestones, sub-tasks, time estimates).
- Project tracking (Asana).
