# Autonomous Ackerman Robot

**A simulated Ackermann-steering vehicle that drives itself, using a purpose-built embedded-scale CNN for perception and an acados MPC for control, over a ROS2 architecture that treats the simulator as a hardware interface.**

[![Status](https://img.shields.io/badge/status-in%20development-yellow)]()
[![Milestone](https://img.shields.io/badge/milestone-M4-blue)]()
<!-- CI badge goes here once the pipeline exists (M5) -->

---

## Demo

[▶ Watch the demo](docs/Robot_demo1.mp4) — full closed-loop lap, real perception, real acados MPC, chase camera.

*This is the validated M3 system: a trained CNN estimating lane geometry from the front camera, feeding a curvature-aware MPC through the full ROS2 chain (bridge → perception → control → bridge). No quantization applied yet — this is the FP32 baseline the M4 benchmark will be measured against.*

---

## What This Is

A robotics portfolio project built between August and September 2026, documenting a career transition into robotics. The goal is not a product. It is a complete, honest engineering pipeline: synthetic dataset generation, a CNN designed against an embedded inference budget, a model predictive controller under physical constraints, and the software practices that make any of it maintainable.

The project is developed in the open, including the parts that went wrong. Gazebo was abandoned three weeks in. A latency measurement that looked alarming turned out to be clock skew. An MPC formulation choice made in M2 — because curvature wasn't available yet — was reversed in M3 once it was. Those are documented rather than quietly fixed, because how an engineer handles a wrong result says more than a clean repository does.

## Architecture

```
[macOS]  MuJoCo (physics + camera)
              ↕  ZeroMQ bridge
[Ubuntu VM]  bridge_node (rclpy)
                  ↓ /carsim/image_raw, /carsim/odom
             perception_node (Python, CNN)
                  ↓ /lane_state
             mpc_node (C++, acados)
                  ↓ /cmd  (steering, acceleration)
             bridge_node → MuJoCo
```


The simulator sits outside the control stack and is treated as hardware. It exposes sensor state and accepts actuator commands over a language-neutral wire protocol, nothing more. Everything above that boundary is ROS2.

This is not an accident of the toolchain. It is how robots are actually built, and it means the optional hardware phase is a substitution of the plant rather than a rewrite of the stack. Perception and control nodes would not change.

## Results

| Metric | Value | Status |
|--------|-------|--------|
| Perception → command latency | ~5 ms per direction | Measured (M0) |
| Control loop frequency | 50 Hz | Measured (M0) |
| Camera pipeline frequency | 25 Hz, deterministic | Measured (M0) |
| Perception curvature error (near-join / away-from-join) | 0.0247 / 0.0137 (1/m), 81.3% improvement over the untrained baseline | Measured (M2) |
| Live perception accuracy degradation near curvature transitions | ~1.7–1.8× worse than steady curvature, matching the offline measurement | Measured (M3, full-chain) |
| MPC solve time per cycle (real ROS2 node, full callback) | mean 0.9–1.1 ms, p95 1.3–1.6 ms | Measured (M3) — well under the 20 ms / 50 Hz budget |
| Closed-loop sustained stability | 24 laps, ~18 min, 0 solve failures across 30k+ solves, no drift | Measured (M3) |
| Safety fallback (stale/invalid perception) | 26/27 boundary trials matched expected behavior exactly | Measured (M3) |
| FP32 vs INT8 (accuracy / latency / size) | — | Pending (M4, in progress) |

Latency is reported as the two-way sum of both transmission directions. A single-direction measurement across two machines reports clock offset, not latency. That distinction cost an afternoon to find and is written up in the decision log.

## Key Decisions

**Gazebo to MuJoCo.** Gazebo's camera rendering was unusable without GPU passthrough inside the VM, and passthrough was not available on this hardware. Rather than fight the environment, the simulator moved to MuJoCo running natively on macOS, with ROS2 remaining in the VM and a ZeroMQ bridge between them. The constraint produced a better architecture than the original plan.

**Custom CNN rather than a fine-tuned detector.** The network is small by design and sized against an embedded inference budget. That choice only earns its keep if it is proven, which is what the INT8 quantization benchmark (M4, in progress) is for. A fine-tuned YOLO would have been faster to stand up and would have demonstrated nothing about deployment constraints.

**Hybrid Python and C++.** Perception in rclpy, control in rclcpp. Control runs in C++ because that is where the real-time requirement lives.

**A formulation choice reversed when its premise changed.** The MPC was first built as pure error regulation, because the perception model's curvature output was untrained and hardcoded to zero — carrying a curvature scalar into the controller would have meant feeding it noise. Once a windowed-relabeling fix made curvature usable (M2), the controller was rebuilt as body-frame path tracking, which lets the solver preview the path's actual shape across its horizon instead of reacting to a single instantaneous error. The reversal, and the measurement that justified it, are both in the decision log rather than silently overwritten.

**Control validated before quantization.** The MPC was tuned and validated in closed loop against the FP32 model first — including a full-chain integration run, an 18-minute sustained-stability run, and a forced-perception-dropout test exercising the safety fallback at its exact threshold boundaries. INT8 is now being measured against that known-good baseline (M4). Developing a controller against a quantized model would have tangled control error with quantization error, and a working unquantized system was worth more than a quantized one that didn't drive yet.

Full write-ups in [`docs/decisions.md`](docs/decisions.md).

## Explore

| Section | Contents |
|---------|----------|
| [Perception](perception/) | Dataset generation and validation, CNN architecture and training, ONNX export, INT8 quantization and benchmark |
| [Control](control/) | Kinematic model, MPC formulation, acados solver, ROS2 control node, closed-loop tuning |
| [Infrastructure](infra/) | ZeroMQ bridge and wire protocol, Docker, CI/CD, test suite |
| [Project Management](docs/project-management.md) | Requirements spec, roadmap, live task board, retrospective |

## Specification and Planning

This project was specified before it was built. The requirements document defines seventeen functional and eight non-functional requirements, each with a verification criterion, and every task in the tracker traces back to one.

- [Software Requirements Specification](docs/srs.md)
- [Roadmap](docs/roadmap.md)
- [Live task board (Asana)](https://app.asana.com/1/1217051674470772/project/1217061963308639/list/1217061965185436)

## Stack

ROS2 · MuJoCo · PyTorch · ONNX Runtime · acados · ZeroMQ · Docker · GitHub Actions · Python · C++

## Contact

Rémi Nollet — [LinkedIn](https://www.linkedin.com/in/remi-nollet/)

<!-- Add email here if you want to be reachable without LinkedIn -->

---

*In active development, August–September 2026. Milestone status in the [roadmap](docs/roadmap.md).*