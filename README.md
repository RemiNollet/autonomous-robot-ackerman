# Autonomous Ackerman Robot

**A simulated Ackermann-steering vehicle that drives itself, using a purpose-built embedded-scale CNN for perception and an acados MPC for control, over a ROS2 architecture that treats the simulator as a hardware interface.**

[![Status](https://img.shields.io/badge/status-in%20development-yellow)]()
[![Milestone](https://img.shields.io/badge/milestone-M1-blue)]()
<!-- CI badge goes here once the pipeline exists (M5) -->

---

## Demo

<!-- REPLACE AT M6: embed demo video/gif here. Host on YouTube and link the thumbnail —
     a raw gif of a full lap will be too large for a README. -->

*Demo video coming at project completion (September 2026). The vehicle currently runs a validated closed control loop with a placeholder controller; perception and MPC are in development.*

---

## What This Is

A robotics portfolio project built between August and September 2026, documenting a career transition into robotics. The goal is not a product. It is a complete, honest engineering pipeline: synthetic dataset generation, a CNN designed against an embedded inference budget, a model predictive controller under physical constraints, and the software practices that make any of it maintainable.

The project is developed in the open, including the parts that went wrong. Gazebo was abandoned three weeks in. A latency measurement that looked alarming turned out to be clock skew. Those are documented rather than quietly fixed, because how an engineer handles a wrong result says more than a clean repository does.

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
| Perception error (lateral / heading) | — | Pending M2 |
| MPC solve time per cycle | — | Pending M3 |
| Closed-loop tracking stability | — | Pending M3 |
| FP32 vs INT8 (accuracy / latency / size) | — | Pending M4 |

Latency is reported as the two-way sum of both transmission directions. A single-direction measurement across two machines reports clock offset, not latency. That distinction cost an afternoon to find and is written up in the decision log.

## Key Decisions

**Gazebo to MuJoCo.** Gazebo's camera rendering was unusable without GPU passthrough inside the VM, and passthrough was not available on this hardware. Rather than fight the environment, the simulator moved to MuJoCo running natively on macOS, with ROS2 remaining in the VM and a ZeroMQ bridge between them. The constraint produced a better architecture than the original plan.

**Custom CNN rather than a fine-tuned detector.** The network is small by design and sized against an embedded inference budget. That choice only earns its keep if it is proven, which is what the INT8 quantization benchmark is for. A fine-tuned YOLO would have been faster to stand up and would have demonstrated nothing about deployment constraints.

**Hybrid Python and C++.** Perception in rclpy, control in rclcpp. Control runs in C++ because that is where the real-time requirement lives.

**Control validated before quantization.** The MPC is tuned and validated in closed loop against the FP32 model first. INT8 is then measured against that known-good baseline. Developing a controller against a quantized model would tangle control error with quantization error, and a working unquantized system is worth more than a quantized one that does not drive.

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
