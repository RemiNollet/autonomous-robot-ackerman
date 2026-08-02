# Ackerman Robot — Autonomous Lane Following

*One-sentence hook: a simulated Ackermann-steering vehicle that drives itself using a custom embedded-scale CNN and an acados MPC controller, over a real ROS2/MuJoCo hardware-interface architecture.*

<!-- Demo video/gif goes here — highest-value artifact, capture at M6 -->

## At a Glance

| Metric | Value |
|--------|-------|
| Perception → command latency | ~5 ms (two-way sum, clock-skew corrected) |
| Perception error | *(fill in at M2 — meters / degrees)* |
| MPC solve time | *(fill in at M3)* |
| FP32 vs INT8 | *(fill in at M4 — accuracy / latency / size)* |

## Architecture

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

The simulator is treated as a hardware interface, not as part of the control stack — see [`docs/srs.md`](docs/srs.md) §2.1 for the rationale. This is what makes a future physical-hardware port (optional, see SRS §1.4) a substitution rather than a rewrite.

## Key Decisions

- **Gazebo → MuJoCo.** Gazebo's camera rendering was unworkable without GPU passthrough in the VM; MuJoCo runs natively on macOS instead. *(full writeup: docs/decisions.md)*
- **Hybrid Python/C++.** Perception in Python (rclpy), control in C++ (rclcpp) — deliberate signal of real-time awareness.
- **Control validated before quantization.** The MPC is tuned and validated in closed loop against the FP32 model first; INT8 is then benchmarked against that known-good baseline, including closed-loop behavior.

## Project Sections

- [`perception/`](perception/) — dataset generation, CNN, training, ONNX/INT8 pipeline
- [`control/`](control/) — MPC formulation, acados solver, ROS2 control node
- [`infra/`](infra/) — bridge, Docker, CI/CD, tests
- [`docs/project-management.md`](docs/project-management.md) — SRS, roadmap, retrospective

## Status

Actively in development, August–September 2026. See [`docs/roadmap.md`](docs/roadmap.md) for milestone status.
