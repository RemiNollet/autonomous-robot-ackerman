#!/usr/bin/env python3
"""Sim-clock <-> wall-clock age calibration, shared by perception_node.py
and mpc_node.py.

ADR-13 made bridge_node.py stamp every /carsim/image_raw and /carsim/odom
message's header.stamp with sim-clock-relative time (RclpyTime(seconds=
t_sim).to_msg(), via bridge_node.py's sim_time_to_stamp) -- NOT wall-clock
epoch time. No node in this ROS2 graph sets use_sim_time, so
self.get_clock().now() everywhere else still returns real wall-clock epoch
time. Subtracting the two directly (now_ns - stamp_ns) therefore mixes two
incompatible clock domains and produces a huge, meaningless "age" instead
of a small millisecond-scale one -- exactly what perception_node.py's own
age telemetry did, silently, from ADR-13 until this fix (flagged in
ADR-23's Consequence).

AgeCalibrator fixes this without needing the actual clock offset (which
this project has no way to measure directly): it tracks the running
MINIMUM of (wall_now_ns - stamp_ns) across every message seen, a proxy for
the constant sim<->wall offset -- if the sim runs in realtime, the only
thing that should vary message-to-message is real pipeline latency, never
negative. Age for any message is then the excess over that running
minimum: offset-agnostic, and converges after the first few messages
(message #1 always reads age=0 by construction -- an acceptable cold-start
artifact, not a correctness bug for the messages after it).

First written for mpc_node.py's ADR-24 delay compensation; extracted here
(ADR-25) so perception_node.py's own age telemetry uses the same fix
instead of a second, divergent implementation.
"""
from rclpy.time import Time as RclpyTime


class AgeCalibrator:
    """Per-stream running-minimum offset calibrator. Create one instance
    per header.stamp source that mixes sim-clock stamps with wall-clock
    now() (e.g. one per subscribed topic) -- the running minimum is a
    calibration of THAT stream's own offset and isn't meaningful shared
    across streams with independent offsets."""

    def __init__(self):
        self._min_offset_ns = None

    def age_s(self, now_ns: int, stamp) -> float:
        """now_ns: self.get_clock().now().nanoseconds, read by the caller
        (not here) so each node keeps using its own clock. stamp: a
        builtin_interfaces/Time (e.g. msg.header.stamp)."""
        stamp_ns = RclpyTime.from_msg(stamp).nanoseconds
        offset_ns = now_ns - stamp_ns
        if self._min_offset_ns is None or offset_ns < self._min_offset_ns:
            self._min_offset_ns = offset_ns
        return max(0.0, (offset_ns - self._min_offset_ns) * 1e-9)
