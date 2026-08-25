# LaneState.msg
#
# Perception -> control contract. See docs/lane-state-contract.md for the
# full design rationale, sign conventions, and acceptance tests.
#
# Frame: base_link, REP-103 (x forward, y left, z up, angles CCW positive).
# header.stamp: simulation render time of the source camera frame,
#   propagated end-to-end from the bridge. NOT publish time, NOT wall clock.
#   This is what makes message age computable without clock-skew error.

std_msgs/Header header

float32 lateral_error     # m,   signed offset of centerline from vehicle, +left
float32 heading_error     # rad, signed angle, tangent relative to vehicle heading, +left
float32 curvature         # 1/m, signed curvature at the projection point, +left turn

float32 confidence        # [0,1], perception confidence
bool    valid              # confidence >= threshold, decided producer-side
