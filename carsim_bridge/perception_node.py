#!/usr/bin/env python3
"""CNN camera -> /lane_state.

Cote VM. Le SEUL noeud qui connait le modele de perception : pour tout le
reste du graphe, /lane_state est un contrat (docs/lane-state-contract.md),
pas une implementation -- le jour ou l'INT8/ONNX ou une pipeline CV
classique (ADR-9) remplace ce modele, on remplace ce noeud, pas les autres.

    /carsim/image_raw  sensor_msgs/Image      (in,  ~30 Hz, rgb8)
    /lane_state        carsim_msgs/LaneState  (out)

Thin rclpy wrapper: all the model/preprocessing logic lives in
perception_inference.py, which has no rclpy dependency and is what's unit
tested directly. This file only exists to plug that logic into the ROS2
graph -- see tests/test_perception_node.py for why that split matters on a
project where ROS2 only runs in the VM (ADR-1).
"""
import os
import sys
import time

import rclpy
import torch
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from carsim_msgs.msg import LaneState  # noqa: E402
from carsim_bridge.perception_inference import (  # noqa: E402
    DEFAULT_CHECKPOINT, load_model, ros_image_to_pil, run_inference,
)


class PerceptionNode(Node):

    def __init__(self, **kwargs):
        # **kwargs forwarded to rclpy.node.Node -- lets tests construct this
        # node with parameter_overrides (e.g. a temp checkpoint_path) without
        # a live graph or a real trained checkpoint on disk.
        super().__init__('perception_node', **kwargs)

        self.declare_parameter('checkpoint_path', DEFAULT_CHECKPOINT)
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('device', 'cpu')

        checkpoint_path = self.get_parameter('checkpoint_path').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.device = torch.device(self.get_parameter('device').value)
        self.model = load_model(checkpoint_path, self.device)

        self.pub = self.create_publisher(LaneState, 'lane_state', 10)
        self.create_subscription(Image, 'carsim/image_raw', self.on_image,
                                  QoSPresetProfiles.SENSOR_DATA.value)

        self.n_msgs = 0
        self.t_preprocess_sum = 0.0
        self.t_forward_sum = 0.0
        self.t_publish_sum = 0.0
        self.create_timer(2.0, self.report)

        self.get_logger().info(
            f'perception_node actif  checkpoint={checkpoint_path}  device={self.device}')

    def on_image(self, msg: Image):
        pil_img = ros_image_to_pil(msg.width, msg.height, msg.encoding, msg.data)
        e_y, e_psi, confidence, t_pre, t_fwd = run_inference(self.model, pil_img, self.device)

        t2 = time.perf_counter()
        out = LaneState()
        # Propagated render time, NOT re-stamped here -- the M1 contract's
        # timestamp semantics (docs/lane-state-contract.md section 3)
        # require the age computation downstream to be exact, which only
        # holds if every hop copies this forward instead of restamping.
        out.header.stamp = msg.header.stamp
        out.header.frame_id = msg.header.frame_id
        out.lateral_error = float(e_y)
        out.heading_error = float(e_psi)
        # ADR-12: kappa's loss weight is 0 -- the head is kept in the
        # architecture but was never trained, so its raw output is
        # initialization drift that could vary unpredictably between
        # checkpoints. Publishing it would look like a real curvature signal
        # to a downstream MPC feedforward term; publish the one value that
        # cannot be mistaken for one.
        out.curvature = 0.0
        out.confidence = float(confidence)
        out.valid = confidence >= self.confidence_threshold

        self.pub.publish(out)
        t3 = time.perf_counter()

        self.n_msgs += 1
        self.t_preprocess_sum += t_pre
        self.t_forward_sum += t_fwd
        self.t_publish_sum += (t3 - t2)

    def report(self):
        if self.n_msgs == 0:
            return
        n = self.n_msgs
        total = self.t_preprocess_sum + self.t_forward_sum + self.t_publish_sum
        self.get_logger().info(
            f'{n} msgs | preprocess {self.t_preprocess_sum/n*1000:.2f} ms | '
            f'forward {self.t_forward_sum/n*1000:.2f} ms | '
            f'publish {self.t_publish_sum/n*1000:.2f} ms | '
            f'total {total/n*1000:.2f} ms')
        self.n_msgs = 0
        self.t_preprocess_sum = self.t_forward_sum = self.t_publish_sum = 0.0


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
