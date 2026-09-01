#!/usr/bin/env python3
"""Controleur bidon -- jalon 1 uniquement.

Aucune perception, aucun MPC : juste de quoi prouver que la boucle
complete circule. Il sera remplace par le noeud MPC.
"""
import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles


class DummyController(Node):

    def __init__(self):
        super().__init__('dummy_controller')
        self.declare_parameter('rate_hz', 50.0)
        self.declare_parameter('accel', 0.35)
        self.declare_parameter('steer_amp', 0.20)
        self.declare_parameter('steer_period', 6.0)

        self.accel = self.get_parameter('accel').value
        self.amp = self.get_parameter('steer_amp').value
        self.period = self.get_parameter('steer_period').value

        self.pub = self.create_publisher(Twist, 'carsim/cmd', 10)
        self.create_subscription(Odometry, 'carsim/odom', self.on_odom,
                                 QoSPresetProfiles.SENSOR_DATA.value)
        self.have_odom = False
        self.t0 = self.get_clock().now()
        self.create_timer(1.0 / self.get_parameter('rate_hz').value, self.tick)

    def on_odom(self, msg):
        self.have_odom = True

    def tick(self):
        if not self.have_odom:
            return  # on ne commande pas un robot dont on ne recoit pas l'etat
        t = (self.get_clock().now() - self.t0).nanoseconds * 1e-9
        cmd = Twist()
        cmd.linear.x = self.accel
        cmd.angular.z = self.amp * math.sin(2.0 * math.pi * t / self.period)
        self.pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = DummyController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
