#!/usr/bin/env python3
"""Pont ZeroMQ <-> ROS2.

Cote VM. C'est le SEUL noeud qui connait la simulation : pour tout le
reste du graphe, il n'y a qu'un robot qui publie une camera et une
odometrie, et qui accepte une commande. Le jour ou l'on branche le
Raspberry Pi, on remplace ce noeud, pas les autres.

    /carsim/image_raw   sensor_msgs/Image      (~30 Hz)
    /carsim/odom        nav_msgs/Odometry      (~50 Hz)
    /carsim/latency_ms  std_msgs/Float32       latence sim -> ros
    /carsim/cmd         geometry_msgs/Twist    linear.x = accel, angular.z = braquage
"""
import math

import rclpy
import zmq
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import Image
from std_msgs.msg import Float32

from carsim_bridge import protocol as P


class BridgeNode(Node):

    def __init__(self):
        super().__init__('carsim_bridge')

        self.declare_parameter('sim_host', '192.168.64.1')
        self.declare_parameter('state_port', 5555)
        self.declare_parameter('cmd_port', 5556)
        self.declare_parameter('poll_hz', 200.0)
        self.declare_parameter('frame_id', 'base_link')

        host = self.get_parameter('sim_host').value
        state_port = self.get_parameter('state_port').value
        cmd_port = self.get_parameter('cmd_port').value
        poll_hz = self.get_parameter('poll_hz').value
        self.frame_id = self.get_parameter('frame_id').value

        self.ctx = zmq.Context()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.setsockopt(zmq.SUBSCRIBE, b'')
        # 50, not 2: with poll() now processing every drained frame (see
        # drain_state below), this is a real safety margin against a burst
        # bigger than the largest observed (2, measured empirically under
        # UTM's virtualised networking -- see drain_state's docstring), not
        # a number that silently discards anything under normal operation.
        self.sub.setsockopt(zmq.RCVHWM, 50)
        self.sub.connect(f'tcp://{host}:{state_port}')

        self.pub_cmd = self.ctx.socket(zmq.PUB)
        self.pub_cmd.setsockopt(zmq.SNDHWM, 2)
        self.pub_cmd.connect(f'tcp://{host}:{cmd_port}')

        sensor_qos = QoSPresetProfiles.SENSOR_DATA.value
        self.pub_img = self.create_publisher(Image, 'carsim/image_raw', sensor_qos)
        self.pub_odom = self.create_publisher(Odometry, 'carsim/odom', sensor_qos)
        self.pub_lat = self.create_publisher(Float32, 'carsim/latency_ms', 10)
        self.create_subscription(Twist, 'carsim/cmd', self.on_cmd, 10)

        self.cmd_seq = 0
        self.last_seq = -1
        self.n_state = 0
        self.n_img = 0
        self.n_skipped = 0
        self.lat_sum = 0.0

        self.create_timer(1.0 / poll_hz, self.poll)
        self.create_timer(2.0, self.report)
        self.get_logger().info(
            f'pont actif  etat<-tcp://{host}:{state_port}  cmd->tcp://{host}:{cmd_port}')

    # ------------------------------------------------------------------ #

    def drain_state(self):
        """Vide la file ZMQ et renvoie TOUTES les trames en attente, dans
        l'ordre d'arrivee.

        Gardait auparavant uniquement la derniere trame (pour ne jamais
        traiter un etat perime en boucle de controle temps reel), mais
        mesure sur la VM : le reseau virtualise d'UTM livre par moments 2
        trames dans la meme fenetre de poll meme avec poll() a 200 Hz,
        largement plus rapide que la publication a 50 Hz (rafale max
        observee : 2, jamais plus, sur 1600 ticks a 200 Hz / 8 s). Ne garder
        que la derniere de chaque rafale jetait ~40% des etats et ~70% des
        images -- une image sur deux tics seulement, donc statistiquement
        plus souvent la trame la plus ancienne d'une rafale, la moins
        susceptible de survivre. Chaque trame drainee ici reste fraiche (elle
        vient d'arriver dans la meme fenetre de quelques ms) : la traiter
        n'introduit pas la latence que la version precedente cherchait a
        eviter.
        """
        frames_list = []
        while True:
            try:
                frames_list.append(self.sub.recv_multipart(zmq.NOBLOCK))
            except zmq.Again:
                return frames_list

    def poll(self):
        for frames in self.drain_state():
            self._process_frame(frames)

    def _process_frame(self, frames):
        header, img_bytes = P.decode_state(frames)
        stamp = self.get_clock().now().to_msg()

        lat_ms = (P.now() - header['t_pub']) * 1e3
        self.lat_sum += lat_ms
        self.n_state += 1
        if self.last_seq >= 0:
            self.n_skipped += max(0, header['seq'] - self.last_seq - 1)
        self.last_seq = header['seq']
        self.pub_lat.publish(Float32(data=float(lat_ms)))

        self.pub_odom.publish(self.make_odom(header, stamp))

        if img_bytes is not None:
            self.pub_img.publish(self.make_image(header['img'], img_bytes, stamp))
            self.n_img += 1

    def make_odom(self, header, stamp):
        p, t = header['pose'], header['twist']
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = 'odom'
        msg.child_frame_id = self.frame_id
        msg.pose.pose.position.x = p['x']
        msg.pose.pose.position.y = p['y']
        msg.pose.pose.orientation.z = math.sin(p['yaw'] / 2.0)
        msg.pose.pose.orientation.w = math.cos(p['yaw'] / 2.0)
        msg.twist.twist.linear.x = t['vx']
        msg.twist.twist.linear.y = t['vy']
        msg.twist.twist.angular.z = t['yaw_rate']
        return msg

    def make_image(self, meta, payload, stamp):
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.height = meta['h']
        msg.width = meta['w']
        msg.encoding = meta['encoding']
        msg.is_bigendian = 0
        msg.step = meta['w'] * meta['c']
        msg.data = payload
        return msg

    def on_cmd(self, msg: Twist):
        self.pub_cmd.send_multipart(
            P.encode_cmd(self.cmd_seq, msg.angular.z, msg.linear.x))
        self.cmd_seq += 1

    def report(self):
        if self.n_state == 0:
            self.get_logger().warn('aucun etat recu -- simulation lancee ? IP correcte ?')
            return
        self.get_logger().info(
            f'etat {self.n_state / 2.0:5.1f} Hz | img {self.n_img / 2.0:5.1f} Hz | '
            f'latence {self.lat_sum / self.n_state:5.2f} ms | '
            f'sautees {self.n_skipped} | cmd {self.cmd_seq}')
        self.n_state = self.n_img = self.n_skipped = 0
        self.lat_sum = 0.0

    def destroy_node(self):
        self.sub.close()
        self.pub_cmd.close()
        self.ctx.term()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
