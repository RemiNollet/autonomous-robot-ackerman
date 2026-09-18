#!/usr/bin/env python3
"""ZeroMQ <-> ROS2 bridge.

Runs on the VM. This is the ONLY node that knows about the simulator --
for the rest of the graph there is just a robot that publishes a camera
and odometry, and accepts a command. The day the Raspberry Pi is wired
in, this node gets replaced, not the others.

    /carsim/image_raw   sensor_msgs/Image      (~30 Hz)
    /carsim/odom        nav_msgs/Odometry      (~50 Hz)
    /carsim/latency_ms  std_msgs/Float32       sim -> ros latency
    /carsim/cmd         geometry_msgs/Twist    linear.x=accel, angular.z=steer
"""
import math

import rclpy
import zmq
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from rclpy.time import Time as RclpyTime
from sensor_msgs.msg import Image
from std_msgs.msg import Float32

from carsim_bridge import protocol as P


def sim_time_to_stamp(t_sim):
    """MuJoCo sim time (seconds since sim start, sim_server.py's data.time)
    -> builtin_interfaces/Time. header.stamp must be the render time, not
    the VM's receipt wall-clock (docs/decisions.md ADR-13, lane-state-
    contract.md section 3). Module-level: a pure function of t_sim."""
    return RclpyTime(seconds=t_sim).to_msg()


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
        # Margin above the largest burst observed under UTM's virtualised
        # network (2 frames -- see drain_state's docstring), not a number
        # that silently drops frames in normal operation.
        self.sub.setsockopt(zmq.RCVHWM, 50)
        self.sub.connect(f'tcp://{host}:{state_port}')

        self.pub_cmd = self.ctx.socket(zmq.PUB)
        self.pub_cmd.setsockopt(zmq.SNDHWM, 2)
        self.pub_cmd.connect(f'tcp://{host}:{cmd_port}')

        sensor_qos = QoSPresetProfiles.SENSOR_DATA.value
        self.pub_img = self.create_publisher(
            Image, 'carsim/image_raw', sensor_qos)
        self.pub_odom = self.create_publisher(
            Odometry, 'carsim/odom', sensor_qos)
        self.pub_lat = self.create_publisher(
            Float32, 'carsim/latency_ms', 10)
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
            f'bridge active  state<-tcp://{host}:{state_port}  '
            f'cmd->tcp://{host}:{cmd_port}')

    def drain_state(self):
        """Drain the ZMQ queue and return ALL pending frames, in arrival
        order.

        Used to keep only the latest frame, to avoid ever processing a
        stale state in a real-time control loop. Measured on the VM:
        UTM's virtualised network occasionally delivers 2 frames in the
        same poll window even at 200 Hz (4x the 50 Hz publish rate) --
        largest observed burst was 2, over 1600 ticks / 8s. Keeping only
        the last of each burst discarded ~40% of states and ~70% of
        images (image capture is already 1 tick in 2, so it's
        disproportionately the older frame of a burst that gets
        dropped). Every frame drained here is still fresh (it arrived
        within the same few-ms poll window), so processing all of them
        doesn't reintroduce the latency the previous approach avoided.
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

        lat_ms = (P.now() - header['t_pub']) * 1e3
        self.lat_sum += lat_ms
        self.n_state += 1
        if self.last_seq >= 0:
            self.n_skipped += max(0, header['seq'] - self.last_seq - 1)
        self.last_seq = header['seq']
        self.pub_lat.publish(Float32(data=float(lat_ms)))

        self.pub_odom.publish(self.make_odom(header))

        if img_bytes is not None:
            self.pub_img.publish(self.make_image(header, img_bytes))
            self.n_img += 1

    def make_odom(self, header):
        p, t = header['pose'], header['twist']
        msg = Odometry()
        msg.header.stamp = sim_time_to_stamp(header['t_sim'])
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

    def make_image(self, header, payload):
        meta = header['img']
        msg = Image()
        msg.header.stamp = sim_time_to_stamp(header['t_sim'])
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
            self.get_logger().warn(
                'no state received -- is the sim running? right IP?')
            return
        self.get_logger().info(
            f'state {self.n_state / 2.0:5.1f} Hz | '
            f'img {self.n_img / 2.0:5.1f} Hz | '
            f'latency {self.lat_sum / self.n_state:5.2f} ms | '
            f'skipped {self.n_skipped} | cmd {self.cmd_seq}')
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
