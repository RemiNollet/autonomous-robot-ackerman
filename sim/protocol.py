"""Sim <-> ROS2 bridge protocol.

Deliberately no Python-version dependency (no pickle): the Mac runs
3.10, the VM runs 3.12. Everything goes over JSON + a raw buffer.

  State   (sim -> ros): multipart [b"state", header_json, img_bytes|b""]
  Command (ros -> sim): [cmd_json]
"""
import json
import time

TOPIC_STATE = b"state"
PROTOCOL_VERSION = 1


def now() -> float:
    """Single wall clock used for every latency measurement."""
    return time.time()


def encode_state(seq, t_sim, pose, twist, img=None):
    """pose=(x,y,yaw); twist=(vx,vy,yaw_rate); img=HxWx3 uint8 ndarray or
    None."""
    header = {
        "v": PROTOCOL_VERSION,
        "seq": seq,
        "t_sim": t_sim,
        "t_pub": now(),
        "pose": {"x": pose[0], "y": pose[1], "yaw": pose[2]},
        "twist": {"vx": twist[0], "vy": twist[1], "yaw_rate": twist[2]},
        "img": None,
    }
    payload = b""
    if img is not None:
        header["img"] = {
            "h": int(img.shape[0]),
            "w": int(img.shape[1]),
            "c": int(img.shape[2]),
            "encoding": "rgb8",
        }
        payload = img.tobytes()
    return [TOPIC_STATE, json.dumps(header).encode(), payload]


def decode_state(frames):
    """-> (header_dict, img_bytes|None)"""
    _, header_raw, payload = frames
    header = json.loads(header_raw.decode())
    return header, (payload if header.get("img") else None)


def encode_cmd(seq, steer, accel):
    """steer: steering angle [rad]; accel: longitudinal command [-1, 1]."""
    return [json.dumps({
        "v": PROTOCOL_VERSION,
        "seq": seq,
        "t_pub": now(),
        "steer": float(steer),
        "accel": float(accel),
    }).encode()]


def decode_cmd(frames):
    return json.loads(frames[0].decode())
