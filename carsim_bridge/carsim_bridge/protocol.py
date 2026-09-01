"""Protocole du pont sim <-> ROS2.

Volontairement sans dependance a Python (pas de pickle) : le Mac tourne en
3.10, la VM en 3.12. Tout passe en JSON + buffer brut.

  Etat   (sim -> ros) : multipart [b"state", header_json, img_bytes|b""]
  Commande (ros -> sim): [cmd_json]
"""
import json
import time

TOPIC_STATE = b"state"
PROTOCOL_VERSION = 1


def now() -> float:
    """Horloge murale unique pour toutes les mesures de latence."""
    return time.time()


def encode_state(seq, t_sim, pose, twist, img=None):
    """pose=(x,y,yaw) ; twist=(vx,vy,yaw_rate) ; img=ndarray HxWx3 uint8 ou None."""
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
    """steer: angle de braquage [rad] ; accel: commande longitudinale [-1, 1]."""
    return [json.dumps({
        "v": PROTOCOL_VERSION,
        "seq": seq,
        "t_pub": now(),
        "steer": float(steer),
        "accel": float(accel),
    }).encode()]


def decode_cmd(frames):
    return json.loads(frames[0].decode())
