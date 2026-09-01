#!/usr/bin/env python3
"""Serveur de simulation MuJoCo -- tourne sur macOS.

Joue le role du "plant" (le robot). Il n'y a AUCUNE intelligence ici :
il expose des capteurs et accepte des commandes actionneur, exactement
comme le fera le Raspberry Pi + chassis reel en Coree.

    python3 sim/sim_server.py --bind 0.0.0.0
"""
import argparse
import math
import os
import time

import mujoco
import numpy as np
import zmq

import protocol as P

WHEELBASE = 0.26   # empattement [m]
TRACK = 0.21       # voie [m]
MAX_TORQUE = 2.0   # couple moteur max [N.m]

# Anchored to this file's own location, not the CWD -- the docstring's
# usage example assumes `cd sim && python3 sim_server.py`, but nothing
# enforces that; run as `python3 sim/sim_server.py` from the repo root
# (as everything else in this project is), a CWD-relative "models/car.xml"
# resolves to a path that doesn't exist.
DEFAULT_MODEL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "car.xml")


def ackermann(delta):
    """Angle de braquage bicyclette -> angles roue gauche / droite."""
    if abs(delta) < 1e-4:
        return delta, delta
    R = WHEELBASE / math.tan(delta)
    return (math.atan(WHEELBASE / (R - TRACK / 2)),
            math.atan(WHEELBASE / (R + TRACK / 2)))


def yaw_from_quat(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--state-port", type=int, default=5555)
    ap.add_argument("--cmd-port", type=int, default=5556)
    ap.add_argument("--ctrl-hz", type=float, default=50.0)
    ap.add_argument("--cam-hz", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--realtime", action="store_true", default=True)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)

    renderer = None
    if not args.no_camera:
        renderer = mujoco.Renderer(model, args.height, args.width)

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 2)          # on jette les vieilles trames
    pub.bind(f"tcp://{args.bind}:{args.state_port}")

    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.setsockopt(zmq.RCVHWM, 2)
    sub.bind(f"tcp://{args.bind}:{args.cmd_port}")

    dt_ctrl = 1.0 / args.ctrl_hz
    n_sub = max(1, round(dt_ctrl / model.opt.timestep))
    # Le rendu ne peut se produire qu'a un tick de controle : on force un
    # diviseur entier, sinon le debit derive (30 Hz demandes -> 25 Hz reels).
    cam_every = max(1, round(args.ctrl_hz / args.cam_hz))
    cam_hz_real = args.ctrl_hz / cam_every

    print(f"[sim] etat   PUB tcp://{args.bind}:{args.state_port}")
    print(f"[sim] cmd    SUB tcp://{args.bind}:{args.cmd_port}")
    if args.no_camera:
        cam_status = "off"
    else:
        cam_status = f"{cam_hz_real:.1f} Hz reels (1 tick sur {cam_every})"
    print(f"[sim] ctrl {args.ctrl_hz:.0f} Hz | cam {cam_status}")

    seq = 0
    steer_cmd = 0.0
    accel_cmd = 0.0
    cmd_seq = -1
    tick = 0
    t_wall0 = time.time()
    stats_t = time.time()
    n_cmd = 0
    lat_cmd_sum = 0.0

    try:
        while True:
            # --- commandes : on vide la file, on ne garde que la derniere ---
            while True:
                try:
                    frames = sub.recv_multipart(zmq.NOBLOCK)
                except zmq.Again:
                    break
                c = P.decode_cmd(frames)
                steer_cmd = float(np.clip(c["steer"], -0.6, 0.6))
                accel_cmd = float(np.clip(c["accel"], -1.0, 1.0))
                cmd_seq = c["seq"]
                # Latence apparente ros -> sim. Combinee a la mesure
                # inverse cote pont, elle permet de separer le vrai
                # temps de transit du decalage d'horloge (cf. README).
                lat_cmd_sum += (P.now() - c["t_pub"]) * 1e3
                n_cmd += 1

            # --- actionneurs ---
            dl, dr = ackermann(steer_cmd)
            torque = accel_cmd * MAX_TORQUE
            data.ctrl[0] = dl
            data.ctrl[1] = dr
            data.ctrl[2] = torque
            data.ctrl[3] = torque

            # --- physique ---
            for _ in range(n_sub):
                mujoco.mj_step(model, data)

            # --- capteurs ---
            pos = data.body("chassis").xpos
            quat = data.body("chassis").xquat
            vel = data.sensor("s_vel").data
            gyro = data.sensor("s_gyro").data
            pose = (float(pos[0]), float(pos[1]), yaw_from_quat(quat))
            twist = (float(vel[0]), float(vel[1]), float(gyro[2]))

            tick += 1
            img = None
            if renderer is not None and tick % cam_every == 0:
                renderer.update_scene(data, camera="cam_front")
                img = renderer.render()

            pub.send_multipart(P.encode_state(seq, data.time, pose, twist, img))
            seq += 1

            if time.time() - stats_t > 2.0:
                rtf = data.time / (time.time() - t_wall0)
                lat_rev = lat_cmd_sum / n_cmd if n_cmd else float("nan")
                print(f"[sim] t={data.time:7.1f}s  RTF={rtf:4.2f}  "
                      f"v={twist[0]:5.2f} m/s  cmds/s="
                      f"{n_cmd / (time.time() - stats_t):4.0f}  "
                      f"lat ros->sim={lat_rev:8.2f} ms")
                stats_t = time.time()
                n_cmd = 0
                lat_cmd_sum = 0.0

            if args.realtime:
                lag = (t_wall0 + data.time) - time.time()
                if lag > 0:
                    time.sleep(lag)
    except KeyboardInterrupt:
        print("\n[sim] arret")
    finally:
        pub.close()
        sub.close()
        ctx.term()


if __name__ == "__main__":
    main()
