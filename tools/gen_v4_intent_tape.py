#!/usr/bin/env python3
"""Synthesize a v4 intent tape -- the exact shape ``pico_intent_sender --record`` writes.

Purpose: exercise the incremental (clutch-IK) upper teleop chain with no
headset attached.  Feed the result to the sender with::

    .venv/bin/python gear_sonic/scripts/pico_intent_sender.py \
        --tape-replay <tape.npz> --pc2-host <host> --intent-port 5573 \
        --hand-source off --no-locomotion

Script (24 s @ 50 Hz): 3 s idle -> LEFT trigger held 6 s with a slow figure
motion -> release 3 s (the arm must HOLD) -> RIGHT trigger held 6 s -> both
4 s with a 25 deg torso twist (waist yaw must follow) -> release 2 s.

Ctrl poses are framed in operator torso axes (fwd/lat/up, pelvis origin),
which is what the recorder stores.

The SMPL skeleton MUST be non-degenerate: the waist chain compares the
shoulder line against the HIP line, and zero hip joints make that angle pure
noise -- a tape with j24[1]/j24[2] left at zero drove the waist at +-180 deg
(found the hard way).  Hips/collars/elbows/wrists are filled in below.
"""

from __future__ import annotations

import sys

import numpy as np

FPS = 50.0
DT = 1.0 / FPS


def rot_x(deg):
    a = np.deg2rad(deg)
    return np.array([np.cos(a / 2), np.sin(a / 2), 0.0, 0.0])


def rot_z(deg):
    a = np.deg2rad(deg)
    return np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def make(n_frames: int = 1200, seed: int = 7):
    rng = np.random.default_rng(seed)
    t = np.arange(n_frames) * DT
    # schedule (s):     idle 0-3 | L 3-9 | hold 9-12 | R 12-18 | both 18-22 | end 22-24
    trig = np.zeros((n_frames, 2), np.float32)
    trig[(t >= 3) & (t < 9), 0] = 0.9
    trig[(t >= 12) & (t < 18), 1] = 0.9
    trig[(t >= 18) & (t < 22)] = 0.9
    grip = np.zeros((n_frames, 2), np.float32)
    grip[(t >= 3) & (t < 9), 0] = 0.8
    grip[(t >= 18) & (t < 22)] = 0.6

    lc = np.zeros((n_frames, 7), np.float32)
    rc = np.zeros((n_frames, 7), np.float32)
    # rest: hands at hip height, slightly out to the sides, pelvis-origin frame
    l_rest = np.array([0.10, 0.30, -0.25])
    r_rest = np.array([0.10, -0.30, -0.25])
    for i, ti in enumerate(t):
        # slow raise/forward figure while the left trigger is held
        if 3.0 <= ti < 9.0:
            u = (ti - 3.0) / 6.0
            lc[i, :3] = l_rest + np.array(
                [
                    0.25 * np.sin(2 * np.pi * u),
                    0.10 * np.sin(4 * np.pi * u),
                    0.20 * u,
                ]
            )
            lc[i, 3:] = qmul(rot_z(20.0 * u), rot_x(-15.0 * np.sin(2 * np.pi * u)))
        else:
            lc[i, :3] = l_rest
            lc[i, 3:] = [1.0, 0.0, 0.0, 0.0]
        if 12.0 <= ti < 18.0:
            u = (ti - 12.0) / 6.0
            rc[i, :3] = r_rest + np.array(
                [
                    0.25 * np.sin(np.pi * u),
                    -0.10 * np.sin(4 * np.pi * u),
                    0.15 * u,
                ]
            )
            rc[i, 3:] = qmul(rot_z(-20.0 * u), rot_x(10.0 * (1.0 - np.cos(2 * np.pi * u)) / 2))
        else:
            rc[i, :3] = r_rest
            rc[i, 3:] = [1.0, 0.0, 0.0, 0.0]
    # sensor noise (f32 scale) -- also makes each frame's raw pose unique,
    # which is what lets the level-0 comparator align runs on input identity
    lc[:, :3] += rng.normal(0, 0.002, (n_frames, 3))
    rc[:, :3] += rng.normal(0, 0.002, (n_frames, 3))

    # body: standing skeleton, twist a little during the both-armed window
    # (standard SMPL-24: 0 pelvis | 1-2 hips | 3 spine-neck | 12 neck |
    #  13-14 collars | 16-17 shoulders | 18-19 elbows | 20-21 wrists).
    j24 = np.zeros((n_frames, 24, 3), np.float32)
    j24[:, 0] = (0.0, 0.0, 0.0)
    j24[:, 1] = (0.0, 0.09, -0.05)
    j24[:, 2] = (0.0, -0.09, -0.05)
    j24[:, 3] = (0.0, 0.0, 0.62)
    j24[:, 12] = (0.0, 0.0, 0.62)
    j24[:, 13] = (0.0, 0.10, 0.55)
    j24[:, 14] = (0.0, -0.10, 0.55)
    j24[:, 16] = (0.0, 0.18, 0.50)
    j24[:, 17] = (0.0, -0.18, 0.50)
    j24[:, 18] = (0.0, 0.20, 0.25)
    j24[:, 19] = (0.0, -0.20, 0.25)
    j24[:, 20] = (0.0, 0.20, 0.00)
    j24[:, 21] = (0.0, -0.20, 0.00)
    for i, ti in enumerate(t):
        if 18.0 <= ti < 22.0:
            a = np.deg2rad(25.0 * np.sin(2 * np.pi * (ti - 18.0) / 4.0))
            c, s = np.cos(a), np.sin(a)
            for j in (12, 13, 14, 16, 17):
                x, y, z = j24[i, j]
                j24[i, j] = (x * c - y * s, x * s + y * c, z)
    j24 += rng.normal(0, 0.002, j24.shape).astype(np.float32)

    return dict(
        t=t.astype(np.float64),
        smpl_joints=j24.reshape(n_frames, 72),
        human_quat=np.tile(np.array([1, 0, 0, 0], np.float32), (n_frames, 1)),
        engaged=np.zeros(n_frames, np.float32),
        wrist_pr=np.zeros((n_frames, 4), np.float32),
        trigger=trig,
        grip=grip,
        lctrl_pose=lc,
        rctrl_pose=rc,
    )


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/v4_tape.npz"
    d = make()
    np.savez_compressed(path, **d)
    n = len(d["t"])
    print(
        f"wrote {path}: {n} frames ({n * DT:.1f}s), "
        f"trigger>0.5 frames L={int((d['trigger'][:, 0] > 0.5).sum())} "
        f"R={int((d['trigger'][:, 1] > 0.5).sum())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
