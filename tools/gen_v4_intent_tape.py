#!/usr/bin/env python3
"""Synthesize a v4 intent tape -- the exact shape ``pico_intent_sender --record`` writes.

Purpose: exercise the incremental (clutch-IK) upper teleop chain with no
headset attached.  Feed the result to the sender with::

    .venv/bin/python gear_sonic/scripts/pico_intent_sender.py \
        --tape-replay <tape.npz> --pc2-host <host> --intent-port 5573 \
        --hand-source off --no-locomotion

Profiles:
* ``arm`` (default, 24 s @ 50 Hz): 3 s idle -> LEFT trigger held 6 s with a
  slow figure motion -> release 3 s (the arm must HOLD) -> RIGHT trigger
  held 6 s -> both 4 s with a 25 deg torso twist (waist yaw must follow)
  -> release 2 s.
* ``wrist`` (v3, 24 s): position held constant, pure controller-QUATERNION
  sweeps (z 0->+-120 deg, then x/y single-axis segments) with hold
  windows, and grip PRESSED (=1, v3 closes on press) duty windows.  The
  j24 skeleton still twists 25 deg during the right-arm window -- the v3
  chain must ignore it (waist stays at stand), which the offline autopsy
  asserts alongside the wrist sign checks.

Ctrl poses are framed in operator torso axes (fwd/lat/up, pelvis origin),
which is what the recorder stores.

The SMPL skeleton MUST be non-degenerate: the waist chain compares the
shoulder line against the HIP line, and zero hip joints make that angle pure
noise -- a tape with j24[1]/j24[2] left at zero drove the waist at +-180 deg
(found the hard way).  Hips/collars/elbows/wrists are filled in below.
"""

from __future__ import annotations

import argparse

import numpy as np

FPS = 50.0
DT = 1.0 / FPS


def rot_x(deg):
    a = np.deg2rad(deg)
    return np.array([np.cos(a / 2), np.sin(a / 2), 0.0, 0.0])


def rot_y(deg):
    a = np.deg2rad(deg)
    return np.array([np.cos(a / 2), 0.0, np.sin(a / 2), 0.0])


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


def _standing_j24(n_frames: int, rng) -> np.ndarray:
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
    j24 += rng.normal(0, 0.002, j24.shape).astype(np.float32)
    return j24


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
    grip[(t >= 18) & (t < 22), 0] = 0.6

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
    j24 = _standing_j24(n_frames, rng)
    for i, ti in enumerate(t):
        if 18.0 <= ti < 22.0:
            a = np.deg2rad(25.0 * np.sin(2 * np.pi * (ti - 18.0) / 4.0))
            c, s = np.cos(a), np.sin(a)
            for j in (12, 13, 14, 16, 17):
                x, y, z = j24[i, j]
                j24[i, j] = (x * c - y * s, x * s + y * c, z)

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


def make_wrist(n_frames: int = 1200, seed: int = 7):
    """v3 wrist-direction tape: positions constant, pure quat sweeps.

    Schedule (s): idle 0-3 | L held 3-12: z 0->+120 (3-6), hold +120 with
    grip PRESSED (6-8), z back to 0 (8-9), x 0->+60 (9-11), hold (11-12) |
    release 12-15 (freeze) | R held 15-21: z 0->-120 (15-18), hold -120
    with grip pressed (18-19), z back (19-20), y 0->-60 (20-21) | end.
    The j24 twists 25 deg during 15-21 -- v3 must ignore it.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_frames) * DT
    trig = np.zeros((n_frames, 2), np.float32)
    trig[(t >= 3) & (t < 12), 0] = 0.9
    trig[(t >= 15) & (t < 21), 1] = 0.9
    # v3: grip 1 = PRESSED = closed; duty windows sit inside the sweeps
    grip = np.zeros((n_frames, 2), np.float32)
    grip[(t >= 6) & (t < 8), 0] = 0.8
    grip[(t >= 18) & (t < 19), 1] = 0.6

    l_rest = np.array([0.10, 0.30, -0.25])
    r_rest = np.array([0.10, -0.30, -0.25])
    lc = np.zeros((n_frames, 7), np.float32)
    rc = np.zeros((n_frames, 7), np.float32)
    for i, ti in enumerate(t):
        lc[i, :3] = l_rest
        rc[i, :3] = r_rest
        q = np.array([1.0, 0.0, 0.0, 0.0])
        if 3.0 <= ti < 6.0:
            q = rot_z(120.0 * (ti - 3.0) / 3.0)
        elif 6.0 <= ti < 8.0:
            q = rot_z(120.0)
        elif 8.0 <= ti < 9.0:
            q = rot_z(120.0 * (1.0 - (ti - 8.0)))
        elif 9.0 <= ti < 11.0:
            q = rot_x(60.0 * (ti - 9.0) / 2.0)
        elif 11.0 <= ti < 12.0:
            q = rot_x(60.0)
        lc[i, 3:] = q
        q = np.array([1.0, 0.0, 0.0, 0.0])
        if 15.0 <= ti < 18.0:
            q = rot_z(-120.0 * (ti - 15.0) / 3.0)
        elif 18.0 <= ti < 19.0:
            q = rot_z(-120.0)
        elif 19.0 <= ti < 20.0:
            q = rot_z(-120.0 * (1.0 - (ti - 19.0)))
        elif 20.0 <= ti < 21.0:
            q = rot_y(-60.0 * (ti - 20.0))
        rc[i, 3:] = q
    lc[:, :3] += rng.normal(0, 0.002, (n_frames, 3))
    rc[:, :3] += rng.normal(0, 0.002, (n_frames, 3))

    j24 = _standing_j24(n_frames, rng)
    for i, ti in enumerate(t):
        if 15.0 <= ti < 21.0:
            a = np.deg2rad(25.0 * np.sin(2 * np.pi * (ti - 15.0) / 6.0))
            c, s = np.cos(a), np.sin(a)
            for j in (12, 13, 14, 16, 17):
                x, y, z = j24[i, j]
                j24[i, j] = (x * c - y * s, x * s + y * c, z)

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
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "path", nargs="?", default="/tmp/v4_tape.npz", help="output .npz (default /tmp/v4_tape.npz)"
    )
    ap.add_argument(
        "--profile",
        choices=("arm", "wrist"),
        default="arm",
        help="arm = v2 figure motion (default); wrist = v3 pure-quat sweeps",
    )
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--frames", type=int, default=1200)
    args = ap.parse_args()
    d = (
        make(args.frames, args.seed)
        if args.profile == "arm"
        else make_wrist(args.frames, args.seed)
    )
    np.savez_compressed(args.path, **d)
    n = len(d["t"])
    print(
        f"wrote {args.path}: profile={args.profile} {n} frames ({n * DT:.1f}s), "
        f"trigger>0.5 frames L={int((d['trigger'][:, 0] > 0.5).sum())} "
        f"R={int((d['trigger'][:, 1] > 0.5).sum())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
