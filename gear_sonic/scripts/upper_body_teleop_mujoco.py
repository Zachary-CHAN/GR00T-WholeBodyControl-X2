#!/usr/bin/env python
"""MuJoCo harness for the AimSDK upper-body teleop (Plan D) retarget stack.

Replays the EXACT PC2-side pipeline -- decode (IntentReceiver) -> analytic
retarget (ArmRetargeter) -> engage/stale gate -> slew -> limit clamp -> the
same kp/kd tables -- but drives the X2 MJCF in MuJoCo instead of the hal
topics, with the floating base pinned (the gantry model). Purpose: quantify
what the robot can/cannot reach before touching the real machine -- e.g. how
much of "arm straight overhead" is lost to the -3.08 rad shoulder_pitch limit
vs. gravity droop under kp=30/20 position servos.

Two input modes:
  --pose <name>   synthetic SMPL poses (down/forward/lateral/up), no ZMQ --
                  instant repro of the overhead question.
  --wire          SUB-bind :intent-port/:cmd-port and consume the REAL laptop
                  sender (tape or live, wrist FRAME conversion + head logic
                  included -- full-fidelity):
                  pico_intent_sender.py --pc2-host 127.0.0.1 --intent-port 15573 \
                      --planner-cmd-port 15563 --tape-replay <npz> --auto-engage

Usage:
    .venv/bin/python gear_sonic/scripts/upper_body_teleop_mujoco.py --pose up
    .venv/bin/python gear_sonic/scripts/upper_body_teleop_mujoco.py --wire --intent-port 15573
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
_sdk = (Path.home() / "Documents" / "x2_sdk_v100" / "src" / "py_examples"
        / "py_examples")
sys.path.insert(0, str(_sdk))
sys.path.insert(0, str(REPO / "src" / "py_examples"))
try:
    import upper_body_teleop as ubt
except ImportError as e:  # pragma: no cover
    sys.exit(f"cannot import upper_body_teleop.py from x2_sdk_v100: {e}")

import mujoco  # noqa: E402

MJCF = REPO / "gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml"


# --- synthetic SMPL poses (same conventions as ubt.run_self_test) -----------
def synth_pose(name: str) -> np.ndarray:
    j = np.zeros((24, 3))
    j[12] = [0, 0, 0.6]                      # neck above pelvis
    j[16], j[17] = [0, 0.2, 0.5], [0, -0.2, 0.5]   # shoulder line
    # neutral arms down
    j[18], j[19] = [0, 0.2, 0.25], [0, -0.2, 0.25]
    j[20], j[21] = [0, 0.2, 0.0], [0, -0.2, 0.0]
    if name == "forward":                     # both arms straight ahead
        j[18], j[20] = [-0.3, 0.2, 0.5], [-0.55, 0.2, 0.5]
        j[19], j[21] = [-0.3, -0.2, 0.5], [-0.55, -0.2, 0.5]
    elif name == "lateral":                   # left arm out sideways
        j[18], j[20] = [0, 0.45, 0.5], [0, 0.7, 0.5]
    elif name == "up":                        # BOTH arms straight overhead
        j[18], j[20] = [0, 0.2, 0.75], [0, 0.2, 1.0]
        j[19], j[21] = [0, -0.2, 0.75], [0, -0.2, 1.0]
    elif name != "down":
        sys.exit(f"unknown pose '{name}' (down/forward/lateral/up)")
    return j


def build_model():
    m = mujoco.MjModel.from_xml_path(str(MJCF))
    d = mujoco.MjData(m)
    # hold pose start: legs/waist at the stand capture, arms/head neutral
    hold = {l[0]: l[1] for l in ubt.LEG}
    hold.update({w[0]: w[3] for w in ubt.WAIST})
    for jid in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        adr = m.jnt_qposadr[jid]
        if m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            d.qpos[adr + 2] = 1.05            # hang height, feet clear
        elif name in hold:
            d.qpos[adr] = hold[name]
    mujoco.mj_forward(m, d)
    return m, d


def joint_maps(m):
    """name -> (qpos_adr, dof_adr, actuator_id, ctrl_lo, ctrl_hi)."""
    out = {}
    for jid in range(m.njnt):
        if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                f"motor_{name}")
        out[name] = (m.jnt_qposadr[jid], m.jnt_dofadr[jid], aid,
                     m.actuator_ctrlrange[aid][0], m.actuator_ctrlrange[aid][1])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pose", choices=["down", "forward", "lateral", "up"],
                    help="synthetic SMPL pose (no ZMQ)")
    ap.add_argument("--wire", action="store_true",
                    help="consume the real sender over ZMQ")
    ap.add_argument("--intent-port", type=int, default=15573)
    ap.add_argument("--cmd-port", type=int, default=15563)
    ap.add_argument("--stale-s", type=float, default=0.5)
    ap.add_argument("--slew", type=float, default=0.06)
    ap.add_argument("--settle-s", type=float, default=4.0,
                    help="pose mode: hold time before reporting steady state")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="wire mode: stop after N s (0 = until viewer closes)")
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args()
    if not args.pose and not args.wire:
        ap.error("need --pose <name> or --wire")

    m, d = build_model()
    jm = joint_maps(m)
    missing = [n for n in list(ubt.ARM_NAMES) + [l[0] for l in ubt.LEG]
               + [w[0] for w in ubt.WAIST] + [h[0] for h in ubt.HEAD]
               if n not in jm]
    if missing:
        sys.exit(f"MJCF lacks joints present in the hal tables: {missing}")

    retargeter = ubt.ArmRetargeter()
    rx = None
    if args.wire:
        rx = ubt.IntentReceiver(args.intent_port, args.cmd_port)
        rx.start()
        print(f"[sim] wire mode: SUB :{args.intent_port}/:{args.cmd_port}")

    # per-joint targets/gains, same tables as the live node
    tgt = {}
    for n in ubt.ARM_NAMES:
        i = ubt.ARM_NAMES.index(n)
        tgt[n] = (0.0, ubt.ARM_KP[i], ubt.ARM_KD[i])
    for l in ubt.LEG:
        tgt[l[0]] = (l[1], l[2], l[3])
    for w in ubt.WAIST:
        tgt[w[0]] = (w[3], w[4], w[5])
    for h in ubt.HEAD:
        tgt[h[0]] = (h[3], h[4], h[5])
    limits = {n: jm[n][0] for n in tgt}  # placeholder, real clamp below
    lo = dict(zip(ubt.ARM_NAMES, ubt.ARM_LO))
    hi = dict(zip(ubt.ARM_NAMES, ubt.ARM_HI))
    for l in ubt.LEG:
        lo[l[0]], hi[l[0]] = l[1], 100.0   # leg upper bound unknown in table
    for w in ubt.WAIST:
        lo[w[0]], hi[w[0]] = w[1], w[2]
    for h in ubt.HEAD:
        lo[h[0]], hi[h[0]] = h[1], h[2]

    state_arm = ubt.NEUTRAL_ARM.copy()
    state_head = ubt.NEUTRAL_HEAD.copy()

    def targets_to_dict(arm14, head2):
        t = {n: arm14[i] for i, n in enumerate(ubt.ARM_NAMES)}
        t[ubt.HEAD[0][0]] = head2[0]
        t[ubt.HEAD[1][0]] = head2[1]
        return t

    def pd_and_step(tdict):
        # zero-order-hold the 50 Hz torque command across 1 kHz substeps --
        # the real hal servo runs at high rate; a single 20 ms mujoco step
        # with these kp/kd is numerically unstable (violent oscillation).
        root_adr = m.jnt_qposadr[0]
        root0 = d.qpos[root_adr:root_adr + 7].copy()
        for name, (q_des, kp, kd) in tdict.items():
            adr, dof, aid, clo, chi = jm[name]
            q, qd = d.qpos[adr], d.qvel[dof]
            tau = kp * (q_des - q) - kd * qd
            d.ctrl[aid] = np.clip(tau, clo, chi)
        for _ in range(SUBSTEPS):
            mujoco.mj_step(m, d)
        # gantry: hard-pin the floating base every frame
        d.qpos[root_adr:root_adr + 7] = root0
        d.qvel[m.jnt_dofadr[0]:m.jnt_dofadr[0] + 6] = 0.0

    viewer = None
    if not args.no_viewer:
        from mujoco import viewer as mj_viewer  # local alias only
        viewer = mj_viewer.launch_passive(m, d)

    dt, rate = 0.02, 50.0
    SUBSTEPS = 20                      # 1 kHz physics under 50 Hz commands
    m.opt.timestep = dt / SUBSTEPS
    t0 = time.monotonic()
    err_log = {n: [] for n in ubt.ARM_NAMES}

    if args.pose:
        arm_t = retargeter.frame(synth_pose(args.pose))
        arm_t = np.clip(arm_t, ubt.ARM_LO, ubt.ARM_HI)
        print(f"[sim] pose={args.pose}: raw targets set, settling "
              f"{args.settle_s}s @50Hz (gravity droop will show as "
              f"target-actual error)")
        n_settle = int(args.settle_s * rate)
        for k in range(n_settle):
            state_arm = ubt.slew_toward(state_arm, arm_t, args.slew)
            state_arm = np.clip(state_arm, ubt.ARM_LO, ubt.ARM_HI)
            pd_and_step({**{n: (state_arm[i], *tgt[n][1:]) for i, n
                            in enumerate(ubt.ARM_NAMES)},
                         **{n: (tgt[n][0], tgt[n][1], tgt[n][2])
                            for n in tgt if n not in ubt.ARM_NAMES}})
            if viewer is not None:
                viewer.sync()
            if k > n_settle - rate:  # last second: log steady state
                for i, n in enumerate(ubt.ARM_NAMES):
                    err_log[n].append(state_arm[i] - d.qpos[jm[n][0]])
        print(f"\n{'joint':34s} {'target':>8s} {'actual':>8s} {'err(deg)':>9s}")
        for i, n in enumerate(ubt.ARM_NAMES):
            e = np.mean(err_log[n]) if err_log[n] else 0.0
            print(f"{n:34s} {state_arm[i]:+8.3f} "
                  f"{d.qpos[jm[n][0]]:+8.3f} {np.degrees(e):+9.2f}")
        if viewer is not None:  # hold the pose so it can be inspected
            print("[sim] holding pose -- close the viewer window to exit")
            while viewer.is_running():
                pd_and_step({**{n: (state_arm[i], *tgt[n][1:]) for i, n
                                in enumerate(ubt.ARM_NAMES)},
                             **{n: (tgt[n][0], tgt[n][1], tgt[n][2])
                                for n in tgt if n not in ubt.ARM_NAMES}})
                viewer.sync()
    else:
        deadline = t0 + args.duration if args.duration > 0 else None
        last_stat = 0.0
        while True:
            if deadline and time.monotonic() > deadline:
                break
            if viewer is not None and not viewer.is_running():
                break
            intent, head, nerr = (rx.snapshot() if rx else (None, None, 0))
            arm_t, head_yaw, engaged, note = ubt.compute_targets(
                retargeter, intent, head, args.stale_s)
            state_arm = ubt.slew_toward(state_arm, arm_t, args.slew)
            state_arm = np.clip(state_arm, ubt.ARM_LO, ubt.ARM_HI)
            state_head = ubt.slew_toward(
                state_head, np.array([head_yaw, 0.0]), args.slew)
            state_head = np.clip(state_head,
                                 np.array([h[1] for h in ubt.HEAD]),
                                 np.array([h[2] for h in ubt.HEAD]))
            tdict = {n: (state_arm[i], *tgt[n][1:])
                     for i, n in enumerate(ubt.ARM_NAMES)}
            tdict[ubt.HEAD[0][0]] = (state_head[0], ubt.HEAD[0][4], ubt.HEAD[0][5])
            tdict[ubt.HEAD[1][0]] = (state_head[1], ubt.HEAD[1][4], ubt.HEAD[1][5])
            tdict.update({n: (tgt[n][0], tgt[n][1], tgt[n][2])
                          for n in tgt if n not in ubt.ARM_NAMES
                          and n not in (ubt.HEAD[0][0], ubt.HEAD[1][0])})
            pd_and_step(tdict)
            if viewer is not None:
                viewer.sync()
            else:
                time.sleep(dt)
            now = time.monotonic()
            if now - last_stat > 2.0:
                last_stat = now
                errs = [abs(state_arm[i] - d.qpos[jm[n][0]])
                        for i, n in enumerate(ubt.ARM_NAMES)]
                lsp = ubt.ARM_NAMES.index("left_shoulder_pitch_joint")
                print(f"[sim] engaged={int(engaged)} err={nerr} "
                      f"max|err|={np.degrees(max(errs)):.1f}deg "
                      f"L_sh_pitch tgt={state_arm[lsp]:+.3f} "
                      f"act={d.qpos[jm[ubt.ARM_NAMES[lsp]][0]]:+.3f}")

    # NOTE: no explicit viewer.close() -- calling it segfaults at teardown
    # on this stack; letting the interpreter exit after the window is gone
    # is clean.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
