#!/usr/bin/env python3
"""MuJoCo harness for the v3 arm teleop (4-DoF arm IK + direct wrist fill).

v2 (incremental_upper_teleop_mujoco.py) keeps the wrist inside the 7-DoF
6-DoF DLS solve; on the robot the gripper shows spurious orientations
(narrow wrist limits x open-loop differential anchoring x min-norm
rotation spreading over the whole arm). v3 changes three things
(chain + rationale in
``gear_sonic/utils/teleop/incremental_chain_v3.py``):

* SMPL channel closed: the waist holds a fixed stand value, smpl_joints
  is ignored end to end;
* arm IK defaults to the v1-style 4-DoF split: position-only solve over
  shoulder-3 + elbow, the three wrist joints LOCKED at the wrist command
  (re-fed into the FK every tick -- wrist_yaw has an ~82 mm EE lever) and
  driven by the controller quaternion differential through the EXACT wrist
  chain solve ``W(q_tgt) = W(q_wa) @ C delta C^T``
  (``W = Rz(yaw)Ry(pitch)Rx(roll)``; slew + near-pi guard + joint
  limits). The wrist is yaw-proximal with pitch/roll cross-coupled, so a
  tool pitch/yaw rotation coordinates pitch+roll off neutral; tool roll is
  always the roll joint alone. ``--full-pose-ik`` opts into the 7-DoF pose
  solve instead (position and orientation task gains both 1.0);
* HAND INVERTED: the hand is OPEN at rest and CLOSES when the TRIGGER is
  pressed (the LPF'd trigger is the hand channel); the GRIP is the per-arm
  clutch (2026-10 role swap; the constant / flag names are historical).
  A stale frame OPENS the hands (v2 closed them) -- --stale-grip-close
  holds the last filtered trigger if that ever matters more than dropping
  the object.

Everything else (grip clutch, dual anchors, torso keep-out sandwich,
reach sphere, target slew, freeze guard, pose LPF, 20x1 ms substeps
with per-substep tau refresh) is inherited from the v2 chain.

Usage:
    # T1 (laptop/sim host): pico_intent_sender.py --pc2-host 127.0.0.1 \
    #        --intent-port 15573 --hand-source off   (old :5572 path off)
    .venv/bin/python gear_sonic/scripts/arm_teleop_v3_mujoco.py \
        --wire --intent-port 15573
    .venv/bin/python gear_sonic/scripts/arm_teleop_v3_mujoco.py --self-test
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# iutm's import side effects (SDK path injection + sys.exit if missing)
# are the same precondition the v2 harness and its pytest mirror already
# carry; importing it also gives us the plant functions verbatim.
import incremental_upper_teleop_mujoco as iutm  # noqa: E402

from gear_sonic.utils.teleop.incremental_chain import (  # noqa: E402
    IncrementalIntentReceiver,
)
from gear_sonic.utils.teleop.incremental_chain_v3 import (  # noqa: E402
    CHECKS as CHAIN_V3_CHECKS,
    DEFAULT_WRIST_SIGNS,
    IncrementalTeleopTickerV3,
    parse_wrist_axis_order,
    run_self_test as _chain_v3_run_self_test,
)
from gear_sonic.utils.teleop.x2_hand_retarget import (  # noqa: E402
    grasp_command_from_ratio,
)

ubt = iutm.ubt  # the SDK-side tables (ARM_NAMES / WAIST / LEG / HEAD ...)


# ---------------------------------------------------------------------------
# Self-test checks that need the plant (pytest mirrors these 1:1)
# ---------------------------------------------------------------------------
def check_wire_roundtrip() -> None:
    iutm.check_wire_roundtrip()


def check_grip_mapping_v3() -> None:
    """v3: r = hand DIRECTLY (0 = open at rest, 1 = closed on press) --
    every mapping in v2's check_grip_mapping is INVERTED here, including
    the stale path (a stale frame forces hand 0 -> hand OPEN; v2 closed
    it). The hand signal is now the LPF'd TRIGGER (2026-10 swap); the
    ratio math itself is unchanged. See also check_grip_stale_open in the
    chain module (historical name)."""
    for side in ("left", "right"):
        open_q = grasp_command_from_ratio(side, 0.0)
        closed_q = grasp_command_from_ratio(side, 1.0)

        def r(g: float) -> float:
            return float(np.clip(np.clip(g, 0.0, 1.0), 0.0, 1.0))

        np.testing.assert_allclose(
            grasp_command_from_ratio(side, r(1.0)), closed_q, atol=1e-12
        )  # grip pressed = closed
        np.testing.assert_allclose(
            grasp_command_from_ratio(side, r(0.0)), open_q, atol=1e-12
        )  # rest/zero (also the stale value) = open
        mid = grasp_command_from_ratio(side, r(0.5))
        assert np.all(mid >= np.minimum(open_q, closed_q) - 1e-9)
        assert np.all(mid <= np.maximum(open_q, closed_q) + 1e-9)


def check_keepout_matches_model() -> None:
    iutm.check_keepout_matches_model()


def check_limits_match_sdk() -> None:
    iutm.check_limits_match_sdk()


def check_neutral_matches_vr_defaults() -> None:
    iutm.check_neutral_matches_vr_defaults()


def check_model_and_head_hold() -> None:
    iutm.check_model_and_head_hold()


# The gate = the v3 chain's pure checks + the plant-dependent ones. The
# v2-only semantics checks (waist, torso frame, pose_err_cap) are NOT
# re-run: v3 removed those paths and its own module pins what replaced
# them. Listed EXPLICITLY because the pytest mirror calls
# a3.check_<name>() directly.
CHECKS = list(CHAIN_V3_CHECKS) + [
    ("wire_roundtrip", check_wire_roundtrip),
    ("grip_mapping_v3", check_grip_mapping_v3),
    ("keepout_matches_model", check_keepout_matches_model),
    ("limits_match_sdk", check_limits_match_sdk),
    ("neutral_matches_vr_defaults", check_neutral_matches_vr_defaults),
    ("model_and_head_hold", check_model_and_head_hold),
]


def run_self_test() -> bool:
    return _chain_v3_run_self_test(CHECKS)


# ---------------------------------------------------------------------------
# Main wire loop
# ---------------------------------------------------------------------------
def _env_sign(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def main() -> int:
    signs_env = {
        "left": {
            "yaw": _env_sign("WRIST3_YAW_SIGN_L", DEFAULT_WRIST_SIGNS["left"][0]),
            "pitch": _env_sign("WRIST3_PITCH_SIGN_L", DEFAULT_WRIST_SIGNS["left"][1]),
            "roll": _env_sign("WRIST3_ROLL_SIGN_L", DEFAULT_WRIST_SIGNS["left"][2]),
        },
        "right": {
            "yaw": _env_sign("WRIST3_YAW_SIGN_R", DEFAULT_WRIST_SIGNS["right"][0]),
            "pitch": _env_sign("WRIST3_PITCH_SIGN_R", DEFAULT_WRIST_SIGNS["right"][1]),
            "roll": _env_sign("WRIST3_ROLL_SIGN_R", DEFAULT_WRIST_SIGNS["right"][2]),
        },
    }
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--wire", action="store_true", help="consume the laptop sender over ZMQ (else nothing runs)"
    )
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--intent-port", type=int, default=15573)
    ap.add_argument("--stale-s", type=float, default=0.5)
    ap.add_argument("--position-scale", type=float, default=1.0)
    ap.add_argument(
        "--ctrl-lever",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Pico controller origin -> operator wrist centre, metres in the "
        "controller body frame (W = C - R @ r). Default 0 keeps the raw "
        "controller origin. A pure wrist rotation with a stationary hand "
        "should leave the robot target still; if it sweeps, fit this vector "
        "(docs/x2/PICO_LIVE_TELEOP_PITFALLS.md). Common default for both "
        "hands; --ctrl-lever-left/right override it",
    )
    ap.add_argument(
        "--ctrl-lever-left",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="left-hand override of --ctrl-lever (fit each controller "
        "separately: the two hands are held at different orientations and "
        "the vectors differ by tens of mm)",
    )
    ap.add_argument(
        "--ctrl-lever-right",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="right-hand override of --ctrl-lever (see --ctrl-lever-left)",
    )
    ap.add_argument(
        "--position-deadband-mm",
        type=float,
        default=3.0,
        help="soft radial deadband on the compensated translation (mm); "
        "0 = off (default 3 mm: suppresses hand tremor / re-anchor creep)",
    )
    ap.add_argument(
        "--posture-gain",
        type=float,
        default=0.01,
        help="null-space posture bias toward the engage configuration; "
        "0 = off (default 0.01). MinkArmIK: PostureTask cost (1e-3..1e-2); "
        "ArmIKSolver: exact-projector null-space gain per tick (0.02..0.1)",
    )
    ap.add_argument(
        "--input-tau",
        type=float,
        default=0.12,
        help="LPF time constant on the wire controller poses (s); 0 = off",
    )
    ap.add_argument(
        "--target-slew",
        type=float,
        default=1.0,
        help="target slew limit (m/s) after the workspace clamp; 0 = off",
    )
    ap.add_argument("--trigger-thresh", type=float, default=iutm.TRIGGER_THRESH)
    ap.add_argument(
        "--full-pose-ik",
        action="store_true",
        help="opt into the 7-DoF pose IK (wrist solved; position and rotation "
        "gains both 1.0) instead of the default 4-DoF position solve + "
        "WristClutch direct wrist fill",
    )
    ap.add_argument(
        "--wrist-scale",
        type=float,
        default=1.0,
        help="gain on the controller differential (scales the rotation angle) "
        "before the exact wrist-chain solve",
    )
    for axis in ("yaw", "pitch", "roll"):
        ap.add_argument(
            f"--wrist-{axis}-sign-l",
            type=float,
            default=signs_env["left"][axis],
            help=f"left wrist {axis} sign (env WRIST3_{axis.upper()}_SIGN_L)",
        )
        ap.add_argument(
            f"--wrist-{axis}-sign-r",
            type=float,
            default=signs_env["right"][axis],
            help=f"right wrist {axis} sign (env WRIST3_{axis.upper()}_SIGN_R)",
        )
    for side, suffix in (("left", "l"), ("right", "r")):
        ap.add_argument(
            f"--wrist-axis-order-{suffix}",
            type=parse_wrist_axis_order,
            default=parse_wrist_axis_order(
                os.environ.get(f"WRIST3_AXIS_ORDER_{suffix.upper()}", "0,1,2")
            ),
            help=(
                f"{side} controller-axis -> wrist-joint correspondence "
                f"(env WRIST3_AXIS_ORDER_{suffix.upper()}); wrist joints are "
                "[yaw,pitch,roll], default 0,1,2 = controller z->yaw, y->pitch, "
                "x->roll. Applied to the differential before the exact wrist "
                "solve; not a per-joint scale"
            ),
        )
    ap.add_argument(
        "--wrist-slew",
        type=float,
        default=6.0,
        help="wrist command slew (rad/s); bounds the freeze-guard jump term",
    )
    ap.add_argument(
        "--wrist-pi-guard",
        type=float,
        default=2.94,
        help="drop differentials past this magnitude (rad): the ZYX split is "
        "ambiguous near pi; engage anchors must stay well under",
    )
    ap.add_argument(
        "--waist-stand",
        type=float,
        default=0.0,
        help="fixed waist yaw (rad) -- the SMPL waist channel is closed in v3",
    )
    ap.add_argument(
        "--stale-grip-close",
        action="store_true",
        help="stale frame HOLDS the last filtered trigger (the hand channel) "
        "instead of OPENING the hands (v3 default opens: dropping the object "
        "beats crushing it uncommanded)",
    )
    ap.add_argument(
        "--hand-collisions",
        action="store_true",
        help="keep OmniHand geom collisions ON (default OFF for "
        "pure teleop; enable for grasping experiments)",
    )
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument(
        "--duration", type=float, default=0.0, help="stop after N s (0 = until viewer closes)"
    )
    ap.add_argument(
        "--record",
        type=str,
        default="",
        help="dump per-tick live-loop telemetry to this .npz for offline "
        "autopsy; empty = off (column layout in the header comment below)",
    )
    args = ap.parse_args()

    if args.self_test:
        return 0 if run_self_test() else 1
    if not args.wire:
        ap.error("need --wire (or --self-test)")

    m, d, layout = iutm.build_model(disable_hand_collisions=not args.hand_collisions)
    jm = iutm.joint_maps(m)
    tgt = iutm.hold_targets(jm)
    m.opt.timestep = iutm.DT / iutm.SUBSTEPS
    if args.waist_stand != 0.0:
        tgt["waist_yaw_joint"] = (
            args.waist_stand,
            ubt.WAIST[0][4],
            ubt.WAIST[0][5],
        )

    wrist_signs = {
        "left": (
            args.wrist_yaw_sign_l,
            args.wrist_pitch_sign_l,
            args.wrist_roll_sign_l,
        ),
        "right": (
            args.wrist_yaw_sign_r,
            args.wrist_pitch_sign_r,
            args.wrist_roll_sign_r,
        ),
    }
    ctrl_levers = {
        side: getattr(args, f"ctrl_lever_{side}") or args.ctrl_lever for side in ("left", "right")
    }
    ticker = IncrementalTeleopTickerV3(
        position_scale=args.position_scale,
        target_slew_m_s=args.target_slew,
        input_tau=args.input_tau,
        trigger_thresh=args.trigger_thresh,
        stale_s=args.stale_s,
        stale_grip_close=args.stale_grip_close,
        wrist_scale=args.wrist_scale,
        wrist_signs=wrist_signs,
        wrist_axis_order={
            "left": tuple(args.wrist_axis_order_l),
            "right": tuple(args.wrist_axis_order_r),
        },
        wrist_slew_rad_s=args.wrist_slew,
        wrist_pi_guard_rad=args.wrist_pi_guard,
        waist_stand=args.waist_stand,
        full_pose_ik=args.full_pose_ik,
        ctrl_lever=ctrl_levers,
        position_deadband_m=args.position_deadband_mm * 1e-3,
        posture_gain=args.posture_gain,
    )
    # trunk keep-out from the REAL model silhouette (d is still at the
    # neutral pose build_model + mj_forward left it in)
    arms = ticker.arms
    for side, arm in arms.items():
        arm.torso_keepout = iutm.torso_keepout_from_model(m, d, side)
    rx = IncrementalIntentReceiver(args.intent_port)
    rx.start()
    print(
        f"[sim] v3 wire mode: SUB :{args.intent_port}; grip (LPF'd)>"
        f"{args.trigger_thresh} engages, trigger drives the hands, "
        f"scale {args.position_scale}, lpf tau {args.input_tau}s, "
        f"wrist scale {args.wrist_scale} signs L{wrist_signs['left']} R{wrist_signs['right']} "
        f"axes L{tuple(args.wrist_axis_order_l)} R{tuple(args.wrist_axis_order_r)}, "
        f"ctrl lever L{tuple(np.round(ticker.ctrl_levers['left'], 4))} "
        f"R{tuple(np.round(ticker.ctrl_levers['right'], 4))}m, "
        f"deadband {ticker.position_deadband_m * 1e3:.1f}mm, "
        f"posture {ticker.posture_gain:g}, "
        f"waist fixed {args.waist_stand}rad, stale->hands {'closed' if args.stale_grip_close else 'OPEN'}, "
        # which solver + which DoF split produced this run (--full-pose-ik
        # opts into 7-DoF; the shipped default is the 4-DoF split)
        f"arm IK {arms['right'].backend}/{'7-DoF' if arms['right'].full_pose_ik else '4-DoF'}"
    )

    viewer = None
    if not args.no_viewer:
        from mujoco import viewer as mj_viewer  # local alias only

        viewer = mj_viewer.launch_passive(m, d)

    t0 = time.monotonic()
    t_stat = t0
    last_stat = 0.0
    n_ticks = 0
    # --record: one row per tick, flat float64. Column layout (FIXED --
    # the autopsy tools index it positionally):
    #   0 t_wall | 1 fresh | 2 lt | 3 rt (raw trigger = hand input)
    #   4 grip_l | 5 grip_r (LPF'd grip = clutch) | 6 frame_t
    #   7:14  lcp_raw (pos3+quat4) | 14:21  rcp_raw
    #   21:28 lcp_filt | 28:35 rcp_filt
    #   35:42 q_left_cmd | 42:49 q_right_cmd
    #   49:52 wrist_tgt_L(3) | 52:55 wrist_tgt_R(3)
    #   55:58 wrist_delta_ypr_L | 58:61 wrist_delta_ypr_R
    #   61 ik_pos_err_L | 62 ik_pos_err_R
    #   63 hold_L | 64 hold_R | 65 eng_L | 66 eng_R
    #   67 wrist_dropped_L | 68 wrist_dropped_R | 69 waist (const stand)
    #   70:84 qpos arms(14) + waist(1)
    rec_rows = None if not args.record else []
    rec_joints = list(ubt.ARM_NAMES) + ["waist_yaw_joint"]
    next_tick = time.monotonic()
    try:
        while True:
            if args.duration > 0 and time.monotonic() - t0 > args.duration:
                break
            if viewer is not None and not viewer.is_running():
                break
            frame, nerr = rx.snapshot()
            st = ticker.step(frame)
            fresh = st["fresh"]
            lt, rt = st["lt"], st["rt"]
            grip_f = st["grip"]  # clutch telemetry (record cols 4/5)
            hand_f = st["hand"]  # OmniHand drive (v3 polarity: hand)
            ctrl = st["ctrl"]

            for side, arm in ticker.arms.items():
                q = st["q"][side]
                base = 0 if side == "left" else 7
                for i, n in enumerate(ubt.ARM_NAMES[base : base + 7]):
                    tgt[n] = (q[i], *tgt[n][1:])
            # waist: NOT driven by the operator -- v3 holds the stand value
            # (hold_targets set it; only --waist-stand ever overrides)
            # trigger v3: r = hand -- hand OPEN at rest, CLOSED on press; a
            # stale frame forces hand 0 -> OPEN (unless --stale-grip-close)
            apply = iutm.apply_active_hand_ctrl
            apply(
                d,
                layout,
                left_active=grasp_command_from_ratio("left", np.clip(hand_f["l"], 0.0, 1.0)),
                right_active=grasp_command_from_ratio("right", np.clip(hand_f["r"], 0.0, 1.0)),
            )
            iutm.pd_and_step(m, d, jm, tgt)
            if rec_rows is not None:
                rec_rows.append(
                    np.concatenate(
                        [
                            [time.monotonic(), float(fresh), lt, rt, grip_f["l"], grip_f["r"]],
                            [float(frame["t"]) if frame is not None else np.nan],
                            st["ctrl_pre"]["left"],
                            st["ctrl_pre"]["right"],
                            ctrl["left"],
                            ctrl["right"],
                            arms["left"].q,
                            arms["right"].q,
                            st["wrist_tgt"]["left"],
                            st["wrist_tgt"]["right"],
                            st["wrist_delta_ypr"]["left"],
                            st["wrist_delta_ypr"]["right"],
                            [
                                arms["left"].last_ik.pos_err_m,
                                arms["right"].last_ik.pos_err_m,
                            ],
                            [
                                float(arms["left"].holding),
                                float(arms["right"].holding),
                                float(arms["left"].engaged),
                                float(arms["right"].engaged),
                            ],
                            [
                                float(st["wrist_dropped"]["left"]),
                                float(st["wrist_dropped"]["right"]),
                                st["waist"],
                            ],
                            [d.qpos[jm[n][0]] for n in rec_joints],
                        ]
                    )
                )
            if viewer is not None:
                viewer.sync()
            # REAL-TIME PACING (see v2): pace to wall-clock DT, re-anchor
            # if we fell behind
            n_ticks += 1
            next_tick += iutm.DT
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()
            now = time.monotonic()
            if now - last_stat > 2.0:
                last_stat = now
                hz = n_ticks / (now - t_stat)
                hold = "".join(s[0].upper() for s in ("left", "right") if arms[s].holding)
                w = arms["left"].wrist.cmd
                print(
                    f"[sim] {'L' if st['held']['left'] else '-'}"
                    f"{'R' if st['held']['right'] else '-'} "
                    f"ik_err L {arms['left'].last_ik.pos_err_m*1000:5.1f}mm "
                    f"R {arms['right'].last_ik.pos_err_m*1000:5.1f}mm "
                    f"wristL ypr {np.degrees(w[0]):+5.1f},{np.degrees(w[1]):+4.1f},"
                    f"{np.degrees(w[2]):+5.1f}deg "
                    f"loop {hz:4.1f}Hz "
                    f"hold={hold or '-'} "
                    f"err={nerr} stale={int(not fresh)}"
                )
                n_ticks = 0
                t_stat = now
    except KeyboardInterrupt:
        print("[sim] Ctrl-C")
    if rec_rows is not None:
        arr = np.asarray(rec_rows, np.float64) if rec_rows else np.zeros((0, 85))
        np.savez_compressed(args.record, rows=arr, joints=np.array(rec_joints))
        print(f"[sim] recorded {len(rec_rows)} ticks -> {args.record}")
    # NOTE: no explicit viewer.close() -- it segfaults at teardown on this
    # stack; letting the interpreter exit after the window is gone is clean.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
