#!/usr/bin/env python
"""MuJoCo harness for the INCREMENTAL upper-body teleop (clutch IK).

v1 (upper_body_teleop_mujoco.py, Plan D) follows the Pico SMPL body
estimate globally -- which compresses overhead reaches by 40-70 deg (the
operator holds 180 deg, SMPL reports ~113). v2 replaces the arm path:
each Pico controller drives its arm's wrist 6-DoF INCREMENTALLY while
its grip is held (clutch; the LPF'd grip crosses TRIGGER_THRESH), the
OmniHand opens/closes with the trigger, the waist follows the operator's
shoulder-line twist, and head motion control is disabled (hold 0). The
trigger/grip roles were swapped in 2026-10; the constant / flag names
are historical.

Per side, grip held -> the wrist target is
``anchor_rob + scale * (ctrl_pose - anchor_op)`` (position) and
``anchor_rob_R * (anchor_op_inv * R_now)`` (orientation), solved with
one DLS tick per 50 Hz frame (x2_arm_ik.ArmIKSolver, ported anchor
logic from vr_arm_teleop.py v1). Trigger released -> the arm freezes.

Input: the laptop sender's pico_intent wire (SUB-bind), which now also
packs lctrl_pose/rctrl_pose f32[7] ALREADY in the operator torso frame
(fwd=+x, left=+y, up=+z, pelvis origin -- the sender builds it from the
raw SDK joints), trigger f32[2] (hands), grip f32[2] (clutch). The SMPL
stream only feeds the waist (shoulder line vs hip line). Model: X2 + OmniHand
(compose_x2_with_omnihand), floating base pinned (gantry), 20x1 ms
substeps under the 50 Hz commands.

The CHAIN (receiver, clutch arm + guards, pose LPF, torso frame, waist
yaw, keep-out constant, checks) lives in
``gear_sonic/utils/teleop/incremental_chain.py`` and is imported here --
the SAME file the on-robot node deploys verbatim (PC2 czq_examples), so
this harness validates exactly what will run on the robot. What stays
local is the MuJoCo plumbing plus the checks that need the plant: the
model-derived keep-out (asserted equal to the chain's baked constant),
the real-packer wire roundtrip, the grip mapping, and the hal-table
cross-checks (limits / neutral values).

Usage:
    # T1 (laptop/sim host): pico_intent_sender.py --pc2-host 127.0.0.1 \
    #        --intent-port 15573 --hand-source off   (old :5572 path off)
    .venv/bin/python gear_sonic/scripts/incremental_upper_teleop_mujoco.py \
        --wire --intent-port 15573
    .venv/bin/python gear_sonic/scripts/incremental_upper_teleop_mujoco.py --self-test
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[2]
_sdk = Path.home() / "Documents" / "x2_sdk_v100" / "src" / "py_examples" / "py_examples"
sys.path.insert(0, str(_sdk))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import upper_body_teleop as ubt
except ImportError as e:  # pragma: no cover
    sys.exit(f"cannot import upper_body_teleop.py from x2_sdk_v100: {e}")

# inside the recv loop and silently swallowed (recorded pitfall)
from compose_x2_with_omnihand import (  # noqa: E402
    apply_active_hand_ctrl,
    apply_active_hand_rest_pose,
    build_x2_with_omnihand_spec,
)
import mujoco  # noqa: E402

# The shared chain: receiver / clutch arm / LPFs / torso frame / waist /
# guards / pure checks. Deployed verbatim to the robot -- see there for the
# per-block rationale comments (live rounds 3-8.2) that used to live here.
from gear_sonic.utils.teleop.incremental_chain import (  # noqa: E402
    ARM_HI_SOLVER,
    ARM_LO_SOLVER,
    ARM_NEUTRAL,
    DT,
    TORSO_KEEPOUT,
    TRIGGER_THRESH,
    ClutchArm,
    IncrementalIntentReceiver,
    IncrementalTeleopTicker,
    check_clutch_freeze_reanchor,
    check_extended_stability,
    check_ik_convergence,
    check_ik_freeze_guard,
    check_input_lpf,
    check_overhead_clamp,
    check_pose_err_cap,
    check_pose_lpf_convergence,
    check_torso_frame,
    check_torso_keepout,
    check_trigger_hand_grip_clutch,
    check_waist_chain_gating,
    check_waist_rest_and_twist,
    check_wire_decode,
    run_self_test as _chain_run_self_test,
    unpack_message as unpack_message_inline,
)
from gear_sonic.utils.teleop.x2_hand_retarget import (  # noqa: E402
    grasp_command_from_ratio,
)
from gear_sonic.utils.teleop.zmq.zmq_packed_message_decoder import (  # noqa: E402
    unpack_message,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (  # noqa: E402
    pack_pose_message,
)

SUBSTEPS = 20  # 1 kHz physics under 50 Hz zero-order-held position references


# ---------------------------------------------------------------------------
# The chain that used to live here -- IncrementalIntentReceiver, ClutchArm,
# PoseLpf, TorsoFrame, WaistYawChain and the whole per-tick loop body
# (IncrementalTeleopTicker) -- now comes from
# gear_sonic/utils/teleop/incremental_chain.py: imported above and deployed
# verbatim to the robot node, so sim and robot run ONE tick body, not two
# copies. All their live-round rationale comments moved with them.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Model / joint plumbing (same pattern as upper_body_teleop_mujoco.py)
# ---------------------------------------------------------------------------
# trunk + hip bodies whose geometry defines the torso keep-out volume.
# The hips are IN (live round 7b): the offline repro's contacts were all
# wrist-vs-hip_roll (769 of 1439) -- a low sweep across the pelvis hits the
# hip shells before any torso mesh.
_TRUNK_BODIES = (
    "pelvis",
    "waist_yaw_link",
    "waist_pitch_link",
    "torso_link",
    "left_hip_yaw_link",
    "left_hip_roll_link",
    "left_hip_pitch_link",
    "right_hip_yaw_link",
    "right_hip_roll_link",
    "right_hip_pitch_link",
)


def torso_keepout_from_model(
    m, d, side: str, margin_m: float = 0.05
) -> tuple[float, float, float, float, float, float, float]:
    """(cx, cy, a_ax, b_front, b_back, z_lo, z_hi) of the trunk keep-out
    ELLIPTICAL cylinder for `side`, in that arm's solver-chain frame. Call
    at start-up with the model at the neutral pose (build_model + mj_forward).

    The silhouette comes from MESH VERTICES, not geom_rbound: the torso
    mesh's bounding sphere reads 0.33 m wide when the real trunk is 0.15 m
    laterally. Per-axis extents, and the fore-aft (x) half-axes are
    ASYMMETRIC: the backpack reaches 0.196 m behind the axis while the
    chest is only 0.100 m in front (the hips 0.147). A symmetric axis
    would apply the backpack's 0.20 m to the FRONT as well and project
    every normal "hand across the body 15-20 cm ahead" trajectory -- the
    live round-7 A/B showed exactly that regression (more shaking than no
    keep-out at all). a_ax is the lateral (y) half-axis. Chain-frame
    alignment: subtract the (constant, translation-only) offset between
    the mj wrist body and the solver-chain wrist at the same neutral pose.
    """
    pts = []
    for gid in range(m.ngeom):
        bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[gid])
        if bname not in _TRUNK_BODIES:
            continue
        bid = m.geom_bodyid[gid]
        if m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            did = m.geom_dataid[gid]
            vs = m.mesh_vert[m.mesh_vertadr[did] : m.mesh_vertadr[did] + m.mesh_vertnum[did]]
            rmat = np.empty(9)
            mujoco.mju_quat2Mat(rmat, m.geom_quat[gid])
            vs = np.asarray(vs) @ rmat.reshape(3, 3).T + m.geom_pos[gid]
            pts.append(d.xpos[bid] + vs @ d.xmat[bid].reshape(3, 3).T)
        else:
            pts.append(d.geom_xpos[gid][None, :])
    pts = np.concatenate(pts)
    cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
    a_ax = float(np.abs(pts[:, 1] - cy).max())
    b_front = float((pts[:, 0] - cx).max())
    b_back = float((cx - pts[:, 0]).max())
    z_lo, z_hi = float(pts[:, 2].min()), float(pts[:, 2].max())
    # world -> this arm's chain frame
    arm = ClutchArm(side)
    chain_w = arm.solver.fk(arm.q)[0]
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_wrist_pitch_joint")
    off = d.xpos[m.jnt_bodyid[jid]] - chain_w
    return (
        cx - float(off[0]),
        cy - float(off[1]),
        a_ax + margin_m,
        b_front + margin_m,
        b_back + margin_m,
        z_lo - float(off[2]) - margin_m,
        z_hi - float(off[2]) + margin_m,
    )


def build_model(disable_hand_collisions: bool = True):
    _spec, m, layout = build_x2_with_omnihand_spec(disable_hand_collisions=disable_hand_collisions)
    d = mujoco.MjData(m)
    hold = {leg[0]: leg[1] for leg in ubt.LEG}
    hold.update({w[0]: w[3] for w in ubt.WAIST})
    hold.update(
        {
            n: ARM_NEUTRAL["left" if n.startswith("left") else "right"][i % 7]
            for i, n in enumerate(ubt.ARM_NAMES)
        }
    )
    for jid in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        adr = m.jnt_qposadr[jid]
        if m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            d.qpos[adr + 2] = 1.05  # hang height, feet clear
        elif name in hold:
            d.qpos[adr] = hold[name]
    # park hand qpos AND actuator setpoints at the canonical rest pose
    # (skipping this leaves fingers on joint-limit edges -> NaN under step)
    apply_active_hand_rest_pose(m, d, layout)
    mujoco.mj_forward(m, d)
    return m, d, layout


def joint_maps(m):
    """name -> (qpos_adr, dof_adr, actuator_id, ctrl_lo, ctrl_hi)."""
    out = {}
    for jid in range(m.njnt):
        if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_{name}")
        if aid < 0:
            continue
        out[name] = (
            m.jnt_qposadr[jid],
            m.jnt_dofadr[jid],
            aid,
            m.actuator_ctrlrange[aid][0],
            m.actuator_ctrlrange[aid][1],
        )
    return out


def hold_targets(jm) -> dict:
    """(name -> (q_des, kp, kd)) for every body joint; arms/head are
    overwritten per tick by the teleop loop."""
    tgt = {
        n: (
            ARM_NEUTRAL["left" if n.startswith("left") else "right"][i % 7],
            ubt.ARM_KP[i],
            ubt.ARM_KD[i],
        )
        for i, n in enumerate(ubt.ARM_NAMES)
    }
    for leg in ubt.LEG:
        tgt[leg[0]] = (leg[1], leg[2], leg[3])
    for w in ubt.WAIST:
        tgt[w[0]] = (w[3], w[4], w[5])
    for h in ubt.HEAD:
        tgt[h[0]] = (h[3], h[4], h[5])
    return tgt


def find_root(m):
    for jid in range(m.njnt):
        if m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            return m.jnt_qposadr[jid], m.jnt_dofadr[jid]
    raise RuntimeError("composed model has no free joint (root)")


def pd_and_step(m, d, jm, tdict):
    # Torque servo emulation (live round 8.2): hold the 50 Hz position
    # reference (zero-order), but refresh tau -- including the D term -- on
    # every 1 kHz physics substep. The real X2 servo runs its position loop
    # at the bus rate against the same 50 Hz reference; the previous
    # implementation held the full torque for the whole 20 ms tick, which
    # makes kd*qd STALE feedback. Live recording showed the wrist joints
    # limit-cycling on exactly that (wrist_yaw ringing at ~90 deg/tick,
    # persisting long after the trigger was released; replaying the recorded
    # command trace through both variants reproduced it only with held
    # torque -- fresh tau dropped every joint to <8 deg/tick).
    root_adr, root_dof = find_root(m)
    root0 = d.qpos[root_adr : root_adr + 7].copy()
    # flatten once per tick; the 1 kHz servo loop below is vectorized over
    # all joints (the scalar-per-joint version cost 4 ms of the 20 ms tick)
    adr = np.fromiter((jm[n][0] for n in tdict), np.int32)
    dof = np.fromiter((jm[n][1] for n in tdict), np.int32)
    aid = np.fromiter((jm[n][2] for n in tdict), np.int32)
    lo = np.fromiter((jm[n][3] for n in tdict), np.float64)
    hi = np.fromiter((jm[n][4] for n in tdict), np.float64)
    q_des = np.fromiter((v[0] for v in tdict.values()), np.float64)
    kp = np.fromiter((v[1] for v in tdict.values()), np.float64)
    kd = np.fromiter((v[2] for v in tdict.values()), np.float64)
    for _ in range(SUBSTEPS):
        tau = kp * (q_des - d.qpos[adr]) - kd * d.qvel[dof]
        d.ctrl[aid] = np.clip(tau, lo, hi)
        mujoco.mj_step(m, d)
    # gantry: hard-pin the floating base every frame
    d.qpos[root_adr : root_adr + 7] = root0
    d.qvel[root_dof : root_dof + 6] = 0.0


# ---------------------------------------------------------------------------
# Self-test checks (pytest mirrors these 1:1)
# ---------------------------------------------------------------------------
def check_wire_roundtrip() -> None:
    """The SENDER's packer -> BOTH decoders must agree on every field: the
    gear_sonic decoder (this harness's own path) and the chain module's
    flat inline copy (the ROBOT NODE's path -- it ships standalone, no
    gear_sonic tree). A wire-format change must fail here, not at :5573
    on the robot."""
    data = {
        "smpl_joints": np.random.RandomState(0).rand(72).astype(np.float32),
        "engaged": np.array([1.0], np.float32),
        "lctrl_pose": np.array([0.1, -0.2, 0.3, 0.5, 0.5, 0.5, 0.5], np.float32),
        "rctrl_pose": np.array([-0.1, 0.2, -0.3, 0.7071, 0.0, 0.7071, 0.0], np.float32),
        "trigger": np.array([0.75, 0.0], np.float32),
        "grip": np.array([0.0, 1.0], np.float32),
        "frame_index": np.array([42], np.int64),
    }
    msg = pack_pose_message(data, topic="pico_intent", version=4)
    f = unpack_message(msg, "pico_intent").fields
    g = unpack_message_inline(msg, "pico_intent")
    for k, v in data.items():
        assert k in f, f"field {k} lost on the wire"
        np.testing.assert_array_equal(f[k].reshape(v.shape), v, err_msg=f"field {k} corrupted")
        assert k in g, f"field {k} lost in the robot node's inline decoder"
        np.testing.assert_array_equal(
            g[k].reshape(v.shape), f[k].reshape(v.shape),
            err_msg=f"field {k}: gear_sonic and inline decoders disagree")



def check_grip_mapping() -> None:
    """v2 hand polarity: ratio = 1 - hand. The hand signal is now the LPF'd
    TRIGGER (2026-10 swap); the ratio math itself is unchanged."""
    for side in ("left", "right"):
        open_q = grasp_command_from_ratio(side, 0.0)
        closed_q = grasp_command_from_ratio(side, 1.0)

        def r(g: float) -> float:
            return float(np.clip(1.0 - np.clip(g, 0.0, 1.0), 0.0, 1.0))

        np.testing.assert_allclose(
            grasp_command_from_ratio(side, r(1.0)), open_q, atol=1e-12
        )  # grip 1 = open
        np.testing.assert_allclose(
            grasp_command_from_ratio(side, r(0.0)), closed_q, atol=1e-12
        )  # rest/zero = closed
        mid = grasp_command_from_ratio(side, r(0.5))
        assert np.all(mid >= np.minimum(open_q, closed_q) - 1e-9)
        assert np.all(mid <= np.maximum(open_q, closed_q) + 1e-9)


def check_model_and_head_hold() -> None:
    m, d, layout = build_model()
    jm = joint_maps(m)
    names = (
        list(ubt.ARM_NAMES)
        + [leg[0] for leg in ubt.LEG]
        + [w[0] for w in ubt.WAIST]
        + [h[0] for h in ubt.HEAD]
    )
    missing = [n for n in names if n not in jm]
    assert not missing, f"composed model lacks body joints: {missing}"
    assert find_root(m)[0] == 0, "root qposadr moved (compose regression)"
    tgt = hold_targets(jm)
    for _ in range(200):  # 4 s, no input
        pd_and_step(m, d, jm, tgt)
        apply_active_hand_ctrl(
            d,
            layout,
            left_active=grasp_command_from_ratio("left", 1.0),
            right_active=grasp_command_from_ratio("right", 1.0),
        )
    for h in ubt.HEAD:
        assert abs(d.qpos[jm[h[0]][0]]) < math.radians(
            2.0
        ), f"{h[0]} drifted {math.degrees(d.qpos[jm[h[0]][0]]):.2f} deg"




def check_keepout_matches_model() -> None:
    """The robot node cannot build the trunk silhouette (no mujoco there):
    it uses the chain's baked TORSO_KEEPOUT. Re-derive the volume from
    THIS model and require agreement -- if the MJCF/URDF or the solver
    chain frame ever moves, the gate fails HERE instead of the robot
    discovering its keep-out is stale."""
    m, d, _layout = build_model()
    for side in ("left", "right"):
        got = torso_keepout_from_model(m, d, side)
        np.testing.assert_allclose(
            got,
            TORSO_KEEPOUT[side],
            rtol=0,
            atol=1e-9,
            err_msg=f"{side} keep-out drifted from the chain's baked constant",
        )


def check_limits_match_sdk() -> None:
    """The chain clips the solved q with the solver catalog's joint limits;
    the hal node ships its own copy in upper_body_teleop.ARM_LO/HI (and
    the sim publishes the commands with its gains). They must agree
    joint-for-joint -- otherwise the sim gate would validate clipping
    bounds the robot does not use."""
    np.testing.assert_array_equal(
        ARM_LO_SOLVER, ubt.ARM_LO, err_msg="lower arm limits diverge chain vs ubt"
    )
    np.testing.assert_array_equal(
        ARM_HI_SOLVER, ubt.ARM_HI, err_msg="upper arm limits diverge chain vs ubt"
    )


def check_neutral_matches_vr_defaults() -> None:
    """ARM_NEUTRAL is carried inside the chain module (the robot node must
    not need the repo tree); it must stay identical to vr_arm_teleop's
    DEFAULT_*_ARM_NEUTRAL_RAD -- the values the VR path / recorder seed
    with. Drift here silently changes the engage seed + workspace floor."""
    from gear_sonic.utils.teleop.vr_arm_teleop import (
        DEFAULT_LEFT_ARM_NEUTRAL_RAD,
        DEFAULT_RIGHT_ARM_NEUTRAL_RAD,
    )

    np.testing.assert_array_equal(ARM_NEUTRAL["left"], DEFAULT_LEFT_ARM_NEUTRAL_RAD)
    np.testing.assert_array_equal(ARM_NEUTRAL["right"], DEFAULT_RIGHT_ARM_NEUTRAL_RAD)


# The gate = the shared chain's pure checks + the plant-dependent ones that
# live here. Listed EXPLICITLY (not CHAIN_CHECKS + local) because the pytest
# mirror calls iutm.check_<name>() directly and this list doubles as the
# re-export that keeps those names bound in this namespace.
CHECKS = [
    ("wire_decode", check_wire_decode),
    ("wire_roundtrip", check_wire_roundtrip),
    ("ik_convergence", check_ik_convergence),
    ("clutch_freeze_reanchor", check_clutch_freeze_reanchor),
    ("waist_rest_and_twist", check_waist_rest_and_twist),
    ("waist_chain_gating", check_waist_chain_gating),
    ("grip_mapping", check_grip_mapping),
    ("input_lpf", check_input_lpf),
    ("torso_frame", check_torso_frame),
    ("overhead_clamp", check_overhead_clamp),
    ("extended_stability", check_extended_stability),
    ("ik_freeze_guard", check_ik_freeze_guard),
    ("torso_keepout", check_torso_keepout),
    ("keepout_matches_model", check_keepout_matches_model),
    ("pose_lpf_convergence", check_pose_lpf_convergence),
    ("pose_err_cap", check_pose_err_cap),
    ("limits_match_sdk", check_limits_match_sdk),
    ("neutral_matches_vr_defaults", check_neutral_matches_vr_defaults),
    ("model_and_head_hold", check_model_and_head_hold),
    ("trigger_hand_grip_clutch", check_trigger_hand_grip_clutch),
]


def run_self_test() -> bool:
    return _chain_run_self_test(CHECKS)



# ---------------------------------------------------------------------------
# Main wire loop
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--wire", action="store_true", help="consume the laptop sender over ZMQ (else nothing runs)"
    )
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--intent-port", type=int, default=15573)
    ap.add_argument("--stale-s", type=float, default=0.5)
    ap.add_argument("--position-scale", type=float, default=1.0)
    ap.add_argument(
        "--rotation-weight",
        type=float,
        default=0.5,
        help="IK orientation weight; 0 = position-only (try first for jitter)",
    )
    ap.add_argument(
        "--rot-err-clip",
        type=float,
        default=0.08,
        help="saturation cap on the pose-mode angular residual fed to the IK "
        "(rad): the pull is bounded at rotation_weight x cap, so far-out "
        "residuals close at a smooth bounded rate instead of grinding; "
        "below the cap the solver is byte-identical to legacy; 0 = off",
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
        help="target slew limit (m/s) after the workspace clamp; 0 = off "
        "(1.0: noise floor only -- 0.6 clipped a brisk 0.7-scale raise)",
    )
    ap.add_argument("--trigger-thresh", type=float, default=TRIGGER_THRESH)
    ap.add_argument("--waist-clamp", type=float, default=0.6)
    ap.add_argument(
        "--waist-sign",
        type=float,
        default=1.0,
        help="+1 default; flip to -1 if the viewer shows the " "waist turning the wrong way",
    )
    ap.add_argument(
        "--waist-compensate",
        action="store_true",
        help="EXPERIMENTAL: undo waist yaw in the arm targets "
        "(arms hold world pose). Default: arms ride the torso.",
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
        help="dump per-tick live-loop telemetry to this .npz for offline jitter "
        "autopsy (wire poses pre-LPF, filtered poses, commanded q, realized "
        "qpos, raw trigger (hands) / LPF'd grip (clutch), ik errors, "
        "holding flags); empty = off",
    )
    args = ap.parse_args()

    if args.self_test:
        return 0 if run_self_test() else 1
    if not args.wire:
        ap.error("need --wire (or --self-test)")

    m, d, layout = build_model(disable_hand_collisions=not args.hand_collisions)
    jm = joint_maps(m)
    tgt = hold_targets(jm)
    m.opt.timestep = DT / SUBSTEPS

    # The per-tick chain lives in ONE shared object; the robot node holds the
    # same class, so "what the sim validated" is structurally "what runs on
    # the robot" (offline tape replay can drive it with no plant at all).
    ticker = IncrementalTeleopTicker(
        position_scale=args.position_scale,
        rotation_weight=args.rotation_weight,
        target_slew_m_s=args.target_slew,
        rot_err_clip_rad=args.rot_err_clip,
        input_tau=args.input_tau,
        trigger_thresh=args.trigger_thresh,
        stale_s=args.stale_s,
        waist_sign=args.waist_sign,
        waist_clamp=args.waist_clamp,
        waist_compensate=args.waist_compensate,
    )
    # trunk keep-out from the REAL model silhouette (d is still at the
    # neutral pose build_model + mj_forward left it in)
    arms = ticker.arms
    for side, arm in arms.items():
        arm.torso_keepout = torso_keepout_from_model(m, d, side)
    rx = IncrementalIntentReceiver(args.intent_port)
    rx.start()
    print(
        f"[sim] wire mode: SUB :{args.intent_port}; grip (LPF'd)>{args.trigger_thresh} "
        f"engages, trigger drives the hands, scale {args.position_scale}, rot_w {args.rotation_weight}, "
        f"rot_err_clip {args.rot_err_clip}, "
        f"lpf tau {args.input_tau}s, waist sign {args.waist_sign:+.0f}, "
        # which solver produced this run -- a silent backend switch is
        # exactly the "tested A, ran B" failure the ALLOW_MISMATCH gate
        # exists for, so the banner names it
        f"arm IK {arms['right'].backend}"
    )

    viewer = None
    if not args.no_viewer:
        from mujoco import viewer as mj_viewer  # local alias only

        viewer = mj_viewer.launch_passive(m, d)

    t0 = time.monotonic()
    # Rate window for the status Hz -- SEPARATE from t0. It is reset every
    # status print; sharing t0 silently kept resetting the --duration timer
    # so the loop never stopped.
    t_stat = t0
    last_stat = 0.0
    n_ticks = 0
    # --record (live round 8.2): one row per tick, flat float64. Column
    # layout (FIXED -- the autopsy tools in /tmp index it positionally):
    #   0 t_wall | 1 fresh | 2 lt | 3 rt (raw trigger = hand input)
    #   4 grip_l | 5 grip_r (LPF'd grip = clutch) | 6 frame_t
    #   7:14  lcp_raw (pos3+quat4) | 14:21  rcp_raw | 21:28  lcp_filt
    #   28:35 rcp_filt | 35:42 q_left_cmd | 42:49 q_right_cmd
    #   49:53 ik err (pos,rot)x(L,R) | 53:57 hold/engaged L,R
    #   57 waist_cmd | 58:67 torso M | 67:82 qpos arms(14)+waist(1)
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
            # ---- the whole per-tick chain (freshness -> grip + trigger LPF
            # -> torso frame -> pose LPF -> engage/disengage -> waist -> IK).
            # Shared with the robot node; see IncrementalTeleopTicker.
            st = ticker.step(frame)
            fresh = st["fresh"]
            lt, rt = st["lt"], st["rt"]
            grip_f = st["grip"]  # clutch telemetry (record cols 4/5)
            hand_f = st["hand"]  # OmniHand drive (v2 polarity: 1 - hand)
            lcp_pre_lpf = st["ctrl_pre"]["left"]
            rcp_pre_lpf = st["ctrl_pre"]["right"]
            ctrl = st["ctrl"]

            for side, arm in ticker.arms.items():
                q = st["q"][side]
                base = 0 if side == "left" else 7
                for i, n in enumerate(ubt.ARM_NAMES[base : base + 7]):
                    tgt[n] = (q[i], *tgt[n][1:])
            tgt["waist_yaw_joint"] = (st["waist"], ubt.WAIST[0][4], ubt.WAIST[0][5])
            # trigger: squeeze (trigger=1) opens the hand; rest/zero closes it
            apply_active_hand_ctrl(
                d,
                layout,
                left_active=grasp_command_from_ratio("left", 1.0 - hand_f["l"]),
                right_active=grasp_command_from_ratio("right", 1.0 - hand_f["r"]),
            )
            pd_and_step(m, d, jm, tgt)
            if rec_rows is not None:
                rec_rows.append(
                    np.concatenate(
                        [
                            [time.monotonic(), float(fresh), lt, rt, grip_f["l"], grip_f["r"]],
                            [float(frame["t"]) if frame is not None else np.nan],
                            lcp_pre_lpf,
                            rcp_pre_lpf,
                            ctrl["left"],
                            ctrl["right"],
                            arms["left"].q,
                            arms["right"].q,
                            [
                                arms["left"].last_ik.pos_err_m,
                                arms["left"].last_ik.rot_err_rad,
                                arms["right"].last_ik.pos_err_m,
                                arms["right"].last_ik.rot_err_rad,
                            ],
                            [
                                float(arms["left"].holding),
                                float(arms["right"].holding),
                                float(arms["left"].engaged),
                                float(arms["right"].engaged),
                            ],
                            [st["waist"]],
                            st["torso_M"].reshape(-1),
                            [d.qpos[jm[n][0]] for n in rec_joints],
                        ]
                    )
                )
            if viewer is not None:
                viewer.sync()
            # REAL-TIME PACING (live round 4: "举手卡顿"): the viewer branch
            # had NO sleep -- the loop ran as fast as the sim+render allowed
            # (~2-4x real time on this box), so the robot chased the 50 Hz
            # operator targets in fast bursts separated by waits. Pace BOTH
            # branches to wall-clock DT; if we fell behind, re-anchor rather
            # than accumulating debt.
            n_ticks += 1
            next_tick += DT
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
                print(
                    f"[sim] {'L' if st['held']['left'] else '-'}"
                    f"{'R' if st['held']['right'] else '-'} "
                    f"ik_err L {arms['left'].last_ik.pos_err_m*1000:5.1f}mm "
                    f"R {arms['right'].last_ik.pos_err_m*1000:5.1f}mm "
                    f"waist {math.degrees(st['waist']):+5.1f}deg "
                    f"loop {hz:4.1f}Hz "
                    f"hold={hold or '-'} "
                    f"err={nerr} stale={int(not fresh)}"
                )
                n_ticks = 0
                t_stat = now
    except KeyboardInterrupt:
        print("[sim] Ctrl-C")
    if rec_rows is not None:
        arr = np.asarray(rec_rows, np.float64) if rec_rows else np.zeros((0, 82))
        np.savez_compressed(args.record, rows=arr, joints=np.array(rec_joints))
        print(f"[sim] recorded {len(rec_rows)} ticks -> {args.record}")
    # NOTE: no explicit viewer.close() -- it segfaults at teardown on this
    # stack; letting the interpreter exit after the window is gone is clean.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
