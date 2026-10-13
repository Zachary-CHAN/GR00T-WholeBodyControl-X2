#!/usr/bin/env python
"""MuJoCo harness for the INCREMENTAL upper-body teleop (clutch IK).

v1 (upper_body_teleop_mujoco.py, Plan D) follows the Pico SMPL body
estimate globally -- which compresses overhead reaches by 40-70 deg (the
operator holds 180 deg, SMPL reports ~113). v2 replaces the arm path:
each Pico controller drives its arm's wrist 6-DoF INCREMENTALLY while
its trigger is held (clutch), the OmniHand opens/closes with the grip
button, the waist follows the operator's shoulder-line twist, and head
motion control is disabled (hold 0).

Per side, trigger held -> the wrist target is
``anchor_rob + scale * (ctrl_pose - anchor_op)`` (position) and
``anchor_rob_R * (anchor_op_inv * R_now)`` (orientation), solved with
one DLS tick per 50 Hz frame (x2_arm_ik.ArmIKSolver, ported anchor
logic from vr_arm_teleop.py v1). Trigger released -> the arm freezes.

Input: the laptop sender's pico_intent wire (SUB-bind), which now also
packs lctrl_pose/rctrl_pose f32[7] (pelvis-anchored Z-up, xyz + quat
wxyz), trigger f32[2], grip f32[2]. The SMPL stream only feeds the
waist (shoulder line vs hip line). Model: X2 + OmniHand
(compose_x2_with_omnihand), floating base pinned (gantry), 20x1 ms
substeps under the 50 Hz commands.

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
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
_sdk = (Path.home() / "Documents" / "x2_sdk_v100" / "src" / "py_examples"
        / "py_examples")
sys.path.insert(0, str(_sdk))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import upper_body_teleop as ubt
except ImportError as e:  # pragma: no cover
    sys.exit(f"cannot import upper_body_teleop.py from x2_sdk_v100: {e}")

import mujoco  # noqa: E402
from scipy.spatial.transform import Rotation as sRot  # noqa: E402

import zmq  # noqa: E402  # module-level: a local import got NameError'd
# inside the recv loop and silently swallowed (recorded pitfall)
from compose_x2_with_omnihand import (  # noqa: E402
    apply_active_hand_ctrl,
    apply_active_hand_rest_pose,
    build_x2_with_omnihand_spec,
)
from gear_sonic.utils.teleop.solver.arm import ArmIKSolver, IKResult  # noqa: E402
from gear_sonic.utils.teleop.vr_arm_teleop import (  # noqa: E402
    DEFAULT_LEFT_ARM_NEUTRAL_RAD,
    DEFAULT_RIGHT_ARM_NEUTRAL_RAD,
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

DT = 0.02          # 50 Hz control
SUBSTEPS = 20      # 1 kHz physics under 50 Hz zero-order-held commands
TRIGGER_THRESH = 0.5   # common.py convention; NOT the 0.6 walking deadman
# DLS iterations per 50 Hz tick. 1 (the recorder default) leaves a ~1.5-tick
# position lag under continuous motion (~10 mm at 5 mm/tick); 2 halves it
# (<5 mm) for negligible cost (two 6x6 solves). 3 buys nothing here.
IK_ITERS_PER_TICK = 2
ARM_NEUTRAL = {
    "left": np.asarray(DEFAULT_LEFT_ARM_NEUTRAL_RAD, np.float64),
    "right": np.asarray(DEFAULT_RIGHT_ARM_NEUTRAL_RAD, np.float64),
}


# ---------------------------------------------------------------------------
# Receiver: like ubt.IntentReceiver but keeps EVERY decoded field
# (ubt's _drain_intent extracts only 4 fixed fields and would drop the
# new controller fields).
# ---------------------------------------------------------------------------
class IncrementalIntentReceiver:
    """SUB-bind the intent port; snapshot() returns the newest full frame."""

    def __init__(self, intent_port: int) -> None:
        self._lock = threading.Lock()
        self._frame = None      # dict(field -> ndarray) + "t"
        ctx = zmq.Context.instance()
        s = ctx.socket(zmq.SUB)
        s.setsockopt_string(zmq.SUBSCRIBE, "pico_intent")
        s.setsockopt(zmq.RCVHWM, 4)
        s.bind(f"tcp://*:{intent_port}")
        self._sock = s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._err = 0

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            raw = None
            while True:  # keep only the newest frame (lag = dropped)
                try:
                    nxt = self._sock.recv(zmq.NOBLOCK)
                except Exception:
                    break
                raw = nxt
            if raw is None:
                time.sleep(0.01)
                continue
            try:
                f = unpack_message(raw, "pico_intent").fields
            except (ValueError, KeyError):
                self._err += 1
                continue
            frame = {k: v for k, v in f.items()}
            frame["t"] = time.monotonic()
            with self._lock:
                self._frame = frame

    def snapshot(self):
        with self._lock:
            frame = dict(self._frame) if self._frame is not None else None
        return frame, self._err


# ---------------------------------------------------------------------------
# Clutch arm: anchor + incremental DLS IK (ported from vr_arm_teleop v1)
# ---------------------------------------------------------------------------
class ClutchArm:
    """One arm: trigger-held incremental 6-DoF wrist tracking.

    engage() snapshots dual anchors (operator controller pose + robot
    wrist FK at the CURRENT COMMANDED q -- solver FK, not mj FK, to
    isolate PD droop and keep the anchor feedback loop from ringing).
    step() composes the target and runs one DLS iteration seeded on the
    previous q. Not held / not anchored -> the commanded q is returned
    unchanged (frozen).
    """

    def __init__(self, side: str, position_scale: float = 1.0) -> None:
        self.side = side
        self.position_scale = float(position_scale)
        self.solver = ArmIKSolver(side, damping=0.08, rotation_weight=0.5)
        self.q = ARM_NEUTRAL[side].copy()
        self.anchor = None  # (op_pos, op_rot_inv, rob_pos, rob_rot)
        self.last_ik = IKResult(0.0, 0.0, 0, 0)

    def engage(self, ctrl_pose7: np.ndarray) -> bool:
        """(Re-)anchor. Returns False and refuses on a zero placeholder
        frame (no-xrt / SDK-read-failure wire zeros: pos and quat are 0)."""
        c = np.asarray(ctrl_pose7, np.float64).reshape(7)
        if np.linalg.norm(c[:3]) < 1e-6 or np.linalg.norm(c[3:]) < 1e-6:
            return False
        rob_pos, rob_R = self.solver.fk(self.q)
        self.anchor = (
            c[:3].copy(),
            sRot.from_quat(c[3:], scalar_first=True).inv(),
            rob_pos,
            sRot.from_matrix(rob_R),
        )
        return True

    def disengage(self) -> None:
        self.anchor = None

    @property
    def engaged(self) -> bool:
        return self.anchor is not None

    def step(self, held: bool, ctrl_pose7: np.ndarray,
             waist_yaw: float = 0.0, waist_compensate: bool = False):
        """One 50 Hz tick. Returns (q7, IKResult); frozen -> previous q."""
        if not held or self.anchor is None:
            self.last_ik = IKResult(0.0, 0.0, 0, 0)
            return self.q.copy(), self.last_ik
        a_op, a_op_inv, a_rob, a_rob_R = self.anchor
        c = np.asarray(ctrl_pose7, np.float64).reshape(7)
        tgt_pos = a_rob + self.position_scale * (c[:3] - a_op)
        tgt_rot = a_rob_R * (a_op_inv * sRot.from_quat(c[3:], scalar_first=True))
        if waist_compensate and abs(waist_yaw) > 1e-9:
            # undo the torso rotation so the wrist holds its WORLD pose
            rz_inv = sRot.from_euler("z", -waist_yaw)
            tgt_pos = rz_inv.apply(tgt_pos)
            tgt_rot = rz_inv * tgt_rot
        self.q, self.last_ik = self.solver.solve(
            self.q, tgt_pos, tgt_rot.as_quat(scalar_first=True),
            max_iters=IK_ITERS_PER_TICK)
        off = 0 if self.side == "left" else 7
        self.q = np.clip(self.q, ubt.ARM_LO[off:off + 7], ubt.ARM_HI[off:off + 7])
        return self.q.copy(), self.last_ik


# ---------------------------------------------------------------------------
# Waist yaw: SMPL shoulder-line twist rel. hip line
# SMPL-24: 16=L shoulder, 17=R shoulder, 1=L hip, 2=R hip (matches
# ubt._SIDES, verified by its self-test; NOT the swapped draft).
# ---------------------------------------------------------------------------
class WaistYawChain:
    """raw -> 3 deg deadzone -> 1st-order LPF (tau) -> clamp -> slew."""

    def __init__(self, sign: float = 1.0,
                 deadzone_rad: float = math.radians(3.0),
                 tau_s: float = 0.08, clamp_rad: float = 0.6,
                 slew_rad_s: float = 1.0) -> None:
        self.sign = float(sign)
        self.deadzone = float(deadzone_rad)
        self.tau = float(tau_s)
        self.clamp = float(clamp_rad)
        self.slew = float(slew_rad_s)
        self.filtered = 0.0
        self.cmd = 0.0

    @staticmethod
    def raw_from_j24(j24: np.ndarray) -> float:
        """Twist of the shoulder line relative to the hip line, wrapped
        to (-pi, pi). Zero at rest (shoulder line parallel to hip line)."""
        j24 = np.asarray(j24, np.float64).reshape(24, 3)
        s = j24[16] - j24[17]      # shoulder line, R->L (left-pointing)
        h = j24[1] - j24[2]        # hip line, R->L (left-pointing)
        raw = math.atan2(s[0], s[1]) - math.atan2(h[0], h[1])
        return (raw + math.pi) % (2.0 * math.pi) - math.pi

    def step(self, j24: np.ndarray, dt: float) -> float:
        raw = self.raw_from_j24(j24)
        if abs(raw) < self.deadzone:
            raw = 0.0
        self.filtered += (dt / (self.tau + dt)) * (raw - self.filtered)
        c = float(np.clip(self.sign * self.filtered, -self.clamp, self.clamp))
        self.cmd += float(np.clip(c - self.cmd, -self.slew * dt, self.slew * dt))
        return self.cmd


# ---------------------------------------------------------------------------
# Model / joint plumbing (same pattern as upper_body_teleop_mujoco.py)
# ---------------------------------------------------------------------------
def build_model(disable_hand_collisions: bool = True):
    _spec, m, layout = build_x2_with_omnihand_spec(
        disable_hand_collisions=disable_hand_collisions)
    d = mujoco.MjData(m)
    hold = {l[0]: l[1] for l in ubt.LEG}
    hold.update({w[0]: w[3] for w in ubt.WAIST})
    hold.update({n: ARM_NEUTRAL["left" if n.startswith("left") else "right"][i % 7]
                 for i, n in enumerate(ubt.ARM_NAMES)})
    for jid in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        adr = m.jnt_qposadr[jid]
        if m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            d.qpos[adr + 2] = 1.05            # hang height, feet clear
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
        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                f"motor_{name}")
        if aid < 0:
            continue
        out[name] = (m.jnt_qposadr[jid], m.jnt_dofadr[jid], aid,
                     m.actuator_ctrlrange[aid][0], m.actuator_ctrlrange[aid][1])
    return out


def hold_targets(jm) -> dict:
    """(name -> (q_des, kp, kd)) for every body joint; arms/head are
    overwritten per tick by the teleop loop."""
    tgt = {n: (ARM_NEUTRAL["left" if n.startswith("left") else "right"][i % 7],
               ubt.ARM_KP[i], ubt.ARM_KD[i]) for i, n in enumerate(ubt.ARM_NAMES)}
    for l in ubt.LEG:
        tgt[l[0]] = (l[1], l[2], l[3])
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
    # zero-order-hold the 50 Hz torque command across 1 kHz substeps --
    # a single 20 ms mujoco step with these kp/kd oscillates violently.
    root_adr, root_dof = find_root(m)
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
    d.qvel[root_dof:root_dof + 6] = 0.0


# ---------------------------------------------------------------------------
# Self-test checks (pytest mirrors these 1:1)
# ---------------------------------------------------------------------------
def check_wire_roundtrip() -> None:
    data = {
        "smpl_joints": np.random.RandomState(0).rand(72).astype(np.float32),
        "engaged": np.array([1.0], np.float32),
        "lctrl_pose": np.array([0.1, -0.2, 0.3, 0.5, 0.5, 0.5, 0.5],
                               np.float32),
        "rctrl_pose": np.array([-0.1, 0.2, -0.3, 0.7071, 0.0, 0.7071, 0.0],
                               np.float32),
        "trigger": np.array([0.75, 0.0], np.float32),
        "grip": np.array([0.0, 1.0], np.float32),
        "frame_index": np.array([42], np.int64),
    }
    msg = pack_pose_message(data, topic="pico_intent", version=4)
    f = unpack_message(msg, "pico_intent").fields
    for k, v in data.items():
        assert k in f, f"field {k} lost on the wire"
        np.testing.assert_array_equal(f[k].reshape(v.shape), v,
                                      err_msg=f"field {k} corrupted")


def check_ik_convergence() -> None:
    """Ground-truth reach: a rigid forward raise about the shoulder PITCH
    axis. The exact solution is known by construction -- FK(neutral with
    only shoulder_pitch += dth) -- so the anchor math must reproduce that
    target exactly and the DLS must track it. (Holding a constant wrist
    ORIENTATION while reaching is physically infeasible on this arm --
    wrist_pitch is +-0.558 rad and saturates -- so it is not a fair test.)"""
    arm = ClutchArm("left", position_scale=1.0)
    pos0, _ = arm.solver.fk(arm.q)
    anchor_pose = np.zeros(7)
    anchor_pose[3] = 1.0                       # identity quat (wxyz)
    anchor_pose[:3] = pos0                     # engage with zero initial delta
    assert arm.engage(anchor_pose)
    a_rob_R = arm.anchor[3]
    errs, q_errs = [], []
    dth = 0.01                                 # 0.01 rad/tick x 50 = 29 deg
    for k in range(1, 51):
        q_star = ARM_NEUTRAL["left"].copy()
        q_star[0] += dth * k
        p_star, R_star_m = arm.solver.fk(q_star)
        R_star = sRot.from_matrix(R_star_m)
        # operator pose whose anchor deltas give EXACTLY (p_star, R_star):
        # pos: a_rob + (op - a_op) = p_star ; rot: a_rob_R @ op = R_star
        pose = np.empty(7)
        pose[:3] = pos0 + (p_star - arm.anchor[2])
        pose[3:] = (a_rob_R.inv() * R_star).as_quat(scalar_first=True)
        q, _ = arm.step(True, pose)
        p, _ = arm.solver.fk(q)
        errs.append(float(np.linalg.norm(p_star - p)))
        q_errs.append(float(np.max(np.abs(q - q_star))))
    assert max(errs[-10:]) < 0.005, \
        f"final pos err {max(errs[-10:])*1000:.1f} mm (> 5 mm)"
    assert q_errs[-1] < 0.05, \
        f"final joint err {q_errs[-1]:.3f} rad vs ground truth"


def check_clutch_freeze_reanchor() -> None:
    arm = ClutchArm("right")
    pose = np.zeros(7)
    pose[3] = 1.0
    pose[:3] = arm.solver.fk(arm.q)[0]
    assert arm.engage(pose)
    for _ in range(10):
        pose = pose.copy()
        pose[0] += 0.004
        arm.step(True, pose)
    q_frozen = arm.q.copy()
    for _ in range(5):                         # release -> freeze
        pose = pose.copy()
        pose[0] += 0.05                        # big operator motion
        q, _ = arm.step(False, pose)
        np.testing.assert_array_equal(q, q_frozen)
    assert arm.engage(pose), "re-engage refused a valid pose"
    q0 = arm.q.copy()
    q, _ = arm.step(True, pose)                # first tick after re-anchor
    np.testing.assert_allclose(q, q0, atol=1e-9,
                               err_msg="re-anchor jump (delta != 0)")
    assert not arm.engage(np.zeros(7)), "zero placeholder must be refused"


def check_waist_rest_and_twist() -> None:
    j24 = np.zeros((24, 3))
    j24[12] = [0, 0, 0.6]
    j24[1], j24[2] = [0, 0.1, 0.0], [0, -0.1, 0.0]       # hip line along +y
    j24[16], j24[17] = [0, 0.2, 0.5], [0, -0.2, 0.5]     # shoulder parallel
    assert abs(WaistYawChain.raw_from_j24(j24)) < 1e-9, \
        "waist raw must be 0 at rest (regression: swapped lines give pi)"
    th = math.radians(10.0)
    rz = np.array([[math.cos(th), -math.sin(th), 0.0],
                   [math.sin(th), math.cos(th), 0.0], [0.0, 0.0, 1.0]])
    j24[16] = rz @ j24[16]                     # twist ONLY the shoulder line
    j24[17] = rz @ j24[17]
    raw = WaistYawChain.raw_from_j24(j24)
    assert abs(abs(raw) - th) < math.radians(0.5), \
        f"10 deg twist -> {math.degrees(raw):.2f} deg (want +-10)"


def check_waist_chain_gating() -> None:
    c = WaistYawChain(sign=1.0)
    j24 = np.zeros((24, 3))
    j24[1], j24[2] = [0, 0.1, 0.0], [0, -0.1, 0.0]
    j24[16], j24[17] = [0, 0.2, 0.5], [0, -0.2, 0.5]
    # 2 deg twist: inside the 3 deg deadzone -> stays 0 forever
    th = math.radians(2.0)
    rz = np.array([[math.cos(th), -math.sin(th), 0.0],
                   [math.sin(th), math.cos(th), 0.0], [0.0, 0.0, 1.0]])
    j24[16], j24[17] = rz @ j24[16], rz @ j24[17]
    for _ in range(200):
        c.step(j24, DT)
    assert abs(c.cmd) < 1e-9, "deadzone violated"
    # huge twist saturates at the clamp, never exceeds it
    c2 = WaistYawChain()
    big = np.zeros((24, 3))
    big[1], big[2] = [0, 0.1, 0.0], [0, -0.1, 0.0]
    big[16], big[17] = [0.5, 0.2, 0.5], [0.5, -0.2, 0.5]  # ~68 deg twist
    for _ in range(400):                        # 8 s: past LPF and slew
        cmd = c2.step(big, DT)
    assert abs(cmd) <= c2.clamp + 1e-9, "clamp violated"


def check_grip_mapping() -> None:
    for side in ("left", "right"):
        open_q = grasp_command_from_ratio(side, 0.0)
        closed_q = grasp_command_from_ratio(side, 1.0)
        r = lambda g: float(np.clip(1.0 - np.clip(g, 0.0, 1.0), 0.0, 1.0))
        np.testing.assert_allclose(grasp_command_from_ratio(side, r(1.0)),
                                   open_q, atol=1e-12)   # grip 1 = open
        np.testing.assert_allclose(grasp_command_from_ratio(side, r(0.0)),
                                   closed_q, atol=1e-12)  # rest/zero = closed
        mid = grasp_command_from_ratio(side, r(0.5))
        assert np.all(mid >= np.minimum(open_q, closed_q) - 1e-9)
        assert np.all(mid <= np.maximum(open_q, closed_q) + 1e-9)


def check_model_and_head_hold() -> None:
    m, d, layout = build_model()
    jm = joint_maps(m)
    names = (list(ubt.ARM_NAMES) + [l[0] for l in ubt.LEG]
             + [w[0] for w in ubt.WAIST] + [h[0] for h in ubt.HEAD])
    missing = [n for n in names if n not in jm]
    assert not missing, f"composed model lacks body joints: {missing}"
    assert find_root(m)[0] == 0, "root qposadr moved (compose regression)"
    tgt = hold_targets(jm)
    for _ in range(200):                        # 4 s, no input
        pd_and_step(m, d, jm, tgt)
        apply_active_hand_ctrl(d, layout,
                               left_active=grasp_command_from_ratio("left", 1.0),
                               right_active=grasp_command_from_ratio("right", 1.0))
    for h in ubt.HEAD:
        assert abs(d.qpos[jm[h[0]][0]]) < math.radians(2.0), \
            f"{h[0]} drifted {math.degrees(d.qpos[jm[h[0]][0]]):.2f} deg"


CHECKS = [
    ("wire_roundtrip", check_wire_roundtrip),
    ("ik_convergence", check_ik_convergence),
    ("clutch_freeze_reanchor", check_clutch_freeze_reanchor),
    ("waist_rest_and_twist", check_waist_rest_and_twist),
    ("waist_chain_gating", check_waist_chain_gating),
    ("grip_mapping", check_grip_mapping),
    ("model_and_head_hold", check_model_and_head_hold),
]


def run_self_test() -> bool:
    print("[self-test] incremental upper teleop checks")
    ok = 0
    for name, fn in CHECKS:
        try:
            fn()
            print(f"  PASS {name}")
            ok += 1
        except AssertionError as e:
            print(f"  FAIL {name}: {e}")
    print(f"[self-test] {ok}/{len(CHECKS)} passed")
    return ok == len(CHECKS)


# ---------------------------------------------------------------------------
# Main wire loop
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--wire", action="store_true",
                    help="consume the laptop sender over ZMQ (else nothing runs)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--intent-port", type=int, default=15573)
    ap.add_argument("--stale-s", type=float, default=0.5)
    ap.add_argument("--position-scale", type=float, default=1.0)
    ap.add_argument("--trigger-thresh", type=float, default=TRIGGER_THRESH)
    ap.add_argument("--waist-clamp", type=float, default=0.6)
    ap.add_argument("--waist-sign", type=float, default=1.0,
                    help="+1 default; flip to -1 if the viewer shows the "
                         "waist turning the wrong way")
    ap.add_argument("--waist-compensate", action="store_true",
                    help="EXPERIMENTAL: undo waist yaw in the arm targets "
                         "(arms hold world pose). Default: arms ride the torso.")
    ap.add_argument("--hand-collisions", action="store_true",
                    help="keep OmniHand geom collisions ON (default OFF for "
                         "pure teleop; enable for grasping experiments)")
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="stop after N s (0 = until viewer closes)")
    args = ap.parse_args()

    if args.self_test:
        return 0 if run_self_test() else 1
    if not args.wire:
        ap.error("need --wire (or --self-test)")

    m, d, layout = build_model(
        disable_hand_collisions=not args.hand_collisions)
    jm = joint_maps(m)
    tgt = hold_targets(jm)
    m.opt.timestep = DT / SUBSTEPS

    arms = {"left": ClutchArm("left", args.position_scale),
            "right": ClutchArm("right", args.position_scale)}
    waist = WaistYawChain(sign=args.waist_sign, clamp_rad=args.waist_clamp)
    rx = IncrementalIntentReceiver(args.intent_port)
    rx.start()
    print(f"[sim] wire mode: SUB :{args.intent_port}; trigger>{args.trigger_thresh} "
          f"engages, scale {args.position_scale}, waist sign {args.waist_sign:+.0f}")

    viewer = None
    if not args.no_viewer:
        from mujoco import viewer as mj_viewer  # local alias only
        viewer = mj_viewer.launch_passive(m, d)

    zero7 = np.zeros(7, np.float64)
    t0 = time.monotonic()
    last_stat = 0.0
    try:
        while True:
            if args.duration > 0 and time.monotonic() - t0 > args.duration:
                break
            if viewer is not None and not viewer.is_running():
                break
            frame, nerr = rx.snapshot()
            fresh = (frame is not None
                     and time.monotonic() - frame["t"] <= args.stale_s)
            if fresh:
                lt, rt = (float(x) for x in frame["trigger"])
                lg, rg = (float(x) for x in frame["grip"])
                lcp = np.asarray(frame["lctrl_pose"], np.float64)
                rcp = np.asarray(frame["rctrl_pose"], np.float64)
                j24 = np.asarray(frame["smpl_joints"], np.float64)
            else:
                lt = rt = lg = rg = 0.0     # stale -> freeze + hands closed
                lcp = rcp = zero7
                j24 = None
            held = {"left": fresh and lt > args.trigger_thresh,
                    "right": fresh and rt > args.trigger_thresh}
            ctrl = {"left": lcp, "right": rcp}
            for side, arm in arms.items():
                if held[side] and not arm.engaged:
                    arm.engage(ctrl[side])   # refuses zero placeholders
                elif not held[side] and arm.engaged:
                    arm.disengage()
            if j24 is not None:
                waist.step(j24, DT)

            for side, arm in arms.items():
                q, _ = arm.step(held[side], ctrl[side], waist.cmd,
                                args.waist_compensate)
                base = 0 if side == "left" else 7
                for i, n in enumerate(ubt.ARM_NAMES[base:base + 7]):
                    tgt[n] = (q[i], *tgt[n][1:])
            tgt["waist_yaw_joint"] = (waist.cmd,
                                      ubt.WAIST[0][4], ubt.WAIST[0][5])
            # grip: squeeze (grip=1) opens the hand; rest/zero closes it
            apply_active_hand_ctrl(
                d, layout,
                left_active=grasp_command_from_ratio("left", 1.0 - lg),
                right_active=grasp_command_from_ratio("right", 1.0 - rg))
            pd_and_step(m, d, jm, tgt)
            if viewer is not None:
                viewer.sync()
            else:
                time.sleep(DT)
            now = time.monotonic()
            if now - last_stat > 2.0:
                last_stat = now
                print(f"[sim] {'L' if held['left'] else '-'}"
                      f"{'R' if held['right'] else '-'} "
                      f"ik_err L {arms['left'].last_ik.pos_err_m*1000:5.1f}mm "
                      f"R {arms['right'].last_ik.pos_err_m*1000:5.1f}mm "
                      f"waist {math.degrees(waist.cmd):+5.1f}deg "
                      f"err={nerr} stale={int(not fresh)}")
    except KeyboardInterrupt:
        print("[sim] Ctrl-C")
    # NOTE: no explicit viewer.close() -- it segfaults at teardown on this
    # stack; letting the interpreter exit after the window is gone is clean.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
