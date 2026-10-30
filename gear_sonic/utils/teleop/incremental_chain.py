#!/usr/bin/env python3
"""Shared incremental (clutch-IK) upper-body teleop chain -- pure numpy/scipy.

ONE source of truth for two consumers:

* the simulation gate
  ``gear_sonic/scripts/incremental_upper_teleop_mujoco.py`` (laptop,
  MuJoCo plant), and
* the on-robot node ``incremental_upper_teleop.py`` (PC2 / czq_examples,
  rclpy plant = the real X2 hal),

so that what was validated in sim IS what runs on the robot. This file
imports NO mujoco / rclpy / ROS: the robot node ships it VERBATIM (scp'd
flat into ``czq_examples/czq_examples/`` next to ``x2_arm_ik.py`` /
``x2_arm_fk.py`` / ``upper_body_teleop.py``) -- the try/except import
below picks the flat siblings when the gear_sonic package is absent.
pyzmq is only needed for the live receiver (module-level try/except:
``--self-test`` works without it, per the PC2 pitfall "zmq must be
imported at module level").

Chain (per side), 50 Hz:
    wire ctrl pose (already framed in the operator torso axes by the
    sender) -> grip clutch: LPF'd grip > TRIGGER_THRESH engages (the
    constant / CLI names are historical -- before the 2026-10 role swap
    the raw trigger was the clutch) --
    ``engage()`` anchors the operator pose against the robot wrist FK at
    the current commanded q -- release disengages and the arm HOLDS its
    last q (v2 semantics; v1 slewed back to neutral);
    the raw trigger is LPF'd into ``hand`` -- the OmniHand drive (v2
    polarity: ratio = 1 - hand, rest closed);
    -> incremental target ``anchor_rob + scale*(ctrl - anchor_op)`` (pos)
    and ``anchor_rob_R * (anchor_op_inv * R_now)`` (rot)
    -> torso keep-out (elliptical cylinder, project -> clamp -> project)
    -> workspace clamp -> target slew -> DLS IK (IK_ITERS_PER_TICK iters
    seeded on the previous q, joint-limit clip) -> freeze guard
    (pos-err / non-finite instant latch, jump-guard voted over
    hold_ticks) -> the q handed to the plant (MuJoCo PD / hal command).

The waist follows the operator's shoulder-line twist (sign/deadzone/LPF/
clamp/slew, WaistYawChain); the head is disabled (the sim harness and the
node hold it at the stand pose).

Self-test: every gate the live rounds 4-8.2 paid for is pinned by a check
in ``CHECKS`` (``run_self_test()``); the sim harness re-runs them plus its
own plant-dependent ones (``torso_keepout_from_model`` /
``pack_pose_message`` roundtrip / grip mapping), and pytest mirrors both.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation as sRot

try:
    import zmq  # only needed for the live receiver, not --self-test
except ImportError:  # pragma: no cover
    zmq = None

try:  # repo tree: the gear_sonic package is importable
    from gear_sonic.utils.teleop.solver.arm import ArmIKSolver, IKResult
    from gear_sonic.utils.teleop.solver.arm.x2_arm_fk import chain_fk_full
    from gear_sonic.utils.teleop.solver.arm.x2_arm_ik import (
        X2_ARM_JOINT_LIMITS_RAD,
    )
except ImportError:  # pragma: no cover - flat vendored copy (PC2 czq_examples)
    from x2_arm_fk import chain_fk_full  # type: ignore[no-redef]
    from x2_arm_ik import (  # type: ignore[no-redef]
        X2_ARM_JOINT_LIMITS_RAD,
        ArmIKSolver,
        IKResult,
    )

# The mink backend pulls in mujoco + mink + a QP backend. Until 2026-10-30
# the robot's JetPack image did not carry them; they are now installed on
# PC2 and the node defaults to mink (2026-10-30 operator request), with the
# same shipped task weights as the laptop. The import stays defensive so
# this module -- the one file both trees share -- remains importable on an
# image without them; selecting the backend is what fails, and loudly.
try:  # pragma: no cover - depends on the environment
    from gear_sonic.utils.teleop.solver.arm.x2_arm_mink import MinkArmIK
except ImportError:  # pragma: no cover
    try:  # flat vendored copy (PC2 czq_examples/ -- mink installed 2026-10-30)
        from x2_arm_mink import MinkArmIK  # type: ignore[no-redef]
    except ImportError:
        MinkArmIK = None


# ---------------------------------------------------------------------------
# Arm IK backend selection
# ---------------------------------------------------------------------------
_ARM_SOLVER_BACKENDS = ("mink", "dls")

# ArmIKSolver arguments with no counterpart on MinkArmIK. Dropped rather than
# forwarded, so one ClutchArm call site can serve both backends; anything else
# unexpected still raises, so a typo cannot silently vanish.
#
# uniform_step_clip is the interesting one. Round 5 found that clipping the
# step PER JOINT distorts its direction near the elbow-straight singularity
# (the joints do not saturate at the same time) and knocks the arm into a
# bang-bang; scaling the whole step uniformly fixed that, and it is why the
# legacy solver grew the flag. On the mink backend the step cap is a
# VelocityLimit constraint inside the QP -- a per-joint box, i.e. the shape
# round 5 rejected -- chosen because a constraint lets the QP redistribute
# over the rest of the arm instead of shrinking everything, and because
# 0.30 rad/tick barely binds in normal motion (peak measured 0.167 on a legal
# turn-around). tools/ab_arm_ik_backends.py measures what that costs instead
# of the design arguing it away.
_LEGACY_ONLY_SOLVER_KWARGS = {
    "uniform_step_clip": "superseded by the mink VelocityLimit step cap",
    # The v3 chain uses the backend-agnostic ``posture_gain`` instead:
    # MinkArmIK turns it into a PostureTask cost, ArmIKSolver into the
    # exact-projector null-space gain. The legacy spellings stay dropped.
    "null_space_gain": "use posture_gain; the mink backend has no such kwarg",
}

# Task-cost knobs only the mink backend has. The A/B harness hands the same
# ticker kwargs to both backends, so the dls branch drops them instead of
# raising TypeError -- the DLS weights ride on the gains, and there is no
# per-axis orientation priority there. Default None on the chain means "use
# the solver's own shipped defaults" (MinkArmIK: position_cost 1.5,
# orientation (1, 0.7, 1) since 2026-10-30).
_MINK_ONLY_SOLVER_KWARGS = {"position_cost", "orientation_axis_weights"}


def resolve_arm_backend(backend: str | None = None) -> str:
    """Arm IK backend for this process: argument, then env, then the default.

    ``X2_IK_BACKEND`` exists because the two entry points on the robot side
    (the sim harness and the node) are shipped as-is and their argument
    parsers are compared verbatim between the sim and robot trees -- an env
    var is the switch that does not disturb that.
    """
    requested = backend if backend is not None else os.environ.get("X2_IK_BACKEND")
    resolved = (requested or "mink").strip().lower()
    if resolved not in _ARM_SOLVER_BACKENDS:
        raise ValueError(
            f"unknown arm IK backend {requested!r}; expected one of "
            f"{_ARM_SOLVER_BACKENDS} (argument or X2_IK_BACKEND)"
        )
    return resolved


def make_arm_solver(side: str, backend: str | None = None, **kwargs):
    """Build one arm's IK solver. Returns ``(solver, resolved_backend)``.

    Defaults to the mink backend (a QP with declarative tasks and limits) and
    keeps the hand-written DLS one as an explicit fallback: pass
    ``backend="dls"`` or set ``X2_IK_BACKEND=dls``. There is deliberately no
    silent fallback -- quietly solving with a different backend than the one
    asked for is exactly the "you think you are testing A but you are running
    B" failure the ALLOW_MISMATCH guard exists to prevent.
    """
    resolved = resolve_arm_backend(backend)
    if resolved == "mink":
        if MinkArmIK is None:
            raise RuntimeError(
                "the mink arm IK backend is the default but mink/mujoco are not "
                "importable in this interpreter. Install them with "
                "`pip install -e 'gear_sonic[sim]'`, or run the legacy solver "
                "with X2_IK_BACKEND=dls."
            )
        kwargs = {k: v for k, v in kwargs.items() if k not in _LEGACY_ONLY_SOLVER_KWARGS}
        return MinkArmIK(side, **kwargs), resolved
    kwargs = {k: v for k, v in kwargs.items() if k not in _MINK_ONLY_SOLVER_KWARGS}
    return ArmIKSolver(side, **kwargs), resolved


# ---------------------------------------------------------------------------
# Quaternion codec -- scalar-first (w, x, y, z) is the wire convention.
# scipy >= 1.14 spells this ``scalar_first=True`` on from_quat/as_quat, but
# PC2 (Ubuntu 22.04) ships scipy 1.8.0 where the kwarg does not exist and the
# first call dies with TypeError. The circular shifts below are exactly what
# scalar_first=True does internally, so one code path runs bit-for-bit
# identically on both scipy generations (A/B-verified against HEAD).
# ---------------------------------------------------------------------------
_Q_WXYZ_TO_XYZW = (1, 2, 3, 0)
_Q_XYZW_TO_WXYZ = (3, 0, 1, 2)


def _rot_from_quat_wxyz(q_wxyz) -> sRot:
    """Rotation from a scalar-first (w, x, y, z) quaternion."""
    return sRot.from_quat(np.asarray(q_wxyz, np.float64)[..., _Q_WXYZ_TO_XYZW])


def _rot_to_quat_wxyz(rot: sRot) -> np.ndarray:
    """Scalar-first (w, x, y, z) quaternion of a Rotation."""
    return rot.as_quat()[..., _Q_XYZW_TO_WXYZ]


DT = 0.02  # 50 Hz control
# common.py convention; NOT the 0.6 walking deadman. Historical name: it now
# gates the LPF'd GRIP clutch (the node/CLI still import this symbol).
TRIGGER_THRESH = 0.5
# DLS iterations per 50 Hz tick. 1 (the recorder default) leaves a ~1.5-tick
# position lag under continuous motion (~10 mm at 5 mm/tick); 2 halves it
# (<5 mm) for negligible cost (two 6x6 solves). 3 buys nothing here.
IK_ITERS_PER_TICK = 2
# One-pole LPF time constant (a = dt/(tau+dt)) applied to the raw grips (the
# clutch) and the raw triggers (the hands). Historical name: live round 4
# added it because raw grip noise fluttered the fingers; the 2026-10 role
# swap kept the value and now uses it on both channels.
GRIP_TAU = 0.15

# Per-side arm neutral. Identical to vr_arm_teleop's
# DEFAULT_{LEFT,RIGHT}_ARM_NEUTRAL_RAD (the sim harness asserts equality);
# carried here so the robot node needs no repo-tree import. The wrist of
# this pose is the engage seed and the IK warm start.
#
# 2026-10-27 operator request: elbows bent ~90 deg with the forearms
# pointing FORWARD (the old pose had the arms hanging with a slight bend).
# FK on x2_arm_fk at this pose: forearm direction [1.00, +-0.02, -0.06] in
# the torso frame, wrist at x=0.221 m, y=+-0.278 m, z=+0.053 m relative to
# torso_link -- forearms level, upper arms hanging at the sides.
ARM_NEUTRAL = {
    "left": np.array([0.0, 0.2, 0.0, -1.57, 0.0, 0.0, 0.0]),
    "right": np.array([0.0, -0.2, 0.0, -1.57, 0.0, 0.0, 0.0]),
}

# ---------------------------------------------------------------------------
# Torso keep-out (live round 7: sweeping one hand toward the other drove the
# wrist INTO the trunk/hip shells -- PD against contact forces = shaking).
# (cx, cy, a_ax, b_front, b_back, z_lo, z_hi) of the elliptical cylinder in
# THIS arm's solver-chain frame; see ClutchArm._project_out_of_trunk.
#
# These are torso_keepout_from_model() outputs (margin 0.05 m) evaluated on
# the sim model at the neutral pose -- MESH-vertex silhouette of the trunk +
# hip bodies, chain-frame aligned. The sim harness RE-DERIVES them from the
# model every run and asserts equality (check_keepout_matches_model), so
# the plant can never silently drift from the robot node's constant.
# The alignment subtracts the wrist offset between the MJCF body and the
# solver chain AT THE NEUTRAL POSE, so these move whenever the chain's
# geometry OR the neutral pose does: the right entry was re-derived after
# x2_arm_fk's right wrist offsets were corrected to the URDF (1.15 mm), and
# BOTH entries were re-derived for the 2026-10-27 bent-elbow neutral
# (cx/cy moved ~8-9 mm, z_lo/z_hi ~10-13 mm).
# ---------------------------------------------------------------------------
TORSO_KEEPOUT = {
    "left": (
        0.011819798758394255,
        -0.001348868136789066,
        0.2542544365041611,
        0.18810346448531395,
        0.2553776834499313,
        -0.5841687380140363,
        0.41540230711695675,
    ),
    "right": (
        0.012512297579944319,
        -0.001521006732450297,
        0.2542544365041611,
        0.18810346448531395,
        0.2553776834499313,
        -0.5992172197306809,
        0.4003538254003122,
    ),
}

# Arm joint limits for the post-solve clip: the solver's own catalog. The
# sim harness asserts these equal the hal tables in upper_body_teleop
# (ubt.ARM_LO/HI) -- same values, this is the dependency-free copy.
_ARM_LIMITS = np.asarray(X2_ARM_JOINT_LIMITS_RAD, np.float64)
ARM_LO_SOLVER = _ARM_LIMITS[:, 0]
ARM_HI_SOLVER = _ARM_LIMITS[:, 1]


# ---------------------------------------------------------------------------
# Packed-binary wire decoder (inline minimal copy of the SONIC
# zmq_packed_message_decoder.py -- kept here so this module is standalone
# on the robot; the sim harness additionally pins it against the real
# gear_sonic decoder in check_wire_roundtrip)
# ---------------------------------------------------------------------------

HEADER_SIZE: int = 1280

_DTYPE_TO_NUMPY = {
    "f32": np.dtype("<f4"),
    "f64": np.dtype("<f8"),
    "i32": np.dtype("<i4"),
    "i64": np.dtype("<i8"),
    "u8": np.dtype("<u1"),
    "bool": np.dtype("<u1"),
}


def unpack_message(message: bytes, expected_topic: str = "pico_intent") -> dict:
    """Decode one packed-format ZMQ message into ``{name: ndarray}``."""
    topic_bytes = expected_topic.encode("utf-8")
    if not message.startswith(topic_bytes):
        raise ValueError(f"message does not start with topic {expected_topic!r}")
    body = message[len(topic_bytes) :]
    if len(body) < HEADER_SIZE:
        raise ValueError(f"body too small: {len(body)} < {HEADER_SIZE}")
    header = json.loads(body[:HEADER_SIZE].rstrip(b"\x00").decode("utf-8"))
    payload = body[HEADER_SIZE:]
    fields, cursor = {}, 0
    for meta in header["fields"]:
        name, shape = meta["name"], tuple(int(s) for s in meta["shape"])
        dt = _DTYPE_TO_NUMPY[meta["dtype"]]
        nelem = int(np.prod(shape)) if shape else 1
        nbytes = nelem * dt.itemsize
        if cursor + nbytes > len(payload):
            raise ValueError(f"field {name!r}: payload truncated")
        fields[name] = np.frombuffer(payload, dtype=dt, count=nelem, offset=cursor).reshape(
            shape or (1,)
        )
        cursor += nbytes
    return fields


# ---------------------------------------------------------------------------
# Receiver: like ubt.IntentReceiver but keeps EVERY decoded field
# (ubt's _drain_intent extracts only 4 fixed fields and would drop the
# controller fields).
# ---------------------------------------------------------------------------
class IncrementalIntentReceiver:
    """SUB-bind the intent port; snapshot() returns the newest full frame."""

    def __init__(self, intent_port: int) -> None:
        self._lock = threading.Lock()
        self._frame = None  # dict(field -> ndarray) + "t"
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
                f = unpack_message(raw, "pico_intent")
            except (ValueError, KeyError, json.JSONDecodeError):
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

    def __init__(
        self,
        side: str,
        position_scale: float = 1.0,
        rotation_weight: float = 0.5,
        target_slew_m_s: float = 0.6,
        freeze_pos_err_m: float = 0.12,
        jump_guard_rad: float = 0.22,
        torso_keepout: tuple | None = None,
        rot_err_clip_rad: float = 0.08,
        backend: str | None = None,
        posture_gain: float = 0.0,
        position_cost: float | None = None,
        orientation_axis_weights: tuple | np.ndarray | None = None,
    ) -> None:
        self.side = side
        self.position_scale = float(position_scale)
        # Backend-agnostic null-space posture strength (0 = off). On the
        # 4-DoF v3 path the position task leaves one swivel DOF; a small
        # bias toward the engage posture keeps that swivel from drifting
        # without moving the wrist point. MinkArmIK: PostureTask cost;
        # ArmIKSolver: exact-projector null-space gain per tick.
        self.posture_gain = float(posture_gain)
        # TORSO KEEP-OUT (live round 7: crossing toward the other arm drove
        # the wrist INTO the trunk/hip shells -- PD against contact forces =
        # shaking; the freeze guard only locked an already-colliding
        # command). A target inside the trunk volume is projected onto its
        # surface, so the wrist slides ALONG the body instead of into it.
        # Tuple: (cx, cy, a_ax, b_ax, z_lo, z_hi) -- an elliptical cylinder
        # (a_ax lateral y, b_ax fore-aft x) in THIS arm's solver-chain
        # frame, from torso_keepout_from_model() (sim) / TORSO_KEEPOUT
        # (robot). None = no keep-out.
        self.torso_keepout = torso_keepout
        # FREEZE GUARD thresholds (see step()): a solve with position error
        # beyond freeze_pos_err_m, or one reachable from the current q only
        # by a per-tick joint jump beyond jump_guard_rad, is treated as
        # "not solvable this tick" and the previous command is resent.
        # Calibrated separation: a legit fold-through-shoulder overhead pass
        # peaks at 0.167 rad/tick, while an elbow-branch reconfiguration
        # hammers the solver step cap (0.30-0.60 rad/tick) continuously --
        # guard 0.22 with a 3-tick persistence requirement sits between.
        self.freeze_pos_err_m = float(freeze_pos_err_m)
        self.jump_guard_rad = float(jump_guard_rad)
        self.hold_ticks = 3
        self._hold_votes = 0
        self.holding = False
        # rotation_weight 0 = position-only IK (the live jitter fix: unfiltered
        # controller orientation noise through 7-DoF shakes the whole arm;
        # the wrist joints saturate near their +-0.558 rad pitch range anyway)
        # solver hardening (live round 5: "手臂伸直后不稳定"): uniform dq
        # scaling near the elbow-straight singularity (per-joint clipping
        # distorts the step direction -> bang-bang). A null-space bent-elbow
        # bias was tried here too but its projection through the DAMPED
        # pseudo-inverse leaks into the task error (17 mm) -- not worth it.
        # ROTATION-RESIDUAL SATURATION (live round 8: "增量式位姿控制还是有
        # 抖动"): in POSE mode the operator wrist sweeps far outside the robot
        # wrist joints' range, so a persistent large angular residual is
        # structurally unreachable -- and DLS answers a large residual with a
        # large per-tick step (~linear: 34 deg -> 0.14 rad/tick, 90 deg ->
        # 0.26), which at the wrist limit grinds and stick-slips against the
        # jump guard. The residual is clamped to rot_err_clip_rad (direction
        # preserved) so the pull saturates at rotation_weight * clip -- the
        # arm keeps closing, just at a bounded human-plausible rate. Below
        # the cap the solver is bit-for-bit the legacy one.
        #
        # NOTE the round-8 live retest made this the third shape of the
        # same idea: round-8.0 shipped a fade band + jam/still/closing
        # freeze gate (stateful); it jittered live immediately -- the fade
        # band bang-banged between ~0 and full pull for any residual
        # hovering at its edge (measured 2x dq oscillation), and the gate
        # was inert during all far-out live motion (n >= band top or
        # target moving), so the real grinding (dq up to the 0.6 step cap)
        # survived it. A stateless saturation has neither failure mode.
        # (The round-8.0 PoseLpf left-multiply fix -- see PoseLpf.step --
        # remains in force; that killed the orbiting target, this bounds
        # the far-out pull.)
        # BACKEND (2026-10-15): the solve step is a mink QP by default -- one
        # QP over two RelativeFrameTasks (position, orientation), with the
        # per-tick step cap handed to mink as a VelocityLimit. Joint RANGES
        # stay a post-step clip, deliberately: as a ConfigurationLimit the QP
        # buys saturated rotation with hand position (measured 50.9 mm off the
        # operator's target -- x2_arm_mink.py's module docstring has the run).
        # The DLS solver and the
        # guards that are *not* solver-internal (freeze guard, keep-out,
        # reachable sphere, target slew) are unchanged and still applied here.
        # `uniform_step_clip` is dropped on the mink path (see
        # _LEGACY_ONLY_SOLVER_KWARGS); `self.backend` is what the harness
        # banner prints, so a run always says which solver produced it.
        # position_cost / orientation_axis_weights are mink-only and None
        # means "the solver's shipped defaults" (1.5 and (1, 0.7, 1) since
        # 2026-10-30); the dls branch drops them (see
        # _MINK_ONLY_SOLVER_KWARGS), so a harness can hand the same kwargs
        # to both backends and still A/B them at equal weighting.
        solver_kwargs = dict(
            damping=0.08,
            rotation_weight=rotation_weight,
            uniform_step_clip=True,
            max_rotation_err_rad=rot_err_clip_rad,
            posture_gain=posture_gain,
            q_preferred=ARM_NEUTRAL[side],
        )
        if position_cost is not None:
            solver_kwargs["position_cost"] = position_cost
        if orientation_axis_weights is not None:
            solver_kwargs["orientation_axis_weights"] = orientation_axis_weights
        self.solver, self.backend = make_arm_solver(side, backend=backend, **solver_kwargs)
        self.q = ARM_NEUTRAL[side].copy()
        self.anchor = None  # (op_pos, op_rot_inv, rob_pos, rob_rot)
        self.last_ik = IKResult(0.0, 0.0, 0, 0)
        # WORKSPACE CLAMP (live 2026-10-13: "手臂无法举过头顶" -- the operator's
        # arm span from the anchor (~1.2 m) exceeds the robot's ~0.55 m reach,
        # so an overhead target lands far outside the workspace and the DLS
        # stalls partway up). Clamp the target onto the reachable sphere
        # around the shoulder: direction preserved, magnitude saturated --
        # the wrist still ends up overhead, just not at an impossible radius.
        # Radius choice (rounds 4-5 lesson): the NEUTRAL wrist already sits
        # at ~90% of the sampled max reach, so a fixed small fraction of max
        # (0.85) clamps EVERY pose -- tracking pulled 20 mm radially from the
        # first tick. Use 0.93 of the sampled max, floored at 1.03x the
        # neutral radius so neutral/engage NEVER clamp, while far reaches
        # stay off the elbow-straight boundary (singularity -> bang-bang,
        # now also damped by the solver's uniform step clip).
        _, origins, _ = chain_fk_full(self.solver._chain, self.q)
        self.shoulder = np.asarray(origins[0], np.float64).copy()
        rng = np.random.RandomState(0)
        reach = 0.0
        for _ in range(400):  # joint-space sweep of |wrist - shoulder|
            q = rng.uniform(self.solver._lo, self.solver._hi)
            p, _R = self.solver.fk(q)
            reach = max(reach, float(np.linalg.norm(p - self.shoulder)))
        r_neutral = float(np.linalg.norm(self.solver.fk(self.q)[0] - self.shoulder))
        self.max_reach = max(0.93 * reach, 1.03 * r_neutral)
        # TARGET SLEW (jitter fix, stage 2): whatever survives the input LPF
        # still drives the DLS at up to 0.3 rad/tick per joint. Rate-limiting
        # the target itself bounds the commanded-q steps to human-plausible
        # motion; 1.0 m/s still bounds noise (post-LPF noise is ~mm-scale)
        # without clipping a brisk raise (0.7 scale x ~1.2 m/s > the old 0.6).
        self.target_slew = float("inf") if target_slew_m_s <= 0 else float(target_slew_m_s) * DT
        self._tgt_slewed = None

    def engage(self, ctrl_pose7: np.ndarray) -> bool:
        """(Re-)anchor. Returns False and refuses on a zero placeholder
        frame (no-xrt / SDK-read-failure wire zeros: pos and quat are 0)."""
        c = np.asarray(ctrl_pose7, np.float64).reshape(7)
        if np.linalg.norm(c[:3]) < 1e-6 or np.linalg.norm(c[3:]) < 1e-6:
            return False
        rob_pos, rob_R = self.solver.fk(self.q)
        self.anchor = (
            c[:3].copy(),
            _rot_from_quat_wxyz(c[3:]).inv(),
            rob_pos,
            sRot.from_matrix(rob_R),
        )
        self._tgt_slewed = rob_pos.copy()  # zero delta at engage
        # Null-space posture is re-anchored with the clutch: the arm should
        # hold the configuration the operator engaged from, not a stale one
        # from a previous grab. No-op when posture is disabled.
        if self.posture_gain != 0.0:
            self.solver.set_posture_target(self.q)
        self.solver.reset_rotation_err_state()
        return True

    def disengage(self) -> None:
        self.anchor = None
        self._tgt_slewed = None

    def _project_out_of_trunk(self, tgt_pos: np.ndarray) -> np.ndarray:
        """Push a target inside the trunk keep-out volume out to its surface.

        The volume is an ELLIPTICAL cylinder: (cx, cy, a_ax, b_front,
        b_back, zlo, zhi) with a_ax the lateral (y) half-axis and the
        fore-aft (x) half-axis ASYMMETRIC (the backpack reaches much
        farther behind than the chest does in front), in this arm's
        solver-chain frame. An ellipse, not a circle, because the live
        round-7 repro showed the wrist sweeping through the HIP region
        (wrist-vs-hip_roll contacts, 769 of them) while a circular
        silhouette either misses the hips or (with a matching radius)
        swallows the whole front workspace. The radial scale keeps the
        target's direction: target' = c + (target - c) / sqrt(E) with E
        the ellipse value, which lands exactly on the surface.
        """
        cx, cy, a_ax, b_front, b_back, zlo, zhi = self.torso_keepout
        if not (zlo < tgt_pos[2] < zhi):
            return tgt_pos
        dx = tgt_pos[0] - cx
        dy = tgt_pos[1] - cy
        b_ax = b_front if dx >= 0.0 else b_back
        e = (dx / b_ax) ** 2 + (dy / a_ax) ** 2
        if e >= 1.0:  # already outside
            return tgt_pos
        tgt_pos = tgt_pos.copy()
        if e < 1e-9:  # dead centre: push out to this arm's own side
            tgt_pos[0] = cx
            tgt_pos[1] = cy + (a_ax if self.side == "left" else -a_ax)
            return tgt_pos
        s = 1.0 / math.sqrt(e)
        tgt_pos[0] = cx + dx * s
        tgt_pos[1] = cy + dy * s
        return tgt_pos

    @property
    def engaged(self) -> bool:
        return self.anchor is not None

    def step(
        self,
        held: bool,
        ctrl_pose7: np.ndarray,
        waist_yaw: float = 0.0,
        waist_compensate: bool = False,
    ):
        """One 50 Hz tick. Returns (q7, IKResult); frozen -> previous q."""
        if not held or self.anchor is None:
            self.last_ik = IKResult(0.0, 0.0, 0, 0)
            return self.q.copy(), self.last_ik
        a_op, a_op_inv, a_rob, a_rob_R = self.anchor
        c = np.asarray(ctrl_pose7, np.float64).reshape(7)
        tgt_pos = a_rob + self.position_scale * (c[:3] - a_op)
        tgt_rot = a_rob_R * (a_op_inv * _rot_from_quat_wxyz(c[3:]))
        if waist_compensate and abs(waist_yaw) > 1e-9:
            # undo the torso rotation so the wrist holds its WORLD pose
            rz_inv = sRot.from_euler("z", -waist_yaw)
            tgt_pos = rz_inv.apply(tgt_pos)
            tgt_rot = rz_inv * tgt_rot
        # workspace clamp + torso keep-out (live round 7): the two
        # constraints interfere -- projecting out of the body can push the
        # target past max_reach, and clamping back can push it into the
        # body. Apply them as a sandwich (project -> clamp -> project):
        # the final project has the last word (never hand the solver a
        # target inside the body), and whatever residual the two leave is
        # bounded per tick by the target slew just below.
        if self.torso_keepout is not None:
            tgt_pos = self._project_out_of_trunk(tgt_pos)
        rad = tgt_pos - self.shoulder
        r = float(np.linalg.norm(rad))
        if r > self.max_reach:
            tgt_pos = self.shoulder + rad * (self.max_reach / r)
        if self.torso_keepout is not None:
            tgt_pos = self._project_out_of_trunk(tgt_pos)
        # target slew: bound per-tick target motion (noise floor only)
        if self._tgt_slewed is None:
            self._tgt_slewed = tgt_pos.copy()
        else:
            step_v = tgt_pos - self._tgt_slewed
            n = float(np.linalg.norm(step_v))
            if n > self.target_slew:
                tgt_pos = self._tgt_slewed + step_v * (self.target_slew / n)
            self._tgt_slewed = tgt_pos.copy()
        q_new, ik = self.solver.solve(
            self.q, tgt_pos, _rot_to_quat_wxyz(tgt_rot), max_iters=IK_ITERS_PER_TICK
        )
        off = 0 if self.side == "left" else 7
        q_new = np.clip(q_new, ARM_LO_SOLVER[off : off + 7], ARM_HI_SOLVER[off : off + 7])
        if self._gate_q_new(q_new, ik):
            self.q = q_new
        self.last_ik = ik
        return self.q.copy(), self.last_ik

    def _gate_q_new(self, q_new: np.ndarray, ik) -> bool:
        """FREEZE GUARD (live round 6: crossing the arms shook the robot -- a
        midline-crossing target has no continuous joint path from the
        current elbow branch, and the DLS answers with max-rate joint
        swings; a diverged solve is worse still). A solution is ACCEPTED
        only if it is finite, its position error is sane, and it is
        reachable from the current q without a jump; otherwise the
        previous command is resent unchanged (the robot holds). Diverged /
        non-finite solves latch instantly; a jump must persist hold_ticks
        consecutive ticks (transient spikes during a legitimate
        fold-through-shoulder pass must not freeze the arm mid-motion).
        Resume is automatic: the first subsequent acceptable solve wins.

        Returns True iff q_new was accepted (the caller then adopts it as
        the commanded q). Extracted as a method so the v3 chain (4-DoF
        solve, wrist frozen separately) runs the SAME guard, not a copy.
        """
        nonfinite = not np.isfinite(q_new).all()
        if nonfinite or ik.pos_err_m > self.freeze_pos_err_m:
            self._hold_votes = self.hold_ticks  # instant, latched
        elif np.abs(q_new - self.q).max() > self.jump_guard_rad:
            self._hold_votes += 1
        else:
            self._hold_votes = 0
        self.holding = self._hold_votes >= self.hold_ticks
        return not self.holding


# ---------------------------------------------------------------------------
# Input filter: 1st-order LPF on a 7-DoF wire pose (live jitter fix,
# 2026-10-13: "机器人抖动太严重了" -- unfiltered controller noise straight
# through the 7-DoF IK). Position filtered directly, orientation blended
# in the tangent space of the filtered quaternion (small-angle slerp).
# ---------------------------------------------------------------------------
class PoseLpf:
    def __init__(self, tau_s: float) -> None:
        self.tau = float(tau_s)
        self.pos: np.ndarray | None = None
        self.rot: sRot | None = None

    def reset(self) -> None:
        self.pos = None
        self.rot = None

    def step(self, pose7: np.ndarray, dt: float) -> np.ndarray:
        c = np.asarray(pose7, np.float64).reshape(7)
        if np.linalg.norm(c) < 1e-6:  # zero placeholder: pass through, drop state
            self.reset()
            return c
        p = c[:3]
        r = _rot_from_quat_wxyz(c[3:])
        if self.pos is None or self.tau <= 0.0:
            self.pos, self.rot = p.copy(), r
        else:
            a = dt / (self.tau + dt)
            self.pos = self.pos + a * (p - self.pos)
            # One step of geodesic slerp: rot <- rot * exp(a * log(rot^-1 r)).
            # The correction MUST be composed on the RIGHT: the rotvec of
            # rot^-1 r lives in rot's BODY frame, and only a right-multiply
            # applies it there. Left-multiplying treats a body vector as a
            # world vector -- wrong by O(theta^2), invisible for the small
            # tracking errors of gentle motion, but at a large orientation
            # step (user whipping the wrist: the round-8 pose-mode jitter)
            # the filter never converges: it orbits at a constant ~8 deg per
            # tick forever, so the target never rests and every downstream
            # freeze/limit logic reads "operator still moving".
            self.rot = self.rot * sRot.from_rotvec(a * (self.rot.inv() * r).as_rotvec())
        return np.concatenate([self.pos, _rot_to_quat_wxyz(self.rot)])


# ---------------------------------------------------------------------------
# Torso frame: canonical(wire) -> the operator's anatomical axes.
# ---------------------------------------------------------------------------
class TorsoFrame:
    """Canonical(wire) -> the operator's anatomical axes (fwd=+x, left=+y,
    up=+z -- the robot convention), built from the wire smpl_joints.

    Primary use: the WAIST path -- the twist formula assumes up=+z /
    lateral=+y, which the raw canonical frame is not (it keeps the SMPL
    120-deg base rotation). The controller poses no longer pass through
    here: the sender frames them in an equivalent raw-joint torso frame at
    the source (live round 3, 2026-10-13 -- the canonical chain cannot
    carry controller data, the raw/FK skeletons are not rigidly related).
    Smoothed (tau) and re-orthonormalised every tick.
    """

    def __init__(self, tau_s: float = 0.2) -> None:
        self.tau = float(tau_s)
        self.M = np.eye(3)

    def step(self, j24: np.ndarray, dt: float) -> np.ndarray:
        j24 = np.asarray(j24, np.float64).reshape(24, 3)
        up = j24[12] - j24[0]  # neck - pelvis
        lat = j24[16] - j24[17]  # R->L shoulder line
        n_up, n_lat = np.linalg.norm(up), np.linalg.norm(lat)
        if n_up < 0.05 or n_lat < 0.05:
            return self.M  # degenerate: keep the last frame
        up = up / n_up
        lat = lat - (lat @ up) * up  # orthogonalise, keep R->L sense
        lat /= np.linalg.norm(lat)
        fwd = np.cross(lat, up)  # right-handed: x = y x z
        tgt = np.stack([fwd, lat, up])  # rows: canonical vec -> torso coords
        if self.tau > 0.0:
            a = dt / (self.tau + dt)
            self.M = self.M + a * (tgt - self.M)
            u, _s, vt = np.linalg.svd(self.M)  # re-orthonormalise
            self.M = u @ vt
        else:
            self.M = tgt
        return self.M

    def transform(self, pose7: np.ndarray) -> np.ndarray:
        """Map one canonical 7-DoF pose into torso coordinates."""
        c = np.asarray(pose7, np.float64).reshape(7)
        if np.linalg.norm(c) < 1e-6:
            return c
        out = np.empty(7)
        out[:3] = self.M @ c[:3]
        rot = sRot.from_matrix(self.M) * _rot_from_quat_wxyz(c[3:])
        out[3:] = _rot_to_quat_wxyz(rot)
        return out


# ---------------------------------------------------------------------------
# Waist yaw: SMPL shoulder-line twist rel. hip line
# SMPL-24: 16=L shoulder, 17=R shoulder, 1=L hip, 2=R hip (matches
# ubt._SIDES, verified by its self-test; NOT the swapped draft).
# ---------------------------------------------------------------------------
class WaistYawChain:
    """raw -> 4 deg deadzone -> 1st-order LPF (tau) -> clamp -> slew.

    Filtering is deliberately heavier than the arms (live round 4: the
    waist joint sits mid-torso, so its target noise shakes EVERY joint
    above it -- the SMPL shoulder-line estimate carries a few deg of
    noise that the original tau 0.08 / 3 deg deadzone let straight
    through)."""

    def __init__(
        self,
        sign: float = 1.0,
        deadzone_rad: float = math.radians(4.0),
        tau_s: float = 0.20,
        clamp_rad: float = 0.6,
        slew_rad_s: float = 0.8,
    ) -> None:
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
        s = j24[16] - j24[17]  # shoulder line, R->L (left-pointing)
        h = j24[1] - j24[2]  # hip line, R->L (left-pointing)
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
# The shared per-tick state machine -- THE thing both plants run.
# ---------------------------------------------------------------------------
class IncrementalTeleopTicker:
    """One 50 Hz tick of the incremental teleop, frame-in / commands-out.

    Both plants hold ONE of these: the sim harness feeds it the newest wire
    frame and applies the returned commands to MuJoCo (PD servos); the robot
    node feeds it the same frame and publishes the commands on the hal
    topics. All chain state lives here (arms + anchors, pose LPFs, torso
    frame, waist chain, hand / clutch LPFs), so "what the sim validated" IS "what
    the robot runs" structurally -- and a tape can be driven through it
    OFFLINE with a synthetic clock and compared tick-for-tick against either
    plant (the level-0 alignment gate of czq_test_v2_incremental.md).

    step() ordering is load-bearing (live rounds 3-8.2); it is the exact
    order of the validated sim loop:
        freshness -> grip + trigger LPF -> torso frame -> pose LPF ->
        engage/disengage -> waist -> per-arm IK.

    Returns a dict (per tick):
        fresh:    bool          -- frame newer than stale_s
        lt, rt:   float         -- trigger values (0 when stale; hand input)
        grip_raw / grip:   {'l','r'} -- wire / LPF'd grip (the clutch)
        hand:     {'l','r'}    -- LPF'd trigger (the OmniHand drive)
        held:     {'left','right'}   -- LPF'd grip > thresh (the clutch)
        ctrl_pre, ctrl:    {'left','right'} -- wire pose / post-LPF pose
                           (zeros when stale)
        q:        {'left','right'}   -- the 7 commanded joints per side
        waist:    float         -- commanded waist_yaw
        torso_M:  3x3           -- the operator torso frame in use
    The arms themselves stay reachable as ``self.arms`` (q / last_ik /
    holding / engaged) -- the sim's --record telemetry reads them.
    """

    def __init__(
        self,
        position_scale: float = 1.0,
        rotation_weight: float = 0.5,
        target_slew_m_s: float = 1.0,
        rot_err_clip_rad: float = 0.08,
        input_tau: float = 0.12,
        trigger_thresh: float = TRIGGER_THRESH,
        stale_s: float = 0.5,
        waist_sign: float = 1.0,
        waist_clamp: float = 0.6,
        waist_compensate: bool = False,
        torso_keepout: dict | None = TORSO_KEEPOUT,
        dt: float = DT,
        grip_tau: float = GRIP_TAU,
        backend: str | None = None,
    ) -> None:
        self.dt = float(dt)
        self.trigger_thresh = float(trigger_thresh)
        self.stale_s = float(stale_s)
        self.waist_compensate = bool(waist_compensate)
        self.grip_tau = float(grip_tau)
        self.arms = {
            side: ClutchArm(
                side,
                position_scale,
                rotation_weight,
                target_slew_m_s,
                torso_keepout=None if torso_keepout is None else torso_keepout[side],
                rot_err_clip_rad=rot_err_clip_rad,
                backend=backend,
            )
            for side in ("left", "right")
        }
        self.lpfs = {"left": PoseLpf(input_tau), "right": PoseLpf(input_tau)}
        self.torso = TorsoFrame()
        self.waist = WaistYawChain(sign=waist_sign, clamp_rad=waist_clamp)
        # grip_f: LPF'd wire grip, the clutch decision (recorded in cols 4/5).
        # hand_f: LPF'd wire trigger, the OmniHand drive (ratio = 1 - hand_f).
        self.grip_f = {"l": 0.0, "r": 0.0}
        self.hand_f = {"l": 0.0, "r": 0.0}

    def step(self, frame, now: float | None = None) -> dict:
        now = time.monotonic() if now is None else now
        dt = self.dt
        zero7 = np.zeros(7, np.float64)
        fresh = frame is not None and now - frame["t"] <= self.stale_s
        if fresh:
            # .get for the v4 fields: a legacy v1 wire (no controller layer)
            # reads as zeros = never engage, instead of a KeyError crash.
            lt, rt = (float(x) for x in frame.get("trigger", (0.0, 0.0)))
            lg, rg = (float(x) for x in frame.get("grip", (0.0, 0.0)))
            lcp = np.asarray(frame.get("lctrl_pose", zero7), np.float64)
            rcp = np.asarray(frame.get("rctrl_pose", zero7), np.float64)
            j24 = np.asarray(frame["smpl_joints"], np.float64).reshape(24, 3)
            # Role swap (2026-10): the TRIGGER drives the hands (hand_f) and
            # the GRIP drives the clutch (grip_f). Both use the same one-pole
            # LPF; both plants filter identically -- it is THIS line and no
            # copy. Held is decided on grip_f, which is also what the record
            # logs (cols 4/5), so the clutch decision stays reproducible.
            a = dt / (self.grip_tau + dt)
            self.hand_f["l"] += a * (lt - self.hand_f["l"])
            self.hand_f["r"] += a * (rt - self.hand_f["r"])
            self.grip_f["l"] += a * (lg - self.grip_f["l"])
            self.grip_f["r"] += a * (rg - self.grip_f["r"])
        else:
            lt = rt = lg = rg = 0.0  # stale -> freeze + hands closed
            self.grip_f["l"] = self.grip_f["r"] = 0.0
            self.hand_f["l"] = self.hand_f["r"] = 0.0
            lcp = rcp = zero7
            j24 = None
            self.lpfs["left"].reset()
            self.lpfs["right"].reset()
        if fresh:
            # controller poses arrive PRE-FRAMED in the operator torso
            # axes (sender-side raw-joint torso map) -- only the pose LPF
            # runs here, so engage/step/deltas all see the same smoothed
            # anatomical pose (zero placeholders pass through)
            self.torso.step(j24, dt)
            lcp_pre_lpf = lcp.copy()
            rcp_pre_lpf = rcp.copy()
            lcp = self.lpfs["left"].step(lcp, dt)
            rcp = self.lpfs["right"].step(rcp, dt)
        else:
            lcp_pre_lpf = rcp_pre_lpf = np.zeros(7)
        held = {
            "left": fresh and self.grip_f["l"] > self.trigger_thresh,
            "right": fresh and self.grip_f["r"] > self.trigger_thresh,
        }
        ctrl = {"left": lcp, "right": rcp}
        for side, arm in self.arms.items():
            if held[side] and not arm.engaged:
                arm.engage(ctrl[side])  # refuses zero placeholders
            elif not held[side] and arm.engaged:
                arm.disengage()
        if j24 is not None:
            # waist in the SAME torso frame: the twist formula assumes
            # up=+z / lateral=+y, which the raw canonical frame is not
            self.waist.step(j24 @ self.torso.M.T, dt)
        q = {}
        for side, arm in self.arms.items():
            q[side], _ = arm.step(held[side], ctrl[side], self.waist.cmd, self.waist_compensate)
        return {
            "fresh": fresh,
            "lt": lt,
            "rt": rt,
            "grip_raw": {"l": lg, "r": rg},
            "grip": dict(self.grip_f),
            "hand": dict(self.hand_f),
            "held": held,
            "ctrl_pre": {"left": lcp_pre_lpf, "right": rcp_pre_lpf},
            "ctrl": ctrl,
            "q": q,
            "waist": self.waist.cmd,
            "torso_M": self.torso.M,
        }


# ---------------------------------------------------------------------------
# Self-test checks (pytest mirrors these 1:1; the sim harness re-runs them
# plus its own plant-dependent ones). All pure -- no mujoco, no ROS, no zmq.
# ---------------------------------------------------------------------------
def _pack_test_message(data: dict, topic: str = "pico_intent", version: int = 4) -> bytes:
    """Minimal stand-in for the sender's pack_pose_message (test-only): the
    same [topic][1280-byte JSON header][payload] layout, so the inline
    decoder is exercised even where the gear_sonic package is absent."""
    fields, blobs = [], []
    dtype_names = {
        np.dtype("<f4"): "f32",
        np.dtype("<f8"): "f64",
        np.dtype("<i4"): "i32",
        np.dtype("<i8"): "i64",
        np.dtype("<u1"): "u8",
    }
    for k, v in data.items():
        a = np.asarray(v)
        dt = dtype_names.get(a.dtype)
        if dt is None:
            a = a.astype(np.float32)
            dt = "f32"
        fields.append({"name": k, "dtype": dt, "shape": list(a.shape)})
        blobs.append(a.tobytes())
    header = json.dumps(
        {"v": version, "endian": "le", "count": 1, "fields": fields}, separators=(",", ":")
    ).encode("utf-8")
    assert len(header) <= HEADER_SIZE, "test header overflow"
    return topic.encode("utf-8") + header.ljust(HEADER_SIZE, b"\x00") + b"".join(blobs)


def check_wire_decode() -> None:
    """The inline decoder must parse the real wire layout with all v4
    fields (this is the receiver the robot node runs; the sim harness's
    check_wire_roundtrip pins the sender's packer against the gear_sonic
    decoder, this pins the flat copy), and must reject wrong-topic /
    truncated frames instead of returning garbage."""
    data = {
        "smpl_joints": np.random.RandomState(1).rand(72).astype(np.float32),
        "human_quat": np.array([1.0, 0.0, 0.0, 0.0], np.float32),
        "engaged": np.array([1.0], np.float32),
        "wrist_pr": np.array([0.1, -0.2, 0.3, -0.4], np.float32),
        "trigger": np.array([0.75, 0.0], np.float32),
        "grip": np.array([0.0, 1.0], np.float32),
        "lctrl_pose": np.array([0.1, -0.2, 0.3, 0.5, 0.5, 0.5, 0.5], np.float32),
        "rctrl_pose": np.array([-0.1, 0.2, -0.3, 0.7071, 0.0, 0.7071, 0.0], np.float32),
        "frame_index": np.array([42], np.int64),
    }
    f = unpack_message(_pack_test_message(data), "pico_intent")
    for k, v in data.items():
        assert k in f, f"field {k} lost on the wire"
        np.testing.assert_array_equal(f[k].reshape(v.shape), v, err_msg=f"field {k} corrupted")
    try:
        unpack_message(b"pose" + _pack_test_message(data), "pico_intent")
        raise AssertionError("wrong topic must be rejected")
    except ValueError:
        pass
    try:
        unpack_message(_pack_test_message(data)[:200], "pico_intent")
        raise AssertionError("truncated body must be rejected")
    except ValueError:
        pass


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
    anchor_pose[3] = 1.0  # identity quat (wxyz)
    anchor_pose[:3] = pos0  # engage with zero initial delta
    assert arm.engage(anchor_pose)
    a_rob_R = arm.anchor[3]
    errs, q_errs = [], []
    dth = 0.01  # 0.01 rad/tick x 50 = 29 deg
    for k in range(1, 51):
        q_star = ARM_NEUTRAL["left"].copy()
        q_star[0] += dth * k
        p_star, R_star_m = arm.solver.fk(q_star)
        R_star = sRot.from_matrix(R_star_m)
        # operator pose whose anchor deltas give EXACTLY (p_star, R_star):
        # pos: a_rob + (op - a_op) = p_star ; rot: a_rob_R @ op = R_star
        pose = np.empty(7)
        pose[:3] = pos0 + (p_star - arm.anchor[2])
        pose[3:] = _rot_to_quat_wxyz(a_rob_R.inv() * R_star)
        q, _ = arm.step(True, pose)
        p, _ = arm.solver.fk(q)
        errs.append(float(np.linalg.norm(p_star - p)))
        q_errs.append(float(np.max(np.abs(q - q_star))))
    assert max(errs[-10:]) < 0.005, f"final pos err {max(errs[-10:])*1000:.1f} mm (> 5 mm)"
    assert q_errs[-1] < 0.05, f"final joint err {q_errs[-1]:.3f} rad vs ground truth"


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
    for _ in range(5):  # release -> freeze
        pose = pose.copy()
        pose[0] += 0.05  # big operator motion
        q, _ = arm.step(False, pose)
        np.testing.assert_array_equal(q, q_frozen)
    assert arm.engage(pose), "re-engage refused a valid pose"
    q0 = arm.q.copy()
    q, _ = arm.step(True, pose)  # first tick after re-anchor
    np.testing.assert_allclose(q, q0, atol=1e-9, err_msg="re-anchor jump (delta != 0)")
    assert not arm.engage(np.zeros(7)), "zero placeholder must be refused"


def check_waist_rest_and_twist() -> None:
    j24 = np.zeros((24, 3))
    j24[12] = [0, 0, 0.6]
    j24[1], j24[2] = [0, 0.1, 0.0], [0, -0.1, 0.0]  # hip line along +y
    j24[16], j24[17] = [0, 0.2, 0.5], [0, -0.2, 0.5]  # shoulder parallel
    assert (
        abs(WaistYawChain.raw_from_j24(j24)) < 1e-9
    ), "waist raw must be 0 at rest (regression: swapped lines give pi)"
    th = math.radians(10.0)
    rz = np.array(
        [[math.cos(th), -math.sin(th), 0.0], [math.sin(th), math.cos(th), 0.0], [0.0, 0.0, 1.0]]
    )
    j24[16] = rz @ j24[16]  # twist ONLY the shoulder line
    j24[17] = rz @ j24[17]
    raw = WaistYawChain.raw_from_j24(j24)
    assert abs(abs(raw) - th) < math.radians(
        0.5
    ), f"10 deg twist -> {math.degrees(raw):.2f} deg (want +-10)"


def check_waist_chain_gating() -> None:
    c = WaistYawChain(sign=1.0)
    j24 = np.zeros((24, 3))
    j24[1], j24[2] = [0, 0.1, 0.0], [0, -0.1, 0.0]
    j24[16], j24[17] = [0, 0.2, 0.5], [0, -0.2, 0.5]
    # 2 deg twist: inside the 3 deg deadzone -> stays 0 forever
    th = math.radians(2.0)
    rz = np.array(
        [[math.cos(th), -math.sin(th), 0.0], [math.sin(th), math.cos(th), 0.0], [0.0, 0.0, 1.0]]
    )
    j24[16], j24[17] = rz @ j24[16], rz @ j24[17]
    for _ in range(200):
        c.step(j24, DT)
    assert abs(c.cmd) < 1e-9, "deadzone violated"
    # huge twist saturates at the clamp, never exceeds it
    c2 = WaistYawChain()
    big = np.zeros((24, 3))
    big[1], big[2] = [0, 0.1, 0.0], [0, -0.1, 0.0]
    big[16], big[17] = [0.5, 0.2, 0.5], [0.5, -0.2, 0.5]  # ~68 deg twist
    for _ in range(400):  # 8 s: past LPF and slew
        cmd = c2.step(big, DT)
    assert abs(cmd) <= c2.clamp + 1e-9, "clamp violated"


def check_input_lpf() -> None:
    f = PoseLpf(tau_s=0.06)
    pose = np.array([0.3, -0.1, 0.4, 1.0, 0.0, 0.0, 0.0], np.float64)
    out = None
    for _ in range(200):  # ~4 s: converged on a constant input
        out = f.step(pose, DT)
    np.testing.assert_allclose(out, pose, atol=1e-3, err_msg="LPF must reach steady state")
    # noise attenuation: zero-mean jitter shrinks by ~tau/dt... just require < half
    f2 = PoseLpf(tau_s=0.06)
    rng = np.random.RandomState(0)
    outs = []
    for _ in range(500):
        noisy = pose.copy()
        noisy[:3] += rng.randn(3) * 0.01
        outs.append(f2.step(noisy, DT)[:3])
    assert np.std(outs, axis=0).max() < 0.5 * 0.01, "LPF must attenuate input noise"
    # zero placeholder passes through and drops state
    f3 = PoseLpf(0.06)
    f3.step(pose, DT)
    np.testing.assert_array_equal(f3.step(np.zeros(7), DT), np.zeros(7))
    assert f3.pos is None, "zero placeholder must reset filter state"


def check_torso_frame() -> None:
    """A synthetic person whose canonical joints are the anatomical pose
    rotated by a constant R0: the torso map must undo R0 (a forward reach
    in canonical coords lands on torso +x whatever R0 is)."""
    j24 = np.zeros((24, 3))
    j24[0] = [0.0, 0.0, 0.0]  # pelvis
    j24[12] = [0.0, 0.0, 0.6]  # neck straight up
    j24[16], j24[17] = [0.0, 0.2, 0.5], [0.0, -0.2, 0.5]  # R->L shoulder line
    r0 = sRot.from_euler("xyz", [35.0, -70.0, 15.0], degrees=True)
    rot = r0.as_matrix()
    j24r = j24 @ rot.T  # the whole body in a rotated canonical frame
    tf = TorsoFrame(tau_s=0.0)
    m = tf.step(j24r, DT)
    # forward reach: anatomical +x expressed in the rotated canonical frame
    fwd_canonical = rot @ np.array([1.0, 0.0, 0.0])
    fwd_torso = tf.transform(np.concatenate([fwd_canonical, [1.0, 0, 0, 0]]))[:3]
    np.testing.assert_allclose(
        fwd_torso,
        [1.0, 0.0, 0.0],
        atol=1e-6,
        err_msg="torso map failed to undo the canonical rotation",
    )
    assert abs(np.linalg.det(m) - 1.0) < 1e-9, "M must stay a proper rotation"
    # degenerate joints: keep the last frame instead of NaN-ing
    tf.step(np.zeros((24, 3)), DT)
    np.testing.assert_array_equal(tf.M, m)


def check_overhead_clamp() -> None:
    """Live 2026-10-13: "手臂无法举过头顶" -- the operator's overhead reach
    from the anchor exceeds the robot workspace and the DLS stalled partway.
    With the workspace clamp an impossible-radius target must still raise
    the wrist overhead (direction kept, radius saturated) and stay inside
    the reachable sphere."""
    arm = ClutchArm("left", 1.0, 0.0)  # scale 1: operator delta ~1 m up
    pos0, _ = arm.solver.fk(arm.q)
    anchor = np.zeros(7)
    anchor[3] = 1.0
    anchor[:3] = pos0
    assert arm.engage(anchor)
    overhead = anchor.copy()
    overhead[2] += 1.0  # a full meter up: far outside the ~0.5 m workspace
    for _ in range(250):  # 5 s: past the 0.6 m/s slew
        q, _ = arm.step(True, overhead)
    p, _ = arm.solver.fk(q)
    assert (
        np.linalg.norm(p - arm.shoulder) <= arm.max_reach * 1.001
    ), "wrist left the reachable sphere"
    assert (
        p[2] - arm.shoulder[2] > 0.35
    ), f"wrist not overhead: only {p[2] - arm.shoulder[2]:.2f} m above shoulder"


def check_extended_stability() -> None:
    """Live round 5: "手臂伸直后不稳定" -- target held just outside the
    reachable sphere (operator reaching past the clamp). The commanded q
    must SETTLE on the clamped sphere instead of oscillating: uniform dq
    scaling + null-space bent-elbow bias; the legacy per-joint clip
    bang-bangs here."""
    arm = ClutchArm("left", 1.0, 0.0)
    pos0, _ = arm.solver.fk(arm.q)
    anchor = np.zeros(7)
    anchor[3] = 1.0
    anchor[:3] = pos0
    assert arm.engage(anchor)
    # far past the clamp: up and forward, ~1.5x reach from the shoulder
    out = anchor.copy()
    out[:3] = arm.shoulder + 1.5 * arm.max_reach * np.array([0.55, 0.0, 0.83]) / np.linalg.norm(
        [0.55, 0.0, 0.83]
    )
    for _ in range(400):  # 8 s: past slew, LPF and settling
        q, _ = arm.step(True, out)
    p, _ = arm.solver.fk(q)
    assert np.linalg.norm(p - arm.shoulder) <= arm.max_reach * 1.001, "clamp violated"
    # steady state: no residual oscillation in the commanded q
    qs = []
    for _ in range(50):
        q, _ = arm.step(True, out)
        qs.append(q.copy())
    dq = np.abs(np.diff(np.asarray(qs), axis=0))
    assert dq.max() < 0.005, f"commanded q still moving: {dq.max():.4f} rad/tick"


def check_ik_freeze_guard() -> None:
    """Live round 6: crossing the arms shook the robot -- a solve needing a
    sustained joint JUMP (elbow branch reconfiguration), a diverged solve,
    or a NaN must be REJECTED: the previous command is resent (robot
    holds). The three failure paths are injected straight into the solver
    so the guard logic itself is what's under test; recovery is verified
    by restoring the real solver."""
    arm = ClutchArm("right", 1.0, 0.0, target_slew_m_s=0.0)  # slew off: full jumps
    pose = np.zeros(7)
    pose[3] = 1.0
    pose[:3] = arm.solver.fk(arm.q)[0]
    assert arm.engage(pose)
    orig = arm.solver.solve

    # 1) sustained max-rate joint swings (an elbow flip answers every tick
    #    with solver-cap steps): latches only after the persistence window

    def flip_solve(q_seed, *_a, **_k):
        return np.asarray(q_seed, np.float64) + 0.5, IKResult(0.01, 0.0, 2, 0)

    arm.solver.solve = flip_solve
    for _ in range(arm.hold_ticks - 1):
        arm.step(True, pose)
    assert not arm.holding, "persistence window: must not latch on the first spike"
    q_mid = arm.step(True, pose)[0]
    assert arm.holding, "sustained solver-cap steps must latch"
    q_a = arm.step(True, pose)[0]
    np.testing.assert_array_equal(q_a, q_mid, err_msg="latched: resend the last command")

    # 2) diverged position error: INSTANT latch
    def lost_solve(q_seed, *_a, **_k):
        return np.asarray(q_seed, np.float64), IKResult(0.9, 0.0, 2, 0)

    arm.solver.solve = lost_solve
    q_b = arm.step(True, pose)[0]
    assert arm.holding and np.array_equal(q_b, q_a), "diverged solve must hold instantly"

    # 3) NaN: INSTANT latch, never command garbage
    arm.solver.solve = lambda _s, *_a, **_k: (np.full(7, np.nan), IKResult(9.9, 9.9, 0, 0))
    q_c = arm.step(True, pose)[0]
    assert arm.holding and np.array_equal(q_c, q_a), "NaN solve must hold instantly"

    # 4) solver recovers -> resumes on the next acceptable solve
    arm.solver.solve = orig
    arm.disengage()
    assert arm.engage(pose)
    q, _ = arm.step(True, pose)
    assert not arm.holding, "resume after the solver recovers"
    np.testing.assert_allclose(q, q_a, atol=1e-9)


def check_torso_keepout() -> None:
    """Live round 7: sweeping the right hand across the body toward the
    left arm drove the wrist into the HIP shells (offline repro: 769
    wrist-vs-hip_roll contacts -- PD fighting contact forces = shaking).
    Two things are under test with the elliptical keep-out (the baked
    TORSO_KEEPOUT constant -- the sim harness asserts it equals the value
    derived from the actual model):

    1. while the crossing is still kinematically feasible, the wrist
       slides ALONG the body surface (never inside the ellipse) and the
       freeze guard does NOT trip;
    2. once the target goes past what the arm can do (the sweep hits the
       shoulder_roll joint limit; diag: the wrist reaches (0.27,-0.11)
       and can go no farther), the guard latches and resends the same
       command -- the robot HOLDS STILL instead of fighting -- and when
       the operator pulls the hand back, tracking resumes.

    (The round-7 first cut failed the first half: a circular silhouette
    plus sequential clamp-then-project drove the target past max_reach
    and latched the guard from tick 149, cascading the IK error to 376 mm.)
    """
    keepout = TORSO_KEEPOUT["right"]
    cx, cy, a_ax, b_front, b_back, _zlo, _zhi = keepout
    arm = ClutchArm("right", 1.0, 0.0, torso_keepout=keepout)
    other = ClutchArm("left")
    pos0, _ = arm.solver.fk(arm.q)
    pos_l, _ = other.solver.fk(other.q)
    pose = np.zeros(7)
    pose[3] = 1.0
    pose[:3] = pos0
    assert arm.engage(pose)
    latched_early, latched, froze, min_e = False, False, True, 1e9
    q_hold_ref, p_at_limit, tail_err = None, None, 9.9
    for k in range(700):  # push 0 -> 0.8 (past the limit), then pull back
        s = 0.8 * k / 399.0 if k < 400 else 0.8 - 0.45 * (k - 400) / 299.0
        op = pose.copy()
        op[:3] = pos0 + s * (pos_l - pos0)
        op[2] += 0.15 * s  # a human crossing lifts the hand a little
        q, ik = arm.step(True, op)
        p, _ = arm.solver.fk(q)
        if _zlo < p[2] < _zhi:
            b_ax = b_front if p[0] - cx >= 0.0 else b_back
            e = ((p[0] - cx) / b_ax) ** 2 + ((p[1] - cy) / a_ax) ** 2
            min_e = min(min_e, e)
        if arm.holding:
            latched_early = latched_early or s < 0.40
            if s > 0.55:
                latched = True
                if p_at_limit is None:
                    p_at_limit = p.copy()
                if q_hold_ref is None:
                    q_hold_ref = q.copy()
                else:
                    froze = froze and bool(np.array_equal(q, q_hold_ref))
        if k >= 690:
            tail_err = min(tail_err, ik.pos_err_m)
    fk_end = arm.solver.fk(arm.q)[0]
    assert min_e > 0.98, f"wrist entered the keep-out (E={min_e:.3f})"
    assert not latched_early, "feasible part of the crossing must not freeze"
    assert latched and q_hold_ref is not None, "past the limit the guard must hold"
    assert froze, "while held, the same joint command must be resent"
    assert not arm.holding, "pulling the hand back must resume tracking"
    assert (
        p_at_limit is not None and np.linalg.norm(fk_end - p_at_limit) > 0.05
    ), "pulled back, the wrist must leave the limit pose"
    assert tail_err < 0.05, f"tracking should resume cleanly (err {tail_err * 1000:.0f} mm)"


def check_pose_lpf_convergence() -> None:
    """Live round 8 ROOT CAUSE: PoseLpf.step composed the slerp correction
    on the LEFT (``exp(a*log(rot^-1 r)) * rot``), treating a body-frame
    rotvec as a world vector. The two agree to O(theta^2), so gentle
    tracking looked fine -- but at a large orientation step the filter
    NEVER converged: measured, it orbited at a constant ~8 deg/tick
    forever. The LPFed target therefore never rested, every downstream
    "target stationary" gate stayed disarmed, and DLS kept re-aiming ->
    the pose-mode jitter. The fix is the standard geodesic-slerp
    right-composition: ``rot * exp(a*log(rot^-1 r))``.

    Two assertions pin the right-composition specifically:
    1. tau=0 (pure passthrough) must land EXACTLY on the input even for a
       non-commuting rotation pair (left-multiply is only exact when rot
       and r commute or one is identity);
    2. after a warm-up, a large step must converge MONOTONICALLY (the
       right-composition gives |log(out^-1 r)| = (1-a) * d exactly; the
       bug orbits instead).
    """
    a = sRot.from_euler("xyz", [95.0, 10.0, -40.0], degrees=True)
    b = sRot.from_euler("xyz", [5.0, 80.0, 25.0], degrees=True)
    pa = np.concatenate([[0.3, -0.1, 0.4], _rot_to_quat_wxyz(a)])
    pb = np.concatenate([[0.3, -0.1, 0.4], _rot_to_quat_wxyz(b)])
    # 1) exact passthrough for a non-commuting pair
    f = PoseLpf(tau_s=0.0)
    f.step(pa, DT)
    out = f.step(pb, DT)
    miss = (_rot_from_quat_wxyz(out[3:]).inv() * b).magnitude()
    assert miss < 1e-9, f"passthrough missed a non-commuting input by {np.degrees(miss):.2f} deg"
    # 2) 90-deg step about the live-repro axis: monotone geodesic convergence
    #    (the buggy left-multiply diverges to a 112-deg orbit here, +4 deg/tick)
    ax = np.array([0.3, -0.2, 0.93])
    ax /= np.linalg.norm(ax)
    c = sRot.from_rotvec(ax * np.radians(90.0)) * a
    pc = np.concatenate([[0.3, -0.1, 0.4], _rot_to_quat_wxyz(c)])
    f2 = PoseLpf(tau_s=0.12)
    for _ in range(50):
        f2.step(pa, DT)
    step_rad = (a.inv() * c).magnitude()
    d = []
    for _ in range(300):  # 6 s: tau=0.12 needs ~5*tau to settle
        out = f2.step(pc, DT)
        d.append((_rot_from_quat_wxyz(out[3:]).inv() * c).magnitude())
    d = np.asarray(d)
    assert d[0] > 0.5 * step_rad, "filter jumped the full step: not a filter"
    assert d[0] < 0.95 * step_rad, "first step overshot the input (left-multiply symptom)"
    dd = np.diff(d)
    assert np.all(dd < 1e-12), (
        f"LPF orbited instead of converging: distance to the input rose by "
        f"{np.degrees(dd.max()):.2f} deg in one tick"
    )
    assert d[-1] < 1e-3, f"LPF did not converge: {np.degrees(d[-1]):.2f} deg left after 6 s"


def check_pose_err_cap() -> None:
    """Live round 8.1: the rotation-residual SATURATION
    (``rot_err_clip_rad``). Shipped behavior under test:
    1. below the cap the solver is EXACTLY the legacy one (a trackable
       pose must keep its accuracy, bit for bit);
    2. a driven far-out target keeps following, but every per-tick step
       stays BOUNDED (the round-8 fade/gate shipped here instead
       oscillated ~2x and let the raw solve grind at the 0.6 rad/tick
       step cap -- live retest jittered immediately);
    3. parked far out of range -> the residual holds and the commanded
       q stays smooth (small, non-alternating steps): no grinding, no
       yank; the arm keeps a gentle bounded pull instead of creep;
    4. turning the target back tracks home with no deadlock (the pull
       is never zero, so the arm always follows back).
    """
    ax = np.array([0.3, -0.2, 0.93])
    ax /= np.linalg.norm(ax)

    def feasible_err(clip: float) -> float:
        arm = ClutchArm("right", 1.0, 0.5, rot_err_clip_rad=clip)
        pos0, R0 = arm.solver.fk(arm.q)
        pose0 = np.concatenate([pos0, _rot_to_quat_wxyz(sRot.from_matrix(R0))])
        assert arm.engage(pose0)
        tgt = pose0.copy()
        tgt[3:] = _rot_to_quat_wxyz(
            sRot.from_rotvec(ax * np.radians(3.0)) * _rot_from_quat_wxyz(pose0[3:])
        )
        err = 9.9
        for _ in range(300):
            _, ik = arm.step(True, tgt)
            err = ik.rot_err_rad
        return err

    e_env, e_off = feasible_err(0.08), feasible_err(0.0)
    assert e_env < np.radians(1.0), f"trackable 3-deg twist not met: {np.degrees(e_env):.2f} deg"
    assert abs(e_env - e_off) < 1e-12, "saturation must be an exact no-op below the cap"

    arm = ClutchArm("right", 1.0, 0.5, rot_err_clip_rad=0.08)
    lpf = PoseLpf(0.12)
    pos0, R0 = arm.solver.fk(arm.q)
    pose0 = np.concatenate([pos0, _rot_to_quat_wxyz(sRot.from_matrix(R0))])
    assert arm.engage(lpf.step(pose0, DT))

    def twist(deg: float) -> np.ndarray:
        out = pose0.copy()
        out[3:] = _rot_to_quat_wxyz(
            sRot.from_rotvec(ax * np.radians(deg)) * _rot_from_quat_wxyz(pose0[3:])
        )
        return out

    q_prev = [arm.q.copy()]

    def run(deg_of_k, ticks: int) -> np.ndarray:
        dqs = []
        for k in range(ticks):
            q, _ = arm.step(True, lpf.step(twist(deg_of_k(k)), DT))
            dqs.append(float(np.abs(q - q_prev[0]).max()))
            q_prev[0] = q.copy()
        return np.asarray(dqs)

    dq_drive = run(lambda k: 30.0 * k * DT, 150)  # 3 s ramp to +90 deg
    dq_park = run(lambda k: 90.0, 400)  # parked far out of range
    dq_back = run(lambda k: max(0.0, 90.0 - 30.0 * (k + 1) * DT), 300)  # turn back
    err_end = 9.9
    for _ in range(100):  # settle at home
        _, ik = arm.step(True, lpf.step(pose0, DT))
        err_end = ik.rot_err_rad
    assert dq_drive.max() > 0.005, "driven target: arm must keep following"
    # Saturation bounds the whole closed loop: cap 0.08 x weight 0.5 =
    # 0.04 rad of task error (through J^+ a few hundredths of a rad/tick).
    # Calibrated: new solver measures <=0.009 across drive/park/return;
    # the round-8 fade+gate shipped 0.027 (drive) / 0.067 (park onset) /
    # 0.094 (slamming on the way back) -- all three trip this bound.
    assert dq_drive.max() < 0.02, f"saturation breached: {dq_drive.max():.3f} rad/tick"
    assert (
        dq_park.max() < 0.02
    ), f"parked far out of range must stay gentle, got {dq_park.max():.4f} rad/tick"
    assert np.abs(np.diff(dq_park[-100:])).max() < 0.001, (
        "parked far out of range must not alternate (grinding/yank), "
        f"tick-to-tick delta {np.abs(np.diff(dq_park[-100:])).max():.4f}"
    )
    assert dq_back.max() > 0.005, "turning back must re-arm tracking (deadlock?)"
    assert dq_back.max() < 0.02, f"turning back must stay smooth, got {dq_back.max():.4f} rad/tick"
    assert err_end < np.radians(
        2.0
    ), f"tracking not restored at home: {np.degrees(err_end):.2f} deg residual"


def check_trigger_hand_grip_clutch() -> None:
    """Role contract after the 2026-10 swap: the TRIGGER drives the OmniHand
    (``hand``) and the GRIP is the per-arm clutch (``held``). The constants
    and flags kept their historical names; only the signal sources moved.

    Phase A: triggers pressed, grips released -- neither arm may engage while
    the filtered trigger rises. Phase B: triggers released, grips pressed --
    both arms engage off the filtered grips and the hand channel decays.
    """
    tick = IncrementalTeleopTicker()
    now = 100.0

    def pose_for(side: str) -> np.ndarray:
        pos, _ = tick.arms[side].solver.fk(tick.arms[side].q)
        out = np.zeros(7)
        out[:3] = pos
        out[3] = 1.0  # unit quat (wxyz): engage() refuses zero placeholders
        return out

    frame = {
        "t": now,
        "trigger": np.array([1.0, 1.0]),
        "grip": np.array([0.0, 0.0]),
        "lctrl_pose": pose_for("left"),
        "rctrl_pose": pose_for("right"),
        "smpl_joints": np.zeros(72),
    }
    for _ in range(20):  # 0.4 s: inside stale_s, past the hand LPF rise
        st = tick.step(frame, now)
        now += DT
    assert not any(st["held"].values()), "trigger alone must not engage the clutch"
    assert not any(arm.engaged for arm in tick.arms.values()), "trigger alone engaged an arm"
    assert (
        st["hand"]["l"] > 0.5 and st["hand"]["r"] > 0.5
    ), f"trigger must drive the hand channel, got {st['hand']}"

    frame["t"] = now
    frame["trigger"] = np.zeros(2)
    frame["grip"] = np.ones(2)
    for _ in range(20):  # 0.4 s: grip LPF rises past the 0.5 threshold
        st = tick.step(frame, now)
        now += DT
    assert st["held"]["left"] and st["held"]["right"], "grip must engage the clutch"
    assert (
        tick.arms["left"].engaged and tick.arms["right"].engaged
    ), "grip > threshold must engage both arms"
    assert (
        st["hand"]["l"] < 0.5 and st["hand"]["r"] < 0.5
    ), f"released trigger must decay the hand channel, got {st['hand']}"
    for side, q in st["q"].items():
        assert np.all(np.isfinite(q)), f"{side} arm produced a non-finite command"


CHECKS = [
    ("wire_decode", check_wire_decode),
    ("ik_convergence", check_ik_convergence),
    ("clutch_freeze_reanchor", check_clutch_freeze_reanchor),
    ("waist_rest_and_twist", check_waist_rest_and_twist),
    ("waist_chain_gating", check_waist_chain_gating),
    ("input_lpf", check_input_lpf),
    ("torso_frame", check_torso_frame),
    ("overhead_clamp", check_overhead_clamp),
    ("extended_stability", check_extended_stability),
    ("ik_freeze_guard", check_ik_freeze_guard),
    ("torso_keepout", check_torso_keepout),
    ("pose_lpf_convergence", check_pose_lpf_convergence),
    ("pose_err_cap", check_pose_err_cap),
    ("trigger_hand_grip_clutch", check_trigger_hand_grip_clutch),
]


def run_self_test(checks=None) -> bool:
    """Run this module's pure checks -- or a caller-composed list (the sim
    harness passes CHECKS + its plant-dependent ones). True iff all pass."""
    checks = CHECKS if checks is None else checks
    print("[self-test] incremental upper teleop checks")
    ok = 0
    for name, fn in checks:
        try:
            fn()
            print(f"  PASS {name}")
            ok += 1
        except AssertionError as e:
            print(f"  FAIL {name}: {e}")
    print(f"[self-test] {ok}/{len(checks)} passed")
    return ok == len(checks)


if __name__ == "__main__":
    raise SystemExit(0 if run_self_test() else 1)
