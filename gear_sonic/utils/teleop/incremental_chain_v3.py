#!/usr/bin/env python3
"""v3 incremental upper-body teleop chain -- wrist decoupled from the IK.

Same skeleton as v2 (``incremental_chain.py``): trigger clutch, dual
anchoring, torso keep-out sandwich, reach sphere, target slew, freeze
guard, pose LPF. Three deliberate changes (user decisions, 2026-10-15):

1. SMPL channel CLOSED: no ``smpl_joints`` consumption, no TorsoFrame /
   WaistYawChain -- the waist joints hold a fixed stand value.
2. Arm IK = position-only DLS over the shoulder-3 + elbow joints
   (``ArmIKSolver.solve_position_free4``); the three wrist joints are
   LOCKED at the wrist command and ride along in the FK (they move the
   solved point -- the spherical wrist centre -- by at most 0.98 mm, so
   the 4-DoF solve re-runs against the CURRENT wrist command every tick
   to keep its sub-mm residual honest). The wrist itself follows the v1
   recipe: per-axis mapping of the controller quaternion differential,
   anchored at engage:
       delta = a_op_inv * R_now                    (body-frame diff)
       d     = W_a^-1 * delta * W_a                (conjugation into the
                                                    wrist-anchor frame --
                                                    a single-sided product
                                                    is only valid at W_a~I)
       q_wrist_tgt = q_wa + scale * signs * d.as_euler("ZYX")
   The ZYX decomposition matches the wrist chain Rz->Ry->Rx exactly;
   constant grasp offsets cancel under the anchored conjugation, leaving
   only per-axis mirror signs (env/CLI knobs, tape-verified before robot).
3. GRIP INVERTED: ``r = g`` -- the hand is OPEN at rest and CLOSES when
   the grip button is pressed (v2 was the opposite). SAFETY SEMANTIC
   CHANGE: a stale/dropped frame now OPENS the hands (v2 closed them).
   If holding an object through a dropout matters more, pass
   ``stale_grip_close=True`` (--stale-grip-close) to keep the last
   filtered grip instead.

Like the v2 chain this module imports no mujoco / rclpy / ROS: the
robot node ships it flat next to ``incremental_chain.py`` and
``x2_arm_ik.py`` (try/except picks the flat siblings when the
gear_sonic package is absent).
"""

from __future__ import annotations

import time

import numpy as np
from scipy.spatial.transform import Rotation as sRot

try:  # repo tree: the gear_sonic package is importable
    from gear_sonic.utils.teleop.incremental_chain import (
        ARM_HI_SOLVER,
        ARM_LO_SOLVER,
        ARM_NEUTRAL,
        DT,
        GRIP_TAU,
        IK_ITERS_PER_TICK,
        TORSO_KEEPOUT,
        TRIGGER_THRESH,
        ClutchArm,
        PoseLpf,
        _rot_from_quat_wxyz,
        run_self_test as _v2_run_self_test,
    )
    from gear_sonic.utils.teleop.solver.arm import IKResult
    from gear_sonic.utils.teleop.solver.arm.x2_arm_ik import (
        X2_ARM_JOINT_LIMITS_RAD,
    )
except ImportError:  # pragma: no cover - flat vendored copy (PC2 czq_examples)
    from incremental_chain import (
        ARM_HI_SOLVER,
        ARM_LO_SOLVER,
        ARM_NEUTRAL,
        DT,
        GRIP_TAU,
        IK_ITERS_PER_TICK,
        TORSO_KEEPOUT,
        TRIGGER_THRESH,
        ClutchArm,
        PoseLpf,
        _rot_from_quat_wxyz,
        run_self_test as _v2_run_self_test,  # type: ignore[no-redef]
    )
    from x2_arm_ik import (  # type: ignore[no-redef]
        X2_ARM_JOINT_LIMITS_RAD,
        IKResult,
    )

# Default per-axis wrist signs (yaw, pitch, roll) per side. UNVERIFIED on
# the robot: derived from v1's _YAW_SIGN / live ROLL_SIGN=-1 plus the
# left/right mirror. The tape-replay direction assertions must confirm
# them before any robot run; flipping is one flag (WRIST3_*_SIGN env /
# --wrist-{yaw,pitch,roll}-sign CLI).
DEFAULT_WRIST_SIGNS = {
    "left": (-1.0, 1.0, -1.0),
    "right": (1.0, -1.0, 1.0),
}


class WristClutch:
    """Per-axis differential wrist mapping, anchored at engage.

    engage() snapshots the wrist q AND the controller orientation; every
    held tick recomputes the FULL differential from the anchor (not a
    per-tick increment -- no integration drift, and a constant controller
    pose yields a bit-exact constant command). The differential is
    conjugated into the wrist-anchor frame before the ZYX Euler split so
    each Euler axis is the corresponding wrist joint axis.
    """

    def __init__(
        self,
        side: str,
        scale: float = 1.0,
        signs: tuple | None = None,
        slew_rad_s: float = 6.0,
        pi_guard_rad: float = 2.94,
        limits=None,
    ) -> None:
        self.side = side
        self.scale = float(scale)
        self.signs = np.asarray(
            DEFAULT_WRIST_SIGNS[side] if signs is None else signs, np.float64
        ).reshape(3)
        # slew: bounds the commanded step per tick so the freeze guard's
        # jump threshold (0.22 rad) can never fire on wrist motion alone
        self.slew = float(slew_rad_s) * DT
        # near-pi guard: past this the ZYX decomposition is ambiguous and
        # the differential can flip representation tick-to-tick -- drop
        # the tick (the operator re-anchors by releasing/re-pressing long
        # before a legitimate >169-deg-per-anchor motion matters)
        self.pi_guard = float(pi_guard_rad)
        if limits is None:
            off = 0 if side == "left" else 7
            limits = X2_ARM_JOINT_LIMITS_RAD[off + 4 : off + 7]
        self.limits = np.asarray(limits, np.float64).reshape(3, 2)
        self.cmd = ARM_NEUTRAL[side][4:7].copy()
        self._q_wa: np.ndarray | None = None
        self._W_a: sRot | None = None
        self._a_op_inv: sRot | None = None
        self.last_delta_ypr = np.zeros(3)
        self.dropped_ticks = 0

    @property
    def anchored(self) -> bool:
        return self._q_wa is not None

    def engage(self, wrist_q3, ctrl_quat_wxyz) -> bool:
        """Anchor the wrist q + controller orientation. Refuses a zero
        placeholder quaternion (returns False, stays unanchored)."""
        q = np.asarray(ctrl_quat_wxyz, np.float64).reshape(4)
        if np.linalg.norm(q) < 1e-6:
            return False
        self._q_wa = np.asarray(wrist_q3, np.float64).reshape(3).copy()
        self._W_a = sRot.from_euler("ZYX", self._q_wa)  # Rz(y)Ry(p)Rx(r)
        self._a_op_inv = _rot_from_quat_wxyz(q).inv()
        self.cmd = self._q_wa.copy()
        self.last_delta_ypr = np.zeros(3)
        return True

    def disengage(self) -> None:
        self._q_wa = None
        self._W_a = None
        self._a_op_inv = None

    def step(self, ctrl_quat_wxyz) -> np.ndarray:
        """One 50 Hz tick while held. Returns the (slewed, clamped)
        wrist command; unanchored / zero placeholder -> frozen command."""
        if not self.anchored:
            return self.cmd.copy()
        q = np.asarray(ctrl_quat_wxyz, np.float64).reshape(4)
        if np.linalg.norm(q) < 1e-6:
            return self.cmd.copy()
        delta = self._a_op_inv * _rot_from_quat_wxyz(q)
        if delta.magnitude() > self.pi_guard:
            self.dropped_ticks += 1
            self.last_delta_ypr = np.zeros(3)
            return self.cmd.copy()
        d = self._W_a.inv() * delta * self._W_a  # conjugation, NOT one-sided
        self.last_delta_ypr = d.as_euler("ZYX")  # (yaw, pitch, roll)
        tgt = self._q_wa + self.scale * self.signs * self.last_delta_ypr
        tgt = np.clip(tgt, self.limits[:, 0], self.limits[:, 1])
        self.cmd = self.cmd + np.clip(tgt - self.cmd, -self.slew, self.slew)
        return self.cmd.copy()


class ClutchArmV3(ClutchArm):
    """v2's ClutchArm with the wrist carved out of the IK.

    Everything positional (anchoring, keep-out sandwich, reach sphere,
    target slew, freeze guard via ``_gate_q_new``) is inherited; the solve
    becomes ``solve_position_free4`` with the wrist joints pinned to the
    WristClutch command, computed FIRST in the tick so the same tick's IK
    already compensates the wrist_yaw end-effector lever.
    """

    def __init__(self, side: str, position_scale: float = 1.0, *args, **kwargs):
        wrist_scale = kwargs.pop("wrist_scale", 1.0)
        wrist_signs = kwargs.pop("wrist_signs", None)
        wrist_slew = kwargs.pop("wrist_slew_rad_s", 6.0)
        wrist_pi_guard = kwargs.pop("wrist_pi_guard_rad", 2.94)
        # position-only IK regardless of what v2 would default to: the
        # orientation task moved to the wrist mapping
        kwargs.setdefault("rotation_weight", 0.0)
        super().__init__(side, position_scale, *args, **kwargs)
        if wrist_signs is None:
            signs = None
        elif isinstance(wrist_signs, dict):
            signs = wrist_signs.get(side)
        else:
            signs = wrist_signs
        self.wrist = WristClutch(
            side,
            scale=wrist_scale,
            signs=signs,
            slew_rad_s=wrist_slew,
            pi_guard_rad=wrist_pi_guard,
        )

    def engage(self, ctrl_pose7: np.ndarray) -> bool:
        if not super().engage(ctrl_pose7):
            return False
        c = np.asarray(ctrl_pose7, np.float64).reshape(7)
        return self.wrist.engage(self.q[4:7], c[3:])

    def disengage(self) -> None:
        super().disengage()
        self.wrist.disengage()

    def step(self, held: bool, ctrl_pose7: np.ndarray):
        """One 50 Hz tick. Returns (q7, IKResult); frozen -> previous q."""
        if not held or self.anchor is None:
            self.last_ik = IKResult(0.0, 0.0, 0, 0)
            return self.q.copy(), self.last_ik
        c = np.asarray(ctrl_pose7, np.float64).reshape(7)
        # wrist FIRST: the 4-DoF solve below must see this tick's wrist
        # command in its FK, or the wrist_yaw lever reads as arm error
        wrist_q3 = self.wrist.step(c[3:])
        a_op, a_op_inv, a_rob, _a_rob_R = self.anchor
        tgt_pos = a_rob + self.position_scale * (c[:3] - a_op)
        # keep-out sandwich + reach sphere + target slew: identical to the
        # v2 step (copied verbatim -- see incremental_chain.ClutchArm.step
        # for the live-round rationale; the waist compensation is gone
        # with the waist itself)
        if self.torso_keepout is not None:
            tgt_pos = self._project_out_of_trunk(tgt_pos)
        rad = tgt_pos - self.shoulder
        r = float(np.linalg.norm(rad))
        if r > self.max_reach:
            tgt_pos = self.shoulder + rad * (self.max_reach / r)
        if self.torso_keepout is not None:
            tgt_pos = self._project_out_of_trunk(tgt_pos)
        if self._tgt_slewed is None:
            self._tgt_slewed = tgt_pos.copy()
        else:
            step_v = tgt_pos - self._tgt_slewed
            n = float(np.linalg.norm(step_v))
            if n > self.target_slew:
                tgt_pos = self._tgt_slewed + step_v * (self.target_slew / n)
            self._tgt_slewed = tgt_pos.copy()
        q_new, ik = self.solver.solve_position_free4(
            self.q, wrist_q3, tgt_pos, max_iters=IK_ITERS_PER_TICK
        )
        q_new[4:7] = wrist_q3  # the wrist is commanded by WristClutch only
        off = 0 if self.side == "left" else 7
        q_new[:4] = np.clip(q_new[:4], ARM_LO_SOLVER[off : off + 4], ARM_HI_SOLVER[off : off + 4])
        # freeze guard over the FULL q: the wrist part can never trip the
        # jump threshold (its slew bounds the step at 0.12 rad < 0.22),
        # and this keeps the NaN / diverged-solve latch covering the wrist
        if self._gate_q_new(q_new, ik):
            self.q = q_new
        self.last_ik = ik
        return self.q.copy(), self.last_ik


class IncrementalTeleopTickerV3:
    """One 50 Hz tick of the v3 teleop, frame-in / commands-out.

    Ordering mirrors v2 minus the SMPL consumers:
        freshness -> grip LPF -> pose LPF -> engage/disengage -> per-arm
        [wrist mapping -> 4-DoF IK].

    ``smpl_joints`` in the frame is IGNORED (decision 1); the waist is a
    constant ``waist_stand``. Stale frame -> triggers read 0 (arms freeze)
    and, by default, the filtered grip goes to 0 = hands OPEN (see the
    module docstring: this is the inverted-grip safety semantic; pass
    ``stale_grip_close=True`` to hold the last grip instead).
    """

    def __init__(
        self,
        position_scale: float = 1.0,
        target_slew_m_s: float = 1.0,
        input_tau: float = 0.12,
        trigger_thresh: float = TRIGGER_THRESH,
        stale_s: float = 0.5,
        stale_grip_close: bool = False,
        torso_keepout: dict | None = TORSO_KEEPOUT,
        dt: float = DT,
        grip_tau: float = GRIP_TAU,
        wrist_scale: float = 1.0,
        wrist_signs: dict | None = None,
        wrist_slew_rad_s: float = 6.0,
        wrist_pi_guard_rad: float = 2.94,
        waist_stand: float = 0.0,
    ) -> None:
        self.dt = float(dt)
        self.trigger_thresh = float(trigger_thresh)
        self.stale_s = float(stale_s)
        self.stale_grip_close = bool(stale_grip_close)
        self.grip_tau = float(grip_tau)
        self.waist_stand = float(waist_stand)
        signs = wrist_signs if wrist_signs is not None else DEFAULT_WRIST_SIGNS
        self.arms = {
            side: ClutchArmV3(
                side,
                position_scale,
                target_slew_m_s=target_slew_m_s,
                torso_keepout=None if torso_keepout is None else torso_keepout[side],
                wrist_scale=wrist_scale,
                wrist_signs={s: signs[s] for s in ("left", "right")},
                wrist_slew_rad_s=wrist_slew_rad_s,
                wrist_pi_guard_rad=wrist_pi_guard_rad,
            )
            for side in ("left", "right")
        }
        self.lpfs = {"left": PoseLpf(input_tau), "right": PoseLpf(input_tau)}
        self.grip_f = {"l": 0.0, "r": 0.0}

    def step(self, frame, now: float | None = None) -> dict:
        now = time.monotonic() if now is None else now
        dt = self.dt
        zero7 = np.zeros(7, np.float64)
        fresh = frame is not None and now - frame["t"] <= self.stale_s
        if fresh:
            lt, rt = (float(x) for x in frame.get("trigger", (0.0, 0.0)))
            lg, rg = (float(x) for x in frame.get("grip", (0.0, 0.0)))
            lcp = np.asarray(frame.get("lctrl_pose", zero7), np.float64)
            rcp = np.asarray(frame.get("rctrl_pose", zero7), np.float64)
            a = dt / (self.grip_tau + dt)
            self.grip_f["l"] += a * (lg - self.grip_f["l"])
            self.grip_f["r"] += a * (rg - self.grip_f["r"])
        else:
            lt = rt = lg = rg = 0.0
            if not self.stale_grip_close:  # v3 default: stale -> hands OPEN
                self.grip_f["l"] = self.grip_f["r"] = 0.0
            lcp = rcp = zero7
            self.lpfs["left"].reset()
            self.lpfs["right"].reset()
        if fresh:
            lcp_pre_lpf = lcp.copy()
            rcp_pre_lpf = rcp.copy()
            lcp = self.lpfs["left"].step(lcp, dt)
            rcp = self.lpfs["right"].step(rcp, dt)
        else:
            lcp_pre_lpf = rcp_pre_lpf = np.zeros(7)
        held = {
            "left": fresh and lt > self.trigger_thresh,
            "right": fresh and rt > self.trigger_thresh,
        }
        ctrl = {"left": lcp, "right": rcp}
        for side, arm in self.arms.items():
            if held[side] and not arm.engaged:
                arm.engage(ctrl[side])  # refuses zero placeholders
            elif not held[side] and arm.engaged:
                arm.disengage()
        q = {}
        for side, arm in self.arms.items():
            q[side], _ = arm.step(held[side], ctrl[side])
        return {
            "fresh": fresh,
            "lt": lt,
            "rt": rt,
            "grip_raw": {"l": lg, "r": rg},
            "grip": dict(self.grip_f),
            "held": held,
            "ctrl_pre": {"left": lcp_pre_lpf, "right": rcp_pre_lpf},
            "ctrl": ctrl,
            "q": q,
            "waist": self.waist_stand,
            "wrist_tgt": {s: self.arms[s].wrist.cmd.copy() for s in ("left", "right")},
            "wrist_delta_ypr": {
                s: self.arms[s].wrist.last_delta_ypr.copy() for s in ("left", "right")
            },
            "wrist_dropped": {s: self.arms[s].wrist.dropped_ticks for s in ("left", "right")},
        }


# ---------------------------------------------------------------------------
# Self-test checks (pytest mirrors these 1:1). All pure -- no mujoco, no
# ROS, no zmq.
# ---------------------------------------------------------------------------
def _engaged_arm(side: str, **kw) -> ClutchArmV3:
    """A v3 arm anchored at its neutral with an identity controller quat."""
    arm = ClutchArmV3(side, **kw)
    pose = np.zeros(7)
    pose[3] = 1.0
    pose[:3] = arm.solver.fk(arm.q)[0]
    assert arm.engage(pose)
    return arm


def _quat_axis(deg: float, axis: str) -> np.ndarray:
    r = sRot.from_euler(axis, np.radians(deg))
    q = r.as_quat()  # xyzw
    return q[[3, 0, 1, 2]]  # -> wxyz


def check_wrist_zero_delta_noop() -> None:
    """A constant controller pose after engage must hold the wrist command
    bit-exact and leave the arm settled (the anchored -- not integrated --
    mapping cannot drift)."""
    for side in ("left", "right"):
        arm = _engaged_arm(side, wrist_signs={"left": (1, 1, 1), "right": (1, 1, 1)})
        anchor_q = arm.q[4:7].copy()
        pose = np.zeros(7)
        pose[3] = 1.0
        pose[:3] = arm.solver.fk(arm.q)[0]
        q_prev = arm.q.copy()
        worst = 0.0
        for _ in range(300):
            q, _ = arm.step(True, pose)
            worst = max(worst, float(np.abs(q - q_prev).max()))
            q_prev = q
        np.testing.assert_array_equal(
            arm.wrist.cmd, anchor_q, err_msg="constant pose must not move the wrist"
        )
        assert worst < 0.005, f"arm still moving on a constant pose: {worst:.4f} rad/tick"


def check_wrist_single_axis_direction() -> None:
    """A pure controller rotation about one world axis must map to exactly
    the corresponding wrist joint (ZYX split), on both sides, and a sign
    flip must flip the output."""
    axes = (("z", 0), ("y", 1), ("x", 2))  # (rot axis, wrist index)
    for side in ("left", "right"):
        for axis, idx in axes:
            for sign in (1.0, -1.0):
                signs = tuple(sign if i == idx else 1.0 for i in range(3))
                arm = _engaged_arm(side, wrist_signs={"left": signs, "right": signs})
                pose = np.zeros(7)
                pose[3] = 1.0
                pose[:3] = arm.solver.fk(arm.q)[0]
                pose[3:] = _quat_axis(30.0, axis)
                for _ in range(300):  # past the 6 rad/s slew
                    arm.step(True, pose)
                want = sign * np.radians(30.0)
                got = arm.wrist.cmd
                assert abs(got[idx] - want) < np.radians(1.0), (
                    f"{side} axis {axis}: wrist[{idx}] {np.degrees(got[idx]):.2f} deg, "
                    f"want {np.degrees(want):.2f}"
                )
                others = [got[i] for i in range(3) if i != idx]
                assert max(abs(o) for o in others) < np.radians(
                    1.0
                ), f"{side} axis {axis}: cross-axis leak {np.degrees(others)} deg"
                assert np.isfinite(got).all()


def check_wrist_limit_clamp() -> None:
    """A wrist driven past its limits must park AT the limit (anchored
    mapping -> no grinding) -- including the ASYMMETRIC left roll range.
    Each drive gets a FRESH neutral anchor (the anchor accumulates: an
    anchor left at a limit would offset the next drive)."""
    limits = WristClutch("left").limits
    # (wrist idx, rot axis, drive deg) -- stay under 180: past it the
    # differential takes the shortest-path sign and parks the OTHER limit
    # (correct behaviour, different assertion)
    for idx, axis, deg in ((0, "z", 150.0), (0, "z", -150.0), (2, "x", -120.0), (2, "x", 120.0)):
        arm = _engaged_arm("left", wrist_signs={"left": (1, 1, 1), "right": (1, 1, 1)})
        p = np.zeros(7)
        p[3] = 1.0
        p[:3] = arm.solver.fk(arm.q)[0]
        p[3:] = _quat_axis(deg, axis)
        for _ in range(400):
            arm.step(True, p)
        got = arm.wrist.cmd[idx]
        want = limits[idx, 1 if deg > 0 else 0]
        assert abs(got - want) < np.radians(1.0), (
            f"wrist[{idx}] driven {deg} deg -> {np.degrees(got):.2f} deg, "
            f"want the limit {np.degrees(want):.2f}"
        )
        assert np.isfinite(arm.wrist.cmd).all()


def check_wrist_slew_and_pi_guard() -> None:
    """One 50 Hz tick may move the wrist command by at most slew*dt; a
    differential past the near-pi guard is DROPPED (command held, counted)
    instead of decomposed ambiguously."""
    w = WristClutch("left", signs=(1, 1, 1), slew_rad_s=6.0, pi_guard_rad=2.94)
    assert w.engage(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    q90 = _quat_axis(90.0, "z")
    prev = w.cmd.copy()
    assert float(np.abs(w.step(q90) - prev).max()) <= 6.0 * DT + 1e-12, "slew violated"
    assert w.dropped_ticks == 0, "90 deg must not trip the near-pi guard"
    q170 = _quat_axis(170.0, "z")  # 2.967 rad > 2.94
    before = w.cmd.copy()
    out = w.step(q170)
    np.testing.assert_array_equal(out, before, err_msg="past-pi tick must be dropped")
    assert w.dropped_ticks == 1


def check_ik4_convergence_wrist_frozen() -> None:
    """Ground-truth reach about the shoulder pitch with the wrist pinned
    at a NON-ZERO anchor: the 4-DoF solve must track the exact solution
    and never touch q[4:7] (mirrors v2's check_ik_convergence)."""
    arm = ClutchArmV3("left", wrist_signs={"left": (1, 1, 1), "right": (1, 1, 1)})
    wrist_anchor = np.array([0.5, 0.2, -0.5])
    arm.q[4:7] = wrist_anchor  # a non-trivial in-range wrist anchor
    pos0, _ = arm.solver.fk(arm.q)
    pose = np.zeros(7)
    pose[3] = 1.0
    pose[:3] = pos0
    assert arm.engage(pose)
    a_rob = arm.anchor[2]
    errs, q_errs = [], []
    dth = 0.01
    for k in range(1, 51):
        q_star = np.concatenate([ARM_NEUTRAL["left"][:4], wrist_anchor])
        q_star[0] += dth * k
        p_star, _ = arm.solver.fk(q_star)
        p = pose.copy()
        p[:3] = pos0 + (p_star - a_rob)
        q, _ = arm.step(True, p)
        errs.append(float(np.linalg.norm(p_star - arm.solver.fk(q)[0])))
        q_errs.append(float(np.max(np.abs(q - q_star))))
        np.testing.assert_allclose(
            q[4:7], wrist_anchor, atol=1e-12, err_msg="the 4-DoF solve must not touch the wrist"
        )
    assert max(errs[-10:]) < 0.005, f"final pos err {max(errs[-10:]) * 1000:.1f} mm (> 5 mm)"
    assert q_errs[-1] < 0.05, f"final joint err {q_errs[-1]:.3f} rad vs ground truth"


def check_ik4_tracks_wrist_coupling() -> None:
    """Position target held constant while the wrist yaws 1 rad: the
    wrist command sweeps its full range, the 4-DoF solve must not fight
    it (the wrist is untouched, checked bit-for-bit) and the solved
    point must stay on the stationary target. The wrist's own leverage
    on that point is small (< 1 mm), so the 8 mm bound here is a coarse
    regression guard, not a tight tracking figure."""
    arm = _engaged_arm("left", wrist_signs={"left": (1, 1, 1), "right": (1, 1, 1)})
    target = arm.anchor[2].copy()
    pose = np.zeros(7)
    pose[3] = 1.0
    pose[:3] = arm.solver.fk(arm.q)[0]
    pose[3:] = _quat_axis(np.degrees(1.0), "z")
    for _ in range(300):
        q, _ = arm.step(True, pose)
    assert abs(arm.wrist.cmd[0] - 1.0) < 1e-6, "wrist did not reach the 1 rad command"
    p, _ = arm.solver.fk(q)
    assert (
        np.linalg.norm(p - target) < 0.008
    ), f"wrist coupling not compensated: EE off by {np.linalg.norm(p - target) * 1000:.1f} mm"


def check_clutch_freeze_reanchor_v3() -> None:
    """Release freezes the whole q7 INCLUDING the wrist; re-engage is a
    zero-jump re-anchor; zero placeholders are refused."""
    arm = _engaged_arm("right")
    pose = np.zeros(7)
    pose[3] = 1.0
    pose[:3] = arm.solver.fk(arm.q)[0]
    p = pose.copy()
    p[3:] = _quat_axis(40.0, "z")
    for _ in range(200):
        arm.step(True, p)
    q_frozen = arm.q.copy()
    moved = p.copy()
    moved[3:] = _quat_axis(120.0, "z")
    moved[0] += 0.2
    for _ in range(5):
        q, _ = arm.step(False, moved)
        np.testing.assert_array_equal(q, q_frozen)
    assert arm.engage(moved), "re-engage refused a valid pose"
    q0 = arm.q.copy()
    q, _ = arm.step(True, moved)
    np.testing.assert_allclose(q, q0, atol=1e-9, err_msg="re-anchor jump (delta != 0)")
    assert not arm.engage(np.zeros(7)), "zero placeholder must be refused"


def check_ik_freeze_guard_v3() -> None:
    """The v2 freeze guard (shared via _gate_q_new) must gate the 4-DoF
    solve identically: sustained solver-cap steps latch after the
    persistence window, a diverged/NaN solve latches instantly, and the
    arm resumes when the solver recovers."""
    arm = ClutchArmV3(
        "right", target_slew_m_s=0.0, wrist_signs={"left": (1, 1, 1), "right": (1, 1, 1)}
    )
    pose = np.zeros(7)
    pose[3] = 1.0
    pose[:3] = arm.solver.fk(arm.q)[0]
    assert arm.engage(pose)
    orig = arm.solver.solve_position_free4

    def flip_solve(q_seed, *_a, **_k):
        return np.asarray(q_seed, np.float64) + 0.5, IKResult(0.01, 0.0, 2, 0)

    arm.solver.solve_position_free4 = flip_solve
    for _ in range(arm.hold_ticks - 1):
        arm.step(True, pose)
    assert not arm.holding, "persistence window: must not latch on the first spike"
    q_mid = arm.step(True, pose)[0]
    assert arm.holding, "sustained solver-cap steps must latch"
    q_a = arm.step(True, pose)[0]
    np.testing.assert_array_equal(q_a, q_mid, err_msg="latched: resend the last command")

    def lost_solve(q_seed, *_a, **_k):
        return np.asarray(q_seed, np.float64), IKResult(0.9, 0.0, 2, 0)

    arm.solver.solve_position_free4 = lost_solve
    q_b = arm.step(True, pose)[0]
    assert arm.holding and np.array_equal(q_b, q_a), "diverged solve must hold instantly"

    arm.solver.solve_position_free4 = lambda _s, *_a, **_k: (
        np.full(7, np.nan),
        IKResult(9.9, 9.9, 0, 0),
    )
    q_c = arm.step(True, pose)[0]
    assert arm.holding and np.array_equal(q_c, q_a), "NaN solve must hold instantly"

    arm.solver.solve_position_free4 = orig
    arm.disengage()
    assert arm.engage(pose)
    q, _ = arm.step(True, pose)
    assert not arm.holding, "resume after the solver recovers"
    np.testing.assert_allclose(q, q_a, atol=1e-9)


def check_torso_keepout_v3() -> None:
    """The inherited keep-out sandwich still holds for the 4-DoF arm (the
    block was copied into ClutchArmV3.step): a cross-body sweep slides
    along the surface, latches past the limit, and resumes on pullback.
    Scenario identical to v2's check_torso_keepout."""
    keepout = TORSO_KEEPOUT["right"]
    cx, cy, a_ax, b_front, b_back, _zlo, _zhi = keepout
    arm = ClutchArmV3(
        "right",
        1.0,
        torso_keepout=keepout,
        wrist_signs={"left": (1, 1, 1), "right": (1, 1, 1)},
    )
    other = ClutchArmV3("left")
    pos0, _ = arm.solver.fk(arm.q)
    pos_l, _ = other.solver.fk(other.q)
    pose = np.zeros(7)
    pose[3] = 1.0
    pose[:3] = pos0
    assert arm.engage(pose)
    latched_early, latched, froze, min_e = False, False, True, 1e9
    q_hold_ref, p_at_limit, tail_err = None, None, 9.9
    for k in range(700):
        s = 0.8 * k / 399.0 if k < 400 else 0.8 - 0.45 * (k - 400) / 299.0
        op = pose.copy()
        op[:3] = pos0 + s * (pos_l - pos0)
        op[2] += 0.15 * s
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


def _twist_j24(deg: float) -> np.ndarray:
    """Standing skeleton with the shoulder line twisted `deg` vs the hips."""
    j24 = np.zeros((24, 3))
    j24[12] = [0, 0, 0.6]
    j24[1], j24[2] = [0, 0.1, 0.0], [0, -0.1, 0.0]
    j24[16], j24[17] = [0, 0.2, 0.5], [0, -0.2, 0.5]
    th = np.radians(deg)
    rz = np.array([[np.cos(th), -np.sin(th), 0.0], [np.sin(th), np.cos(th), 0.0], [0.0, 0.0, 1.0]])
    j24[16] = rz @ j24[16]
    j24[17] = rz @ j24[17]
    return j24


def check_waist_fixed_and_smpl_ignored() -> None:
    """A 25-deg shoulder-line twist in smpl_joints must move NOTHING: the
    waist stays at stand and the arm commands are element-for-element
    identical to a straight skeleton (the SMPL channel is closed)."""
    tick = IncrementalTeleopTickerV3()
    lcp = np.zeros(7)
    lcp[3] = 1.0
    lcp[:3] = tick.arms["left"].solver.fk(tick.arms["left"].q)[0]
    base = {
        "trigger": np.array([0.9, 0.0]),
        "grip": np.array([0.0, 0.0]),
        "lctrl_pose": lcp,
        "rctrl_pose": np.zeros(7),
        "t": 100.0,
    }
    out = {}
    for tag, j24 in (("straight", _twist_j24(0.0)), ("twisted", _twist_j24(25.0))):
        f = dict(base, smpl_joints=_pack_j24(j24))
        out[tag] = tick.step(f, now=100.01)
    assert out["straight"]["waist"] == out["twisted"]["waist"] == tick.waist_stand
    for side in ("left", "right"):
        np.testing.assert_array_equal(
            out["straight"]["q"][side],
            out["twisted"]["q"][side],
            err_msg=f"{side}: smpl_joints must not leak into the arms",
        )


def _pack_j24(j24: np.ndarray) -> np.ndarray:
    return np.asarray(j24, np.float32).reshape(-1)


def check_grip_stale_open() -> None:
    """Stale frame -> grip 0 = hands OPEN (the v3 inverted semantic);
    --stale-grip-close holds the last filtered grip instead."""
    tick = IncrementalTeleopTickerV3()
    frame = {
        "trigger": np.array([0.9, 0.9]),
        "grip": np.array([1.0, 1.0]),
        "lctrl_pose": np.zeros(7),
        "rctrl_pose": np.zeros(7),
        "smpl_joints": np.zeros(72),
        "t": 100.0,
    }
    tick.step(frame, now=100.01)
    held = dict(tick.grip_f)
    assert held["l"] > 0.0, "grip LPF must have started rising"
    st = tick.step(frame, now=100.01 + 5.0)  # way past stale_s
    assert not st["fresh"]
    assert st["grip"]["l"] == 0.0 and st["grip"]["r"] == 0.0, "stale must OPEN the hands"
    tick2 = IncrementalTeleopTickerV3(stale_grip_close=True)
    tick2.step(frame, now=100.01)
    st2 = tick2.step(frame, now=100.01 + 5.0)
    assert st2["grip"]["l"] > 0.0, "--stale-grip-close must hold the last grip"


CHECKS = [
    ("wrist_zero_delta_noop", check_wrist_zero_delta_noop),
    ("wrist_single_axis_direction", check_wrist_single_axis_direction),
    ("wrist_limit_clamp", check_wrist_limit_clamp),
    ("wrist_slew_and_pi_guard", check_wrist_slew_and_pi_guard),
    ("ik4_convergence_wrist_frozen", check_ik4_convergence_wrist_frozen),
    ("ik4_tracks_wrist_coupling", check_ik4_tracks_wrist_coupling),
    ("clutch_freeze_reanchor_v3", check_clutch_freeze_reanchor_v3),
    ("ik_freeze_guard_v3", check_ik_freeze_guard_v3),
    ("torso_keepout_v3", check_torso_keepout_v3),
    ("waist_fixed_and_smpl_ignored", check_waist_fixed_and_smpl_ignored),
    ("grip_stale_open", check_grip_stale_open),
]


def run_self_test(checks=None) -> bool:
    """Run this module's pure checks -- or a caller-composed list (the sim
    harness passes CHECKS + its plant-dependent ones). True iff all pass."""
    checks = CHECKS if checks is None else checks
    return _v2_run_self_test(checks)


if __name__ == "__main__":
    raise SystemExit(0 if run_self_test() else 1)
