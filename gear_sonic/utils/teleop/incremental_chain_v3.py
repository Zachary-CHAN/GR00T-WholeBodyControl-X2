#!/usr/bin/env python3
"""v3 incremental upper-body teleop chain -- 4-DoF arm IK + direct wrist fill.

Same skeleton as v2 (``incremental_chain.py``): grip clutch, dual
anchoring, torso keep-out sandwich, reach sphere, target slew, freeze
guard, pose LPF. Deliberate changes:

1. SMPL channel CLOSED: no ``smpl_joints`` consumption, no TorsoFrame /
   WaistYawChain -- the waist joints hold a fixed stand value.
2. Arm IK = the v1-style 4-DoF split (``full_pose_ik=False``, the
   default): position-only DLS/QP over the shoulder-3 + elbow joints
   (``ArmIKSolver.solve_position_free4`` / ``MinkArmIK.solve_position_free4``);
   the three wrist joints are LOCKED at the WristClutch command and ride
   along in the FK (they move the solved point -- the spherical wrist
   centre -- by at most 0.98 mm, so the 4-DoF solve re-runs against the
   CURRENT wrist command every tick to keep its sub-mm residual honest).
   The wrist is filled from the controller quaternion differential,
   anchored at engage, by solving the wrist chain exactly:
       delta       = a_op_inv * R_now              (controller body frame)
       delta_robot = C * delta * C^T               (C: signs + axis_order)
       W(q_wrist)  = W(q_wa) * delta_robot         (W = Rz(yaw)Ry(pitch)Rx(roll))
       q_wrist_tgt = euler_ZYX(W(q_wa) * delta_robot)
   C is the constant signed-permutation correspondence between the
   controller body axes and the wrist joint axes (the historical
   --wrist-*-sign-* / --wrist-axis-order-* knobs). The X2 wrist as
   modeled by the vendor URDF/MJCF (and used by the vendor RL deploy and
   HAL limit tables) is yaw proximal (z), then pitch (y) and roll (x,
   distal) whose axes intersect at one point -- a cross-axis pair driven
   in parallel through the forearm. With that chain a rotation about the
   tool's own x axis is always the roll joint alone, but a rotation about
   the tool's y or z axis is NOT a single-joint motion once the distal
   joints are off neutral: exact hand-orientation tracking coordinates
   pitch+roll (or all three). Do not replace the solve with the
   joint-space mirroring q_wa + euler(delta): it is exact only at a zero
   anchor / tool roll, and at a bent wrist it left 10-14 deg of tool
   error on a 20 deg input (2026-10-20 robot report: track still poor).
   The older W_a^-1 delta W_a conjugation was worse (18-26 deg).

   ``full_pose_ik=True`` is an explicit opt-in alternative: the FULL
   7-DoF pose IK (position AND orientation tasks; mink uses
   position_cost 1.5 and wrist-frame orientation costs (1.0, 0.7, 1.0),
   i.e. pitch yields to position while roll/yaw keep full authority; DLS
   treats both equally) solves the wrist too. ``WristClutch.step`` still runs every
   tick on both paths; on the full-pose path its command is telemetry only
   (record columns 49:61).
3. HAND INVERTED: ``r = hand`` -- the hand is OPEN at rest and CLOSES when
   the TRIGGER is pressed (v2 was the opposite). ``hand`` is the LPF'd raw
   trigger; the GRIP now drives the per-arm clutch (2026-10 role swap; the
   constant / CLI names are historical). SAFETY SEMANTIC CHANGE: a
   stale/dropped frame now OPENS the hands (v2 closed them). If holding an
   object through a dropout matters more, pass ``stale_grip_close=True``
   (--stale-grip-close) to keep the last filtered trigger instead.

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
        _rot_to_quat_wxyz,
        make_arm_solver,
        run_self_test as _v2_run_self_test,
    )
    from gear_sonic.utils.teleop.solver.arm import IKResult
    from gear_sonic.utils.teleop.solver.arm.x2_arm_ik import (
        X2_ARM_JOINT_LIMITS_RAD,
    )
except ImportError:  # pragma: no cover - flat vendored copy (PC2 czq_examples)
    # The v3 robot bundle ships its base chain as ``incremental_chain_v3base``
    # so it does NOT overwrite the PC2 copy of ``incremental_chain`` that the
    # v2 node still imports (the v2 trigger/grip semantics predate the role
    # swap in this tree). Fall back to the bare name for older bundles.
    try:
        from incremental_chain_v3base import (
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
            _rot_to_quat_wxyz,  # type: ignore[no-redef]
            make_arm_solver,  # type: ignore[no-redef]
            run_self_test as _v2_run_self_test,  # type: ignore[no-redef]
        )
    except ImportError:
        from incremental_chain import (  # type: ignore[no-redef]
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
            _rot_to_quat_wxyz,  # type: ignore[no-redef]
            make_arm_solver,  # type: ignore[no-redef]
            run_self_test as _v2_run_self_test,  # type: ignore[no-redef]
        )
    from x2_arm_ik import (  # type: ignore[no-redef]
        X2_ARM_JOINT_LIMITS_RAD,
        IKResult,
    )

# Default per-axis wrist signs (yaw, pitch, roll) per side, applied to the
# controller body-frame axes (z -> yaw, y -> pitch, x -> roll).
#
# LIVE A/B IN PROGRESS (2026-10-21, second v4 trackerless 3-point session).
# History: the original constants were a hand-derived guess -- v1's
# x2_wrist_map signs plus an assumed left/right MIRROR -- and were marked
# UNVERIFIED.  Session 1 with the mirror (left == -right) showed the left
# yaw AND pitch reversed while the right yaw was fine.  Session 2 with the
# sides made parallel at (1, -1, 1) fixed the left hand but left the RIGHT
# pitch wrong.  The current experiment is (1, 1, 1) on both sides (::
# controller +y -> +pitch), run WITHOUT --ctrl-lever calibration, to settle
# the pitch sign empirically.  Do not treat these as validated until the
# live result is in; override per axis with WRIST3_*_SIGN /
# --wrist-{yaw,pitch,roll}-sign if only one axis is off.
DEFAULT_WRIST_SIGNS = {
    "left": (1.0, 1.0, 1.0),
    "right": (1.0, 1.0, 1.0),
}


def parse_wrist_axis_order(text) -> tuple[int, int, int]:
    """Parse '0,1,2' into a wrist-joint <- controller-axis correspondence.

    Wrist joints are [yaw, pitch, roll]; the controller differential's
    ZYX components are [about z, about y, about x]. The default (0,1,2)
    maps controller z->yaw, y->pitch, x->roll. Use 2,1,0 to swap the x/z
    assignment when a single anatomical DOF lands on the wrong joint.
    The permutation feeds the constant frame correspondence C applied to
    the differential BEFORE the exact wrist solve (see WristClutch); it is
    not a per-joint scaling of Euler components.
    """
    try:
        order = tuple(int(v) for v in str(text).replace(",", " ").split())
    except ValueError as exc:
        raise ValueError(f"expected three integers, got {text!r}") from exc
    if sorted(order) != [0, 1, 2]:
        raise ValueError(f"must be a permutation of 0,1,2; got {text!r}")
    return order


class WristClutch:
    """Exact X2 wrist-chain solve from the controller differential, anchored
    at engage.

    engage() snapshots the wrist q AND the controller orientation; every
    held tick recomputes the FULL differential from the anchor (not a
    per-tick increment -- no integration drift, and a constant controller
    pose yields a bit-exact constant command).

    The differential ``R_a^-1 R_now`` is expressed in the CONTROLLER's
    anchor frame, and the controller body axes are rigid with the hand.
    The robot wrist must realize that same body-frame rotation in its tool
    frame, so the joint target is the ZYX decomposition of
    ``W(q_wa) @ delta_robot`` with ``W = Rz(yaw)Ry(pitch)Rx(roll)`` --
    the exact inverse of the X2 wrist chain (yaw proximal, pitch/roll
    cross-axis, roll distal).

    Consequences: a controller rotation about x (tool roll) stays a pure
    roll-joint motion at any anchor; a rotation about y/z (tool pitch/yaw)
    moves pitch+roll (or all three) once the distal joints are off
    neutral. That coordination is the mechanism, not a mapping bug -- the
    TOOL orientation is what must track. The old joint-space mirroring
    ``q_wa + euler(delta)`` (2026-10-20 first fix) is exact only at a zero
    anchor / tool roll and leaves 10-14 deg of tool error at a bent wrist;
    the older ``W_a^-1 delta W_a`` conjugation was worse (18-26 deg).

    ``signs`` and ``axis_order`` define the constant signed-permutation
    correspondence C between controller body axes and wrist joint axes
    (default: controller z->yaw, y->pitch, x->roll, all +). It is applied
    as ``C delta C^T`` BEFORE the solve.
    """

    def __init__(
        self,
        side: str,
        scale: float = 1.0,
        signs: tuple | None = None,
        slew_rad_s: float = 6.0,
        pi_guard_rad: float = 2.94,
        limits=None,
        axis_order: tuple | None = None,
    ) -> None:
        self.side = side
        self.scale = float(scale)
        self.signs = np.asarray(
            DEFAULT_WRIST_SIGNS[side] if signs is None else signs, np.float64
        ).reshape(3)
        # wrist joint i takes controller Euler component axis_order[i]
        # (Euler ZYX order is [yaw(z), pitch(y), roll(x)]). A permutation
        # only -- anything else would duplicate/drop a physical axis.
        order = np.array((0, 1, 2) if axis_order is None else axis_order, np.int64).reshape(3)
        if sorted(order.tolist()) != [0, 1, 2]:
            raise ValueError(f"axis_order must be a permutation of 0,1,2; got {order.tolist()}")
        self.axis_order = order
        # Constant controller-axis -> wrist-axis correspondence C (signed
        # permutation): C maps a controller-frame rotation vector to the
        # wrist-frame one, delta_robot = C delta C^T. Wrist joint i is the
        # xyz axis (z, y, x) = (2, 1, 0); Euler component k is that same
        # xyz axis. Signs flip the direction of the corresponding
        # controller axis.
        self._corr = np.zeros((3, 3), np.float64)
        for i in range(3):
            robot_axis = (2, 1, 0)[i]  # yaw->z, pitch->y, roll->x
            ctrl_axis = (2, 1, 0)[order[i]]
            self._corr[robot_axis, ctrl_axis] = self.signs[i]
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
        self._a_op_inv = _rot_from_quat_wxyz(q).inv()
        self.cmd = self._q_wa.copy()
        self.last_delta_ypr = np.zeros(3)
        return True

    def disengage(self) -> None:
        self._q_wa = None
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
        delta_mag = delta.magnitude()
        if delta_mag > self.pi_guard:
            self.dropped_ticks += 1
            self.last_delta_ypr = np.zeros(3)
            return self.cmd.copy()
        self.last_delta_ypr = delta.as_euler("ZYX")  # (yaw, pitch, roll)
        if delta_mag < 1e-12:
            # bit-exact hold for a constant controller pose: skip the
            # euler -> matrix -> euler round trip (it perturbs the last
            # bits and would defeat the anchored no-drift guarantee)
            return self.cmd.copy()
        # controller body-frame rotation -> wrist-frame rotation, then the
        # exact wrist chain solve W(q_tgt) = W(q_wa) * delta_robot.
        rotvec = self._corr @ delta.as_rotvec()
        delta_robot = sRot.from_rotvec(self.scale * rotvec)
        w_anchor = sRot.from_euler("ZYX", self._q_wa)
        tgt = (w_anchor * delta_robot).as_euler("ZYX")
        tgt = np.clip(tgt, self.limits[:, 0], self.limits[:, 1])
        self.cmd = self.cmd + np.clip(tgt - self.cmd, -self.slew, self.slew)
        return self.cmd.copy()


class ClutchArmV3(ClutchArm):
    """v2's ClutchArm, with the v1-style 4-DoF split as the default.

    Everything positional (anchoring, keep-out sandwich, reach sphere,
    target slew, freeze guard via ``_gate_q_new``) is inherited from v2 on
    both paths.

    ``full_pose_ik=False`` (the default) is the v1-style split:
    ``solve_position_free4`` over shoulder-3 + elbow, with the wrist pinned
    to the WristClutch command. The wrist mapping is computed FIRST in the
    tick so the same tick's IK already compensates the wrist_yaw
    end-effector lever. There is no orientation task on this path -- the
    wrist mapping owns orientation -- so its rotation weight is 0.

    ``full_pose_ik=True`` opts into v2's full-pose target path: the
    controller's pose drives position AND orientation through the IK, so the
    wrist joints are solved rather than scripted. Position and rotation task
    gains are both 1.0 on this path (equal weighting).

    The WristClutch is stepped every tick on both paths: on the 4-DoF path it
    IS the wrist command, on the full-pose path it is telemetry (columns
    49:61 of the record) and keeps its slew and near-pi state warm.
    """

    def __init__(self, side: str, position_scale: float = 1.0, *args, **kwargs):
        wrist_scale = kwargs.pop("wrist_scale", 1.0)
        wrist_signs = kwargs.pop("wrist_signs", None)
        wrist_axis_order = kwargs.pop("wrist_axis_order", None)
        wrist_slew = kwargs.pop("wrist_slew_rad_s", 6.0)
        wrist_pi_guard = kwargs.pop("wrist_pi_guard_rad", 2.94)
        # CONTROLLER LEVER (Pico origin -> operator wrist centre), metres in
        # the controller body frame. The Pico pose is the controller's
        # tracking origin, not the human wrist: rotating the controller about
        # a stationary wrist sweeps that origin on a lever arc, which the
        # position target used to chase -- dragging the robot's shoulder and
        # elbow. ``r_c`` removes the arc: ``W = C - R_c @ r_c``.
        # Zero = legacy behaviour; measure it with a wrist-only sweep (see
        # docs/x2/PICO_LIVE_TELEOP_PITFALLS.md) before trusting a value.
        lever = kwargs.pop("ctrl_lever", None)
        self.ctrl_lever = (
            np.zeros(3, np.float64) if lever is None else np.asarray(lever, np.float64).reshape(3)
        )
        # OPERATOR X/Y SIGN (2026-10-30, live-corrected). The 3-point sender
        # builds [fwd, lat, up] with ``fwd = cross(lat, up)`` from the SDK
        # world. That world is RIGHT-handed (X right, Y up, Z in; the PC
        # service README pins it and notes that the actual data flips
        # Unity's +Z), NOT the left-handed Unity figure the 2026-10-27 round
        # assumed. With the left controller at -X the cross product is
        # therefore -Z = operator FORWARD: wire = (operator forward,
        # operator left, up). The X2 base frame is (forward, left, up) as
        # well (ARM_NEUTRAL puts the wrists at +x) and the operator faces
        # the SAME direction (2026-10-30 live, NOT face-to-face), so the
        # avatar map is the IDENTITY: +1 / +1. -1 mirrors an axis and is
        # only correct for a face-to-face rig; the 2026-10-27 x=-1 round
        # trusted the Unity left-handed figure and made the real robot
        # reach BACKWARD (and, with the trunk keep-out, sideways). The
        # wrist needs neither sign: a constant frame cancels in
        # R_anchor^-1 R_now, which is why rotation looked fine all along.
        # Both flips MUST happen after the lever correction: ``r_c`` and the
        # wire quaternion share the old frame, so mirroring their input
        # coordinates would inject a 2*(R_c r_c)_i error during wrist
        # rotation. Defaults +1 preserve the full-body wire behaviour.
        self.operator_x_sign = float(kwargs.pop("operator_x_sign", 1.0))
        self.operator_y_sign = float(kwargs.pop("operator_y_sign", 1.0))
        if self.operator_x_sign not in (-1.0, 1.0):
            raise ValueError("operator_x_sign must be +1 or -1")
        if self.operator_y_sign not in (-1.0, 1.0):
            raise ValueError("operator_y_sign must be +1 or -1")
        self.full_pose_ik = bool(kwargs.pop("full_pose_ik", False))
        # Soft radial deadband on the compensated translation, metres. The
        # operator's hand tremor and the clutch re-anchor both live at the
        # millimetre scale; subtracting a small radius (ramped continuously
        # past it) keeps that noise out of the upstream joints while a real
        # reach still passes through with slope 1.
        #
        # B/C are DEFAULT-PATH features. The full-pose path is the opt-in
        # alternative whose A/B gate is already marginal (4.38 mm of the
        # 5 mm budget at its freeze-guard bifurcation, tick 854); even a
        # 1 mm deadband perturbs that latch and pushes the two backends
        # apart (measured 5.84 mm). Keep `--full-pose-ik` bit-identical to
        # the pre-A/B/C path; its posture term is a separate migration.
        self.position_deadband_m = float(kwargs.pop("position_deadband_m", 0.0))
        if self.position_deadband_m < 0.0:
            raise ValueError("position_deadband_m must be >= 0")
        if self.full_pose_ik:
            self.position_deadband_m = 0.0
            kwargs["posture_gain"] = 0.0
        if not self.full_pose_ik:
            # the 4-DoF path: position-only IK regardless of what v2 would
            # default to, because the orientation task moved to the wrist
            # mapping there. Force it even if a caller passed a weight.
            kwargs["rotation_weight"] = 0.0
        else:
            # equal weighting: position and orientation are both first-class
            # on the opt-in full-pose path (the 4-DoF default has no
            # orientation task at all -- WristClutch owns it)
            kwargs.setdefault("rotation_weight", 1.0)
        super().__init__(side, position_scale, *args, **kwargs)
        if wrist_signs is None:
            signs = None
        elif isinstance(wrist_signs, dict):
            signs = wrist_signs.get(side)
        else:
            signs = wrist_signs
        if wrist_axis_order is None:
            order = None
        elif isinstance(wrist_axis_order, dict):
            order = wrist_axis_order.get(side)
        else:
            order = wrist_axis_order
        self.wrist = WristClutch(
            side,
            scale=wrist_scale,
            signs=signs,
            slew_rad_s=wrist_slew,
            pi_guard_rad=wrist_pi_guard,
            axis_order=order,
        )

    @property
    def wrist_record(self) -> np.ndarray:
        """The wrist the tick actually commanded -- record columns 49:61.

        Full-pose path: the SOLVED wrist (``self.q[4:7]``); the clutch
        command is telemetry there and legitimately differs from what the
        IK chose. 4-DoF path: the clutch command, which IS the wrist.
        """
        return self.q[4:7] if self.full_pose_ik else self.wrist.cmd

    def _operator_wrist_pos(self, c: np.ndarray) -> np.ndarray:
        """Controller origin -> operator wrist centre, ``C - R_c @ r_c``.

        Either operator sign may mirror the result's x/y, both AFTER the
        lever correction.  The real v4 node uses (x, y) = (+1, +1): the
        3-point wire is already operator (forward, left, up) -- the SDK
        world is right-handed X-right/Y-up/Z-in, so ``fwd = cross(lat,
        up)`` is operator-FORWARD (the 2026-10-27 round mistook it for
        BACKWARD by reading the Unity figure instead of the delivered
        data) -- and the operator faces the SAME direction as the robot
        (2026-10-30 live correction), whose base is also (forward, left,
        up).  The map is the identity; -1 mirrors an axis and is only
        right for a face-to-face rig.
        Doing the flips here (rather than on the wire position or the
        controller quaternion) keeps ``C`` and ``R_c @ r_c`` in the same
        frame: a pure wrist rotation still cancels exactly, while an
        operator translation reaches the robot in the robot's own
        forward/left sense.
        """
        if np.linalg.norm(c[3:]) < 1e-6:
            # zero placeholder (no-xrt / SDK failure): no orientation, no
            # lever correction. engage() refuses this pose before we get here.
            w = np.asarray(c[:3], np.float64).copy()
        else:
            w = c[:3] - _rot_from_quat_wxyz(c[3:]).apply(self.ctrl_lever)
        if self.operator_x_sign < 0.0:
            w[0] = -w[0]
        if self.operator_y_sign < 0.0:
            w[1] = -w[1]
        return w

    def _compensated_target(self, c: np.ndarray, a_op, a_rob) -> np.ndarray:
        """Robot position target from the lever-corrected operator wrist."""
        return a_rob + self.position_scale * (self._operator_wrist_pos(c) - a_op)

    def _clamp_reach_from_anchor(self, tgt_pos: np.ndarray, a_rob: np.ndarray) -> np.ndarray:
        """Clamp to the reach sphere ALONG THE COMMANDED RAY FROM THE ANCHOR.

        The legacy clamp below preserved the direction from the SHOULDER, so
        once a long command saturated, the wrist slid along the sphere toward
        the shoulder's own lateral/height line.  With the left wrist anchored
        13.5 cm lateral of the shoulder (ARM_NEUTRAL: y 0.278 vs 0.143), a
        40 cm operator forward extension slid the wrist ~4.8 cm toward the
        robot's right and ~6.7 cm up while the operator's hand went straight
        forward -- the 2026-10-30 live report "屈肘前伸左手时机器人手往右侧
        偏移".  Intersecting the sphere with the anchor -> raw-target ray keeps
        the operator's commanded direction exact; the wrist just stops at the
        boundary instead of sliding (self-test: reach_clamp_anchor_direction).

        The radial clamp stays as a backstop for an anchor that is itself
        outside the sphere (a fully-extended pose at engage: max_reach is
        0.93 of the sampled max, so that is possible, not a bug).
        """
        v = np.asarray(a_rob, np.float64) - self.shoulder
        d = np.asarray(tgt_pos, np.float64) - a_rob
        dd = float(d @ d)
        if dd > 1e-12:
            b = float(v @ d)
            disc = b * b - dd * (float(v @ v) - self.max_reach**2)
            if disc >= 0.0:
                lam = (-b + np.sqrt(disc)) / dd
                if lam < 1.0:
                    tgt_pos = a_rob + max(lam, 0.0) * d
        rad = tgt_pos - self.shoulder
        r = float(np.linalg.norm(rad))
        if r > self.max_reach:
            tgt_pos = self.shoulder + rad * (self.max_reach / r)
        return tgt_pos

    def _deadband(self, tgt_pos: np.ndarray, a_rob: np.ndarray) -> np.ndarray:
        """Soft radial deadband around the engage anchor.

        ``d -> d * max(0, 1 - r/|d|)``: inside radius ``r`` the target is the
        anchor exactly; outside it the output is ``|d| - r`` along the same
        direction (continuous, unit slope). Applied to the COMPENSATED
        translation, before the keep-out / reach / slew sandwich.
        """
        r = self.position_deadband_m
        if r <= 0.0:
            return tgt_pos
        d = tgt_pos - a_rob
        n = float(np.linalg.norm(d))
        if n <= r:
            return a_rob.copy()
        return a_rob + d * (1.0 - r / n)

    def engage(self, ctrl_pose7: np.ndarray) -> bool:
        c = np.asarray(ctrl_pose7, np.float64).reshape(7)
        if np.linalg.norm(c[:3]) < 1e-6 or np.linalg.norm(c[3:]) < 1e-6:
            return False  # same zero-placeholder refusal as ClutchArm.engage
        # Anchor the corrected wrist-centre position; the orientation used by
        # WristClutch and the full-pose path is the raw controller rotation.
        c_corr = c.copy()
        c_corr[:3] = self._operator_wrist_pos(c)
        if not super().engage(c_corr):
            return False
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
        # stepped on both paths, and FIRST: on the 4-DoF path the solve below
        # must see this tick's wrist command in its FK, or the wrist_yaw lever
        # reads as arm error; on the full-pose path it is telemetry only
        wrist_q3 = self.wrist.step(c[3:])
        a_op, a_op_inv, a_rob, a_rob_R = self.anchor
        # a_op is the corrected (lever-removed) operator wrist position from
        # engage(); the same correction here makes a pure controller rotation
        # about a stationary wrist produce ZERO translation target.
        tgt_pos = self._compensated_target(c, a_op, a_rob)
        tgt_pos = self._deadband(tgt_pos, a_rob)
        # v2's orientation target: the controller's rotation delta carried
        # into the robot frame by the anchor. Only the full-pose path feeds it
        # to the solver; the 4-DoF path drops it (the wrist mapping owns
        # orientation there, and the solver was built with rotation_weight=0)
        tgt_rot = a_rob_R * (a_op_inv * _rot_from_quat_wxyz(c[3:]))
        # keep-out sandwich + reach sphere + target slew: identical to the
        # v2 step (copied verbatim -- see incremental_chain.ClutchArm.step
        # for the live-round rationale; the waist compensation is gone
        # with the waist itself)
        if self.torso_keepout is not None:
            tgt_pos = self._project_out_of_trunk(tgt_pos)
        tgt_pos = self._clamp_reach_from_anchor(tgt_pos, a_rob)
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
        off = 0 if self.side == "left" else 7
        if self.full_pose_ik:
            q_new, ik = self.solver.solve(
                self.q, tgt_pos, _rot_to_quat_wxyz(tgt_rot), max_iters=IK_ITERS_PER_TICK
            )
            q_new = np.clip(q_new, ARM_LO_SOLVER[off : off + 7], ARM_HI_SOLVER[off : off + 7])
        else:
            q_new, ik = self.solver.solve_position_free4(
                self.q, wrist_q3, tgt_pos, max_iters=IK_ITERS_PER_TICK
            )
            q_new[4:7] = wrist_q3  # the wrist is commanded by WristClutch only
            q_new[:4] = np.clip(
                q_new[:4], ARM_LO_SOLVER[off : off + 4], ARM_HI_SOLVER[off : off + 4]
            )
        # freeze guard over the FULL q: on the 4-DoF path the wrist part can
        # never trip the jump threshold (its slew bounds the step at
        # 0.12 rad < 0.22); on the full-pose path the wrist is a solved
        # degree of freedom and must be covered, or a wrist jump would slip
        # past the latch
        if self._gate_q_new(q_new, ik):
            self.q = q_new
        self.last_ik = ik
        return self.q.copy(), self.last_ik


class IncrementalTeleopTickerV3:
    """One 50 Hz tick of the v3 teleop, frame-in / commands-out.

    Ordering mirrors v2 minus the SMPL consumers:
        freshness -> grip + trigger LPF -> pose LPF -> engage/disengage ->
        per-arm [wrist mapping -> 4-DoF position IK] (or, with
        ``full_pose_ik=True``, [wrist mapping telemetry -> 7-DoF pose IK]).

    ``smpl_joints`` in the frame is IGNORED (decision 1); the waist is a
    constant ``waist_stand``. Stale frame -> triggers/grips read 0 (arms
    freeze) and, by default, the filtered hand channel (trigger) goes to
    0 = hands OPEN (see the module docstring: this is the inverted-hand
    safety semantic; pass ``stale_grip_close=True`` to hold the last
    filtered trigger instead).
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
        wrist_axis_order: tuple | dict | None = None,
        wrist_slew_rad_s: float = 6.0,
        wrist_pi_guard_rad: float = 2.94,
        waist_stand: float = 0.0,
        rotation_weight: float = 1.0,
        full_pose_ik: bool = False,
        backend: str | None = None,
        ctrl_lever: tuple | dict | None = None,
        operator_x_sign: float = 1.0,
        operator_y_sign: float = 1.0,
        position_cost: float | None = None,
        orientation_axis_weights: tuple | None = None,
        position_deadband_m: float = 0.003,
        posture_gain: float = 0.01,
        home_speed_rad_s: float = 0.5,
    ) -> None:
        self.dt = float(dt)
        self.trigger_thresh = float(trigger_thresh)
        self.stale_s = float(stale_s)
        self.stale_grip_close = bool(stale_grip_close)
        self.grip_tau = float(grip_tau)
        self.waist_stand = float(waist_stand)
        # ARM HOME (X/A long-press, v4 3-point): return one arm to
        # ARM_NEUTRAL at a bounded JOINT rate. 0.5 rad/s (~29 deg/s) keeps
        # the PD target human-plausible; refuse <=0 rather than silently
        # freezing the arm forever.
        self.home_speed_rad_s = float(home_speed_rad_s)
        if self.home_speed_rad_s <= 0.0:
            raise ValueError("home_speed_rad_s must be > 0")
        # Active homing per side, and the previous wire request level (edge
        # detection: the sender latches a short pulse per long press, so a
        # level held high across completion must not restart the motion).
        self.home = {"left": False, "right": False}
        self._home_prev = {"left": False, "right": False}
        # A single 3-vector applies to both arms; a {"left": v, "right": v}
        # mapping calibrates them separately. The two controllers are held at
        # different orientations in the body frame, so a real fit differs by
        # tens of millimetres between hands (see
        # tools/fit_pico_ctrl_lever.py and PICO_LIVE_TELEOP_PITFALLS.md §9).
        self.ctrl_levers = self._normalize_ctrl_levers(ctrl_lever)
        # Legacy single-vector view. It is exactly the left arm's lever; the
        # per-side mapping above is authoritative. Kept for callers written
        # before per-hand calibration existed.
        self.ctrl_lever = self.ctrl_levers["left"]
        self.operator_x_sign = float(operator_x_sign)
        self.operator_y_sign = float(operator_y_sign)
        if self.operator_x_sign not in (-1.0, 1.0):
            raise ValueError("operator_x_sign must be +1 or -1")
        if self.operator_y_sign not in (-1.0, 1.0):
            raise ValueError("operator_y_sign must be +1 or -1")
        self.full_pose_ik = bool(full_pose_ik)
        # The deadband and posture term ship on the 4-DoF default path; the
        # opt-in full-pose path keeps its pre-A/B/C behavior (see
        # ClutchArmV3.__init__ for the A/B measurement that forced this).
        self.position_deadband_m = 0.0 if self.full_pose_ik else float(position_deadband_m)
        self.posture_gain = 0.0 if self.full_pose_ik else float(posture_gain)
        signs = wrist_signs if wrist_signs is not None else DEFAULT_WRIST_SIGNS
        # A single permutation applies to both arms; a mapping lets a grip
        # that differs left/right carry its own (None = identity).
        if wrist_axis_order is None:
            orders = {"left": None, "right": None}
        elif isinstance(wrist_axis_order, dict):
            unknown = set(wrist_axis_order) - {"left", "right"}
            if unknown:
                raise ValueError(f"unknown wrist_axis_order sides: {sorted(unknown)}")
            orders = {s: wrist_axis_order.get(s) for s in ("left", "right")}
        else:
            orders = {"left": wrist_axis_order, "right": wrist_axis_order}
        self.arms = {
            side: ClutchArmV3(
                side,
                position_scale,
                target_slew_m_s=target_slew_m_s,
                torso_keepout=None if torso_keepout is None else torso_keepout[side],
                wrist_scale=wrist_scale,
                wrist_signs={s: signs[s] for s in ("left", "right")},
                wrist_axis_order=orders[side],
                wrist_slew_rad_s=wrist_slew_rad_s,
                wrist_pi_guard_rad=wrist_pi_guard_rad,
                rotation_weight=rotation_weight,
                full_pose_ik=full_pose_ik,
                backend=backend,
                ctrl_lever=self.ctrl_levers[side],
                operator_x_sign=self.operator_x_sign,
                operator_y_sign=self.operator_y_sign,
                # None = the mink solver's shipped task costs (position 1.5,
                # orientation (1, 0.7, 1)); dropped on the dls backend.
                position_cost=position_cost,
                orientation_axis_weights=orientation_axis_weights,
                position_deadband_m=self.position_deadband_m,
                posture_gain=self.posture_gain,
            )
            for side in ("left", "right")
        }
        self.lpfs = {"left": PoseLpf(input_tau), "right": PoseLpf(input_tau)}
        # grip_f: LPF'd wire grip, the clutch decision (recorded in cols 4/5).
        # hand_f: LPF'd wire trigger, the OmniHand drive (v3 ratio = hand_f).
        self.grip_f = {"l": 0.0, "r": 0.0}
        self.hand_f = {"l": 0.0, "r": 0.0}

    @staticmethod
    def _normalize_ctrl_levers(ctrl_lever: tuple | dict | None) -> dict[str, np.ndarray]:
        """Resolve the requested controller lever to one vector per side.

        ``None`` / a 3-vector is the common case (both arms). A mapping lets
        the two hands carry the vectors their own calibration produced; a
        missing side keeps the legacy zero (no compensation) rather than
        silently borrowing the other hand's vector, which they do NOT share.
        """
        if ctrl_lever is None:
            return {side: np.zeros(3, np.float64) for side in ("left", "right")}
        if isinstance(ctrl_lever, dict):
            unknown = set(ctrl_lever) - {"left", "right"}
            if unknown:
                raise ValueError(
                    f"ctrl_lever: unknown side(s) {sorted(unknown)}; use 'left'/'right'"
                )
            return {
                side: np.asarray(ctrl_lever.get(side, (0.0, 0.0, 0.0)), np.float64).reshape(3)
                for side in ("left", "right")
            }
        v = np.asarray(ctrl_lever, np.float64).reshape(3)
        return {"left": v.copy(), "right": v.copy()}

    def _home_slew(self, side: str, arm) -> np.ndarray:
        """One tick of the bounded arm-home return; returns the commanded q.

        Every joint moves toward ARM_NEUTRAL by at most
        ``home_speed_rad_s * dt`` (all joints in parallel, so a 0.5 rad/s
        bound is the per-joint speed, not a norm). The wrist clutch command
        follows the moving wrist so the record and a later re-engage stay
        consistent; the arm's IK telemetry is zeroed. Completion snaps to
        the exact neutral and clears ``self.home[side]``; if the grip is
        still held, the normal engage path re-anchors at the neutral pose
        on the next tick.
        """
        target = np.asarray(ARM_NEUTRAL[side], np.float64)
        dq = target - arm.q
        step = self.home_speed_rad_s * self.dt
        if np.max(np.abs(dq)) <= step:
            arm.q = target.copy()
            self.home[side] = False
        else:
            arm.q = arm.q + np.clip(dq, -step, step)
        arm.wrist.cmd[:] = arm.q[4:7]
        arm.wrist.last_delta_ypr[:] = 0.0
        arm.last_ik = IKResult(0.0, 0.0, 0, 0)
        arm.holding = False
        return arm.q.copy()

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
            # Role swap (2026-10): trigger -> hands (hand_f), grip -> clutch
            # (grip_f). Both one-pole LPFs; held uses grip_f, which is also
            # the recorded clutch channel (cols 4/5), so it stays diagnosable.
            a = dt / (self.grip_tau + dt)
            self.hand_f["l"] += a * (lt - self.hand_f["l"])
            self.hand_f["r"] += a * (rt - self.hand_f["r"])
            self.grip_f["l"] += a * (lg - self.grip_f["l"])
            self.grip_f["r"] += a * (rg - self.grip_f["r"])
        else:
            lt = rt = lg = rg = 0.0
            if not self.stale_grip_close:  # v3 default: stale -> hands OPEN
                self.grip_f["l"] = self.grip_f["r"] = 0.0
                self.hand_f["l"] = self.hand_f["r"] = 0.0
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
            "left": fresh and self.grip_f["l"] > self.trigger_thresh,
            "right": fresh and self.grip_f["r"] > self.trigger_thresh,
        }
        # ARM HOME request (X/A long-press, v4 3-point): the frame may carry
        # a 2-vector of request levels. A rising edge starts that side's
        # bounded return to ARM_NEUTRAL and releases its clutch; the sender
        # turns one long press into a short pulse, so an unbroken level does
        # not restart the motion after completion.
        home_req = {"left": False, "right": False}
        if fresh:
            hv = np.asarray(frame.get("home", (0.0, 0.0)), np.float64).reshape(-1)
            if hv.size >= 2:
                home_req = {"left": bool(hv[0] > 0.5), "right": bool(hv[1] > 0.5)}
        for side in ("left", "right"):
            if home_req[side] and not self._home_prev[side]:
                self.home[side] = True
                self.arms[side].disengage()
                self.arms[side].holding = False
            self._home_prev[side] = home_req[side]
        ctrl = {"left": lcp, "right": rcp}
        for side, arm in self.arms.items():
            if self.home[side]:
                continue  # homing owns this arm; no clutch work this tick
            if held[side] and not arm.engaged:
                arm.engage(ctrl[side])  # refuses zero placeholders
            elif not held[side] and arm.engaged:
                arm.disengage()
        q = {}
        for side, arm in self.arms.items():
            if self.home[side]:
                q[side] = self._home_slew(side, arm) if fresh else arm.q.copy()
            else:
                q[side], _ = arm.step(held[side], ctrl[side])
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
            "waist": self.waist_stand,
            # on the full-pose path this is the SOLVED wrist, not the clutch
            # command (see ClutchArmV3.wrist_record); 85-column layout and
            # column meaning unchanged either way
            "wrist_tgt": {
                s: np.asarray(self.arms[s].wrist_record).copy() for s in ("left", "right")
            },
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
    """At a ZERO wrist anchor a pure controller rotation about one world
    axis must map to exactly the corresponding wrist joint (the exact
    solve reduces to the ZYX split there), on both sides, and a sign flip
    must flip the output.

    ``full_pose_ik=False`` on purpose: this pins the WristClutch mapping,
    which only IS the wrist on the 4-DoF path. On the opt-in full-pose
    path the clutch command is telemetry and the wrist is whatever the IK
    solved -- see check_full_pose_ik_tracks_orientation for that path."""
    axes = (("z", 0), ("y", 1), ("x", 2))  # (rot axis, wrist index)
    for side in ("left", "right"):
        for axis, idx in axes:
            for sign in (1.0, -1.0):
                signs = tuple(sign if i == idx else 1.0 for i in range(3))
                arm = _engaged_arm(
                    side, full_pose_ik=False, wrist_signs={"left": signs, "right": signs}
                )
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


def check_wrist_nonzero_anchor_tracks_tool_rotation() -> None:
    """A controller body-frame rotation must be realized as the SAME tool
    rotation, even when wrist and controller anchor off-neutral:
    ``W(q_cmd) == W(q_wa) @ C delta C^T`` with ``W = Rz Ry Rx``.

    This is the 2026-10-20 robot regression. The old code conjugated by
    ``W(q_wa)`` (``W_a^-1 delta W_a``), and the first fix replaced it with
    joint-space mirroring ``q_wa + euler(delta)``. Both are exact only at
    a zero wrist anchor (the mirroring is also exact for tool roll, the
    distal x axis). At a bent anchor a 20 deg pitch/yaw input left 10-14
    deg (mirroring) or 18-26 deg (conjugation) of TOOL orientation error,
    which the operator saw as the wrist not tracking.

    Why a single input axis legitimately moves >1 joint here: the X2
    wrist is yaw (proximal, z) -> pitch (y) -> roll (distal, x), with the
    pitch/roll crossing at one point (cross-axis pair). Rotating the tool
    about its own y or z axis is not a single-joint motion once pitch/roll
    are off neutral; the exact solve coordinates them. What must be
    one-DOF-in/one-DOF-out is the TOOL motion, which this check pins.
    """
    wrist_anchor = np.array([0.35, 0.15, -0.30])  # in range on BOTH sides
    rot_anchor = sRot.from_euler("ZYX", [0.6, -0.4, 0.8])  # non-identity grip
    anchor_quat = _rot_to_quat_wxyz(rot_anchor)
    cases = (
        ((0, 1, 2), (1.0, 1.0, 1.0)),
        ((2, 1, 0), (1.0, 1.0, 1.0)),
        ((0, 1, 2), (-1.0, 1.0, -1.0)),
    )
    checked = 0
    for side in ("left", "right"):
        for order, signs in cases:
            for axis in ("x", "y", "z"):
                for deg in (20.0, -20.0):
                    w = WristClutch(side, signs=signs, axis_order=order)
                    assert w.engage(wrist_anchor, anchor_quat)
                    delta = sRot.from_euler(axis, np.radians(deg))
                    moved = _rot_to_quat_wxyz(rot_anchor * delta)
                    cmd = w.cmd.copy()
                    for _ in range(300):  # past the 6 rad/s slew
                        cmd = w.step(moved)
                    expected = w._corr @ delta.as_rotvec()
                    desired = sRot.from_euler("ZYX", wrist_anchor) * sRot.from_rotvec(expected)
                    q_exact = desired.as_euler("ZYX")
                    # skip inputs that legitimately clamp at a joint limit
                    if np.any(q_exact < w.limits[:, 0]) or np.any(q_exact > w.limits[:, 1]):
                        continue
                    got = sRot.from_euler("ZYX", cmd)
                    err = (got.inv() * desired).magnitude()
                    assert err < 1e-6, (
                        f"{side} axis {axis} {deg:+.0f} order {order} signs {signs}: "
                        f"tool error {np.degrees(err):.3f} deg"
                    )
                    np.testing.assert_allclose(
                        cmd,
                        q_exact,
                        atol=1e-6,
                        err_msg=f"{side} axis {axis} {deg:+.0f} order {order} signs {signs}",
                    )
                    checked += 1
    assert checked >= 18, f"only {checked} unclipped cases exercised"

    # Pure tool roll (controller x, identity correspondence) must remain a
    # pure roll-joint motion at ANY anchor: W_a Rx(t) = Rz Ry Rx(r+t).
    for side in ("left", "right"):
        w = WristClutch(side, signs=(1.0, 1.0, 1.0), axis_order=(0, 1, 2))
        assert w.engage(wrist_anchor, anchor_quat)
        moved = _rot_to_quat_wxyz(rot_anchor * sRot.from_euler("x", np.radians(20.0)))
        for _ in range(300):
            cmd = w.step(moved)
        np.testing.assert_allclose(cmd[:2], wrist_anchor[:2], atol=1e-6)
        assert abs(cmd[2] - (wrist_anchor[2] + np.radians(20.0))) < 1e-6


def check_wrist_limit_clamp() -> None:
    """A wrist driven past its limits must park AT the limit (anchored
    mapping -> no grinding) -- including the ASYMMETRIC left roll range.
    Each drive gets a FRESH neutral anchor (the anchor accumulates: an
    anchor left at a limit would offset the next drive).

    ``full_pose_ik=False`` for the same reason as the direction check: it
    asserts the *mapping's* clamp. A solved wrist parking at its bound
    without grinding is a different (also tested) property."""
    limits = WristClutch("left").limits
    # (wrist idx, rot axis, drive deg) -- stay under 180: past it the
    # differential takes the shortest-path sign and parks the OTHER limit
    # (correct behaviour, different assertion)
    for idx, axis, deg in ((0, "z", 150.0), (0, "z", -150.0), (2, "x", -120.0), (2, "x", 120.0)):
        arm = _engaged_arm(
            "left", full_pose_ik=False, wrist_signs={"left": (1, 1, 1), "right": (1, 1, 1)}
        )
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
    arm = ClutchArmV3(
        "left", full_pose_ik=False, wrist_signs={"left": (1, 1, 1), "right": (1, 1, 1)}
    )
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
    arm = _engaged_arm(
        "left", full_pose_ik=False, wrist_signs={"left": (1, 1, 1), "right": (1, 1, 1)}
    )
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
    """The v2 freeze guard (shared via _gate_q_new) must gate BOTH solves:
    sustained solver-cap steps latch after the persistence window, a
    diverged/NaN solve latches instantly, and the arm resumes when the
    solver recovers.

    Run over both paths -- the 4-DoF one it was written for and the
    full-pose one that ships -- because the guard is only as good as the
    return value it reads, and mink returns ``(q, IKResult)`` through the
    same seam as DLS.
    """
    for full_pose_ik, solve_name in ((False, "solve_position_free4"), (True, "solve")):
        arm = ClutchArmV3(
            "right",
            full_pose_ik=full_pose_ik,
            target_slew_m_s=0.0,
            wrist_signs={"left": (1, 1, 1), "right": (1, 1, 1)},
        )
        pose = np.zeros(7)
        pose[3] = 1.0
        pose[:3] = arm.solver.fk(arm.q)[0]
        assert arm.engage(pose)
        orig = getattr(arm.solver, solve_name)

        def flip_solve(q_seed, *_a, **_k):
            return np.asarray(q_seed, np.float64) + 0.5, IKResult(0.01, 0.0, 2, 0)

        setattr(arm.solver, solve_name, flip_solve)
        for _ in range(arm.hold_ticks - 1):
            arm.step(True, pose)
        assert not arm.holding, "persistence window: must not latch on the first spike"
        q_mid = arm.step(True, pose)[0]
        assert arm.holding, "sustained solver-cap steps must latch"
        q_a = arm.step(True, pose)[0]
        np.testing.assert_array_equal(q_a, q_mid, err_msg="latched: resend the last command")

        def lost_solve(q_seed, *_a, **_k):
            return np.asarray(q_seed, np.float64), IKResult(0.9, 0.0, 2, 0)

        setattr(arm.solver, solve_name, lost_solve)
        q_b = arm.step(True, pose)[0]
        assert arm.holding and np.array_equal(q_b, q_a), "diverged solve must hold instantly"

        setattr(
            arm.solver,
            solve_name,
            lambda _s, *_a, **_k: (np.full(7, np.nan), IKResult(9.9, 9.9, 0, 0)),
        )
        q_c = arm.step(True, pose)[0]
        assert arm.holding and np.array_equal(q_c, q_a), "NaN solve must hold instantly"

        setattr(arm.solver, solve_name, orig)
        arm.disengage()
        assert arm.engage(pose)
        q, _ = arm.step(True, pose)
        assert not arm.holding, "resume after the solver recovers"
        np.testing.assert_allclose(q, q_a, atol=1e-9)


def check_torso_keepout_v3() -> None:
    """The inherited keep-out sandwich still holds with the default 4-DoF
    solve (the block was copied into ClutchArmV3.step, and it acts on the
    TARGET, so the wrist being scripted cannot route around it): a
    cross-body sweep slides along the surface, latches past the limit, and
    resumes on pullback. Scenario identical to v2's check_torso_keepout."""
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
    identical to a straight skeleton (the SMPL channel is closed).

    Two independent tickers are driven in lockstep, one with each skeleton.
    A single ticker stepped straight-then-twisted would also compare two
    DIFFERENT tick indices: with the posture term enabled the mink solve
    keeps settling its null space by ~1e-7 rad per tick even at a constant
    target, so the comparison would be order-sensitive rather than
    SMPL-sensitive.
    """
    lcp = np.zeros(7)
    lcp[3] = 1.0
    lcp[:3] = IncrementalTeleopTickerV3().arms["left"].solver.fk(ARM_NEUTRAL["left"])[0]
    base = {
        "trigger": np.array([0.0, 0.0]),
        "grip": np.array([0.9, 0.0]),  # the clutch is the GRIP (2026-10 swap)
        "lctrl_pose": lcp,
        "rctrl_pose": np.zeros(7),
        "t": 100.0,
    }
    ticks = {}
    for tag, j24 in (("straight", _twist_j24(0.0)), ("twisted", _twist_j24(25.0))):
        tick = IncrementalTeleopTickerV3()
        # the grip LPF needs several ticks to cross the 0.5 clutch threshold;
        # one tick would leave grip_f at 0.12 and the SMPL comparison vacuous
        for k in range(10):
            tick.step(
                dict(base, smpl_joints=_pack_j24(j24)),
                now=100.01 + k * DT,
            )
        assert tick.arms["left"].engaged, "grip warm-up failed to engage the left arm"
        ticks[tag] = tick
    out = {}
    for tag, j24 in (("straight", _twist_j24(0.0)), ("twisted", _twist_j24(25.0))):
        f = dict(base, smpl_joints=_pack_j24(j24))
        out[tag] = ticks[tag].step(f, now=100.21)
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
    """Stale frame -> hand channel 0 = hands OPEN (the v3 inverted semantic);
    --stale-grip-close holds the last filtered trigger instead. The check
    name is historical: the hand signal is now the TRIGGER (2026-10 swap)."""
    tick = IncrementalTeleopTickerV3()
    frame = {
        "trigger": np.array([1.0, 1.0]),  # the hand channel (2026-10 swap)
        "grip": np.array([0.9, 0.9]),  # the clutch channel
        "lctrl_pose": np.zeros(7),
        "rctrl_pose": np.zeros(7),
        "smpl_joints": np.zeros(72),
        "t": 100.0,
    }
    tick.step(frame, now=100.01)
    assert tick.hand_f["l"] > 0.0, "trigger LPF must have started rising"
    st = tick.step(frame, now=100.01 + 5.0)  # way past stale_s
    assert not st["fresh"]
    assert st["hand"]["l"] == 0.0 and st["hand"]["r"] == 0.0, "stale must OPEN the hands"
    tick2 = IncrementalTeleopTickerV3(stale_grip_close=True)
    tick2.step(frame, now=100.01)
    st2 = tick2.step(frame, now=100.01 + 5.0)
    assert st2["hand"]["l"] > 0.0, "--stale-grip-close must hold the last hand"


def _full_pose_drive(side: str, axis: str, deg: float, ticks: int = 500):
    """Anchor at neutral, then command one controller rotation for `ticks`.

    Returns (arm, pose, R_target, p_target, q_trace). Pure -- the opt-in
    ``full_pose_ik=True`` path, no monkeypatching.
    """
    arm = ClutchArmV3(side, full_pose_ik=True)
    pose = np.zeros(7)
    pose[3] = 1.0
    p0, R0 = arm.solver.fk(arm.q)
    pose[:3] = p0
    assert arm.engage(pose)
    pose[3:] = _quat_axis(deg, axis)
    trace = [arm.step(True, pose)[0] for _ in range(ticks)]
    R_target = R0 @ sRot.from_euler(axis, np.radians(deg)).as_matrix()
    return arm, pose, R_target, p0, np.asarray(trace)


def _ee_rot_err_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R_a.T @ R_b) - 1.0) / 2.0, -1.0, 1.0))))


def check_full_pose_ik_tracks_orientation() -> None:
    """The opt-in 7-DoF path: the hand does what the operator's wrist does.

    The thing the default 4-DoF split does not do (there the wrist is
    scripted by WristClutch). Two halves, both on
    ``full_pose_ik=True``, both on the right arm:

    (1) A 90 deg yaw of the controller, held at a fixed position: the
        end-effector rotation must land within 3 deg of the commanded one
        and 2 mm of the target. Measured 0.0 deg / 0.35 mm at the
        2026-10-27 bent-elbow neutral -- the 3 deg gate is slack on
        purpose, this is a regression guard on the whole chain
        (anchor -> target synthesis -> clip), not a solver figure.
    (2) A 60 deg pitch, which PINS ``wrist_pitch`` at its +0.558 rad
        bound: the unsaturated rotation residual must be left standing
        (bounded, ~3.6 deg at the bent-elbow neutral) instead of being
        bought with position. Both task gains are 1.0 on this path (equal
        weighting by design), so the saturated residual is split between
        the two tasks instead of the rotation giving way to position:
        measured ~2.3 mm of position error against ~3.6 deg of rotation
        residual. The 20 mm gate is a regression guard on that split, well
        above both measurements and far below the 50.9 mm
        ``ConfigurationLimit`` failure mode (see the x2_arm_mink module
        docstring). The 500-tick drive (10 s at 50 Hz) is chosen so the
        bounded per-tick step has converged: at the bent-elbow neutral the
        first 200 ticks still have ~0.04 rad of travel left in the last-20
        window, while at 500 the step is at machine precision. The last 20
        ticks must also be settled, not grinding against the bound.

    and in both halves every tick must keep q inside
    ``X2_ARM_JOINT_LIMITS_RAD`` -- the end-to-end statement of "the limits
    are still enforced" now that the solver, not the legacy loop, proposes
    the step.
    """
    lo = np.asarray(X2_ARM_JOINT_LIMITS_RAD, float)[7:14, 0]
    hi = np.asarray(X2_ARM_JOINT_LIMITS_RAD, float)[7:14, 1]

    # (1) trackable rotation: followed, and inside the catalog all the way
    arm, _pose, R_t, p_t, trace = _full_pose_drive("right", "z", 90.0)
    out = np.max(np.maximum(lo - trace, trace - hi))
    assert out <= 0.0, f"a tick left the joint catalog by {out:.3e} rad"
    assert not arm.holding, "a reachable rotation must not trip the freeze guard"
    p, R = arm.solver.fk(arm.q)
    rot_err = _ee_rot_err_deg(R_t, R)
    assert rot_err < 3.0, f"orientation tracked to {rot_err:.2f} deg"
    pos_err = float(np.linalg.norm(p - p_t))
    assert pos_err < 0.002, f"hand left the target by {pos_err * 1e3:.2f} mm"

    # (2) saturated rotation: the limit holds, position is NOT sacrificed
    arm, _pose, R_t, p_t, trace = _full_pose_drive("right", "y", 60.0)
    out = np.max(np.maximum(lo - trace, trace - hi))
    assert out <= 0.0, f"a tick left the joint catalog by {out:.3e} rad"
    pinned = np.isclose(arm.q[5], hi[5], atol=1e-9)
    assert pinned, f"wrist_pitch should pin at {hi[5]:.3f} rad, got {arm.q[5]:.3f}"
    assert not arm.holding, "a pinned wrist is a legal pose, not a freeze"
    p, R = arm.solver.fk(arm.q)
    rot_err = _ee_rot_err_deg(R_t, R)
    assert rot_err < 35.0, f"rotation residual must stay bounded, got {rot_err:.2f} deg"
    pos_err = float(np.linalg.norm(p - p_t))
    assert pos_err < 0.020, (
        f"equal-weight saturated split leaked position: {pos_err * 1e3:.1f} mm "
        "(measured ~12 mm; the ConfigurationLimit failure mode is ~51 mm)"
    )
    settled = float(np.max(np.abs(trace[-1] - trace[-20])))
    assert settled < 0.02, f"grinding against the bound: {settled:.4f} rad over 20 ticks"


def check_trigger_hand_grip_clutch() -> None:
    """Role contract after the 2026-10 swap: the TRIGGER drives the OmniHand
    (``hand``) and the GRIP is the per-arm clutch (``held``). The constants
    and flags kept their historical names; only the signal sources moved.

    Phase A: triggers pressed, grips released -- neither arm may engage while
    the filtered trigger rises. Phase B: triggers released, grips pressed --
    both arms engage off the filtered grips and the hand channel decays.
    """
    tick = IncrementalTeleopTickerV3()
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


def check_default_4dof_and_equal_full_pose_weights() -> None:
    """Pin the split/weight contract requested for v3:

    * the shipped default is the v1-style 4-DoF position solve + WristClutch
      direct fill, so both arms report ``full_pose_ik=False`` and the solver
      carries no orientation weight (WristClutch owns orientation);
    * the opt-in full-pose path keeps the position/rotation GAINS equal
      (1.0 / 1.0), but the mink backend ships the 2026-10-30 priority
      tuning: ``position_cost=1.5`` (mink caps ``gain`` at 1.0, so the cost
      is the only way to raise position) and orientation per-axis costs
      ``(1, 0.7, 1)`` -- wrist PITCH yields to position, roll/yaw stay
      first-class. The DLS backend (the PC2 node) has no such knobs and
      keeps the equal-gain behaviour, so the mink-only pins are skipped
      there.
    """
    tick = IncrementalTeleopTickerV3()
    for side, arm in tick.arms.items():
        assert not arm.full_pose_ik, f"{side}: default must be the 4-DoF split"
        assert (
            arm.solver.rotation_weight == 0.0
        ), f"{side}: 4-DoF path must not carry an orientation task weight"
        if getattr(arm.solver, "backend", "dls") == "mink":
            assert arm.solver.position_cost == 1.5, (
                f"{side}: mink position_cost must be raised to 1.5, "
                f"got {arm.solver.position_cost}"
            )
    full = ClutchArmV3("left", full_pose_ik=True)
    assert full.full_pose_ik
    assert full.solver.position_weight == 1.0
    assert full.solver.rotation_weight == 1.0, "full-pose path must keep the gains equal"
    if getattr(full.solver, "backend", "dls") == "mink":
        assert full.solver.position_cost == 1.5
        w = np.asarray(full.solver.orientation_axis_weights)
        assert np.allclose(w, (1.0, 0.7, 1.0)), f"orientation axis weights: {w}"
        assert w[1] < min(w[0], w[2]), "wrist pitch must yield to roll/yaw"


def check_ctrl_lever_pure_wrist_rotation() -> None:
    """A controller rotating about a stationary wrist must not move the target.

    Synthetic physical controller: its origin rides a lever ``r_true`` from
    the wrist centre, ``C = W + R @ r_true``. Sweeping ``R`` is exactly the
    "human shoulder/elbow still, wrist rotates" motion from the live report.
    With ``--ctrl-lever = r_true`` the compensated target is invariant; with
    the legacy estimate (0,0,0) it sweeps by centimetres -- the non-vacuity
    control that shows this check can fail. Also pins the per-side routing:
    a single vector feeds both arms, a mapping keeps the two calibrated
    levers apart (they differ by tens of mm in practice).
    """
    r_true = np.array([0.0, 0.02, 0.09])
    R0 = sRot.from_euler("ZYX", [0.4, -0.3, 0.2])
    sweeps = (
        sRot.from_euler("x", 0.25),
        sRot.from_euler("y", -0.2),
        sRot.from_euler("z", 0.3),
        sRot.from_euler("y", 0.5) * sRot.from_euler("x", -0.35),
    )

    def run(lever_est: np.ndarray) -> tuple[float, float]:
        arm = ClutchArmV3(
            "left",
            position_scale=1.0,
            torso_keepout=None,
            target_slew_m_s=0.0,
            ctrl_lever=lever_est,
        )
        p0, _ = arm.solver.fk(arm.q)
        c0 = np.concatenate([p0 + R0.apply(r_true), _rot_to_quat_wxyz(R0)])
        assert arm.engage(c0), "engage refused a valid synthetic controller pose"
        p_anchor = arm.solver.fk(arm.q)[0].copy()
        q_anchor = arm.q.copy()
        worst_tgt = 0.0
        worst_up = 0.0
        for dR in sweeps:
            R = dR * R0
            c = np.concatenate([p0 + R.apply(r_true), _rot_to_quat_wxyz(R)])
            arm.step(True, c)
            worst_tgt = max(worst_tgt, float(np.linalg.norm(arm._tgt_slewed - p_anchor)))
            # upstream joints only: the reported symptom is shoulder/elbow
            # being dragged by a wrist-only rotation
            worst_up = max(worst_up, float(np.linalg.norm(arm.q[:4] - q_anchor[:4])))
        return worst_tgt, worst_up

    fixed, fixed_q = run(r_true)
    legacy, legacy_q = run(np.zeros(3))
    assert (
        fixed < 1e-6
    ), f"lever-corrected target moved {fixed * 1e3:.3f} mm under a wrist-only sweep"
    assert legacy > 5e-3, (
        "the legacy controller-origin target barely moved "
        f"({legacy * 1e3:.3f} mm); the check would be vacuous"
    )
    assert fixed_q < 0.01, (
        f"lever-corrected upstream joints moved {np.degrees(fixed_q):.2f} deg "
        "under a wrist-only sweep"
    )
    assert legacy_q > 0.05, (
        f"legacy upstream joints barely moved ({np.degrees(legacy_q):.1f} deg); "
        "the drift assertion would be vacuous"
    )
    # Per-side routing. The controllers are not held in mirrored body-frame
    # orientations, so the fitted vectors differ (the live fit was
    # +55.5 mm vs -4.6 mm in x); handing one arm the other's vector would
    # leave a ~60 mm error that this feature exists to remove.
    other = np.array([-0.01, -0.03, 0.07])
    paired = IncrementalTeleopTickerV3(ctrl_lever={"left": r_true, "right": other})
    assert np.allclose(paired.arms["left"].ctrl_lever, r_true)
    assert np.allclose(
        paired.arms["right"].ctrl_lever, other
    ), "per-side ctrl_lever was not routed to the right arm"
    common = IncrementalTeleopTickerV3(ctrl_lever=r_true)
    assert np.allclose(common.arms["left"].ctrl_lever, r_true)
    assert np.allclose(
        common.arms["right"].ctrl_lever, r_true
    ), "a single ctrl_lever must apply to both arms"


def check_operator_signs() -> None:
    """3-point wire x is operator-FORWARD and +y is operator-LEFT.

    ``fixed_operator_frame_from_controllers`` builds ``fwd = cross(lat, up)``
    from the SDK world, which the PC service delivers RIGHT-handed (X right,
    Y up, Z in; its README also notes the real data flips Unity's +Z).
    The left controller sits at -X, so that cross product is -Z = operator
    FORWARD.  The wire's +y (operator left) and +z (up) are right in the
    operator's own frame: wire = (forward, left, up).

    The operator and the robot face the SAME direction (2026-10-30 live
    correction -- NOT face-to-face) and the X2 base is also (forward, left,
    up), so the avatar map is the IDENTITY:

        wire x INCREASE (operator forward) -> robot +x  => operator_x_sign=+1
        wire y INCREASE (operator left)    -> robot +y  => operator_y_sign=+1

    (the 2026-10-27 x=-1 round read the Unity figure instead of the
    delivered data and made the real robot reach BACKWARD).  Pin:

    (A) a wire x INCREASE (operator forward) must reach the robot as +x with
        +1, and as -x with -1 (the live "hand forward, robot backward" bug);
        the x flip must leave y/z alone, and the y flip must leave x/z alone;
    (B) a fronto-parallel circle: under the same-facing avatar map the
        robot's (y, z) traversal must have the SAME sense as the wire's own
        (y_sign=+1); -1 mirrors it, which is only right for a face-to-face
        rig (the 2026-10-30 y=-1 round made the real robot's lateral motion
        and circle direction come out mirrored);
    (C) the flips happen AFTER the lever correction: a pure wrist rotation
        with a calibrated lever still leaves the target invariant, because
        ``C`` and ``R_c @ r_c`` stay in the same frame.
    """
    pose0 = np.zeros(7, np.float64)
    pose0[3] = 1.0  # unit quat (wxyz): engage() refuses zero placeholders

    def make_arm(x_sign: float = 1.0, y_sign: float = 1.0, lever=None) -> ClutchArmV3:
        return ClutchArmV3(
            "left",
            position_scale=1.0,
            torso_keepout=None,
            target_slew_m_s=0.0,
            position_deadband_m=0.0,
            posture_gain=0.0,
            ctrl_lever=lever,
            operator_x_sign=x_sign,
            operator_y_sign=y_sign,
        )

    p0, _ = make_arm().solver.fk(ARM_NEUTRAL["left"])
    pose0[:3] = p0
    # operator forward = wire x INCREASE; +y = operator left; +z = up
    d_wire = np.array([0.06, 0.05, 0.04])

    def target_delta(arm, d: np.ndarray) -> np.ndarray:
        assert arm.engage(pose0), "engage refused a valid synthetic pose"
        a_op, _, a_rob, _ = arm.anchor
        moved = pose0.copy()
        moved[:3] = pose0[:3] + d
        return arm._compensated_target(moved, a_op, a_rob) - a_rob

    for x_sign, want_x in ((1.0, +1.0), (-1.0, -1.0)):
        delta = target_delta(make_arm(x_sign=x_sign), d_wire)
        assert delta[0] * want_x > 0.5 * abs(d_wire[0]), (
            f"operator_x_sign={x_sign:+.0f}: wire forward (dx={d_wire[0]:+.3f}) "
            f"reached the robot as x={delta[0]:+.3f}; expected {want_x:+.0f}"
        )
        assert (
            abs(delta[1] - d_wire[1]) < 1e-12 and abs(delta[2] - d_wire[2]) < 1e-12
        ), f"operator_x_sign={x_sign:+.0f} leaked into lateral/vertical: {delta}"
    for y_sign, want_y in ((1.0, +1.0), (-1.0, -1.0)):
        delta = target_delta(make_arm(y_sign=y_sign), d_wire)
        assert delta[1] * want_y > 0.5 * abs(d_wire[1]), (
            f"operator_y_sign={y_sign:+.0f}: wire lateral (dy={d_wire[1]:+.3f}) "
            f"reached the robot as y={delta[1]:+.3f}; expected {want_y:+.0f}"
        )
        assert (
            abs(delta[0] - d_wire[0]) < 1e-12 and abs(delta[2] - d_wire[2]) < 1e-12
        ), f"operator_y_sign={y_sign:+.0f} leaked into forward/vertical: {delta}"

    # (B) circle direction is the SIGNED AREA in the (y, z) plane.  The
    # operator faces the SAME direction as the robot, so the robot must
    # reproduce the wire's own traversal (y_sign=+1); -1 mirrors it, which
    # is only correct for a face-to-face rig.
    phis = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    circle = np.stack([np.zeros_like(phis), 0.06 * np.cos(phis), 0.06 * np.sin(phis)], axis=1)
    area_wire = 0.5 * float(
        np.sum(circle[:-1, 1] * circle[1:, 2] - circle[:-1, 2] * circle[1:, 1])
    )
    areas = {}
    for y_sign in (1.0, -1.0):
        arm = make_arm(y_sign=y_sign)
        assert arm.engage(pose0)
        a_op, _, a_rob, _ = arm.anchor
        d = np.array(
            [
                arm._compensated_target(np.concatenate([pose0[:3] + w, pose0[3:]]), a_op, a_rob)
                - a_rob
                for w in circle
            ]
        )
        areas[y_sign] = 0.5 * float(np.sum(d[:-1, 1] * d[1:, 2] - d[:-1, 2] * d[1:, 1]))
    assert areas[1.0] * area_wire > 0.0, (
        f"same-facing avatar map must keep the wire's circle sense: wire "
        f"area {area_wire:+.2e}, y_sign=+1 -> {areas[1.0]:+.2e}"
    )
    assert areas[-1.0] * area_wire < 0.0, (
        f"y_sign=-1 must mirror the circle (face-to-face rig): wire area "
        f"{area_wire:+.2e}, y_sign=-1 -> {areas[-1.0]:+.2e}"
    )

    # (C) ordering: the wire position and quaternion share the OLD frame, so
    # the mirror must be applied to their combined output, after the lever.
    r_true = np.array([0.0, 0.02, 0.09])
    R0 = sRot.from_euler("ZYX", [0.4, -0.3, 0.2])
    for x_sign, y_sign in ((1.0, 1.0), (-1.0, -1.0)):
        arm = make_arm(x_sign, y_sign, lever=r_true)
        pp, _ = arm.solver.fk(arm.q)
        c0 = np.concatenate([pp + R0.apply(r_true), _rot_to_quat_wxyz(R0)])
        assert arm.engage(c0)
        a_op, _, a_rob, _ = arm.anchor
        for dR in (
            sRot.from_euler("x", 0.25),
            sRot.from_euler("y", -0.2),
            sRot.from_euler("z", 0.3),
        ):
            R = dR * R0
            c = np.concatenate([pp + R.apply(r_true), _rot_to_quat_wxyz(R)])
            tgt = arm._compensated_target(c, a_op, a_rob)
            assert np.linalg.norm(tgt - a_rob) < 1e-9, (
                f"operator signs ({x_sign:+.0f},{y_sign:+.0f}): mirroring before "
                f"the lever correction moved the target "
                f"{np.linalg.norm(tgt - a_rob) * 1e3:.3f} mm"
            )


def check_position_deadband() -> None:
    """Soft radial deadband: inside r the target is the anchor; outside it
    passes through with unit slope, offset by exactly r."""
    r = 0.005
    arm = ClutchArmV3(
        "left",
        position_scale=1.0,
        torso_keepout=None,
        target_slew_m_s=0.0,
        position_deadband_m=r,
    )
    p0, _ = arm.solver.fk(arm.q)
    c0 = np.concatenate([p0, [1.0, 0.0, 0.0, 0.0]])
    assert arm.engage(c0)
    p_anchor = arm.solver.fk(arm.q)[0].copy()

    def target_after(dx: float) -> np.ndarray:
        c = c0.copy()
        c[:3] = p_anchor + np.array([dx, 0.0, 0.0])
        arm.step(True, c)
        return arm._tgt_slewed.copy()

    for dx in (0.0, 0.001, 0.003, 0.0049):
        got = target_after(dx)
        assert (
            np.linalg.norm(got - p_anchor) < 1e-9
        ), f"deadband leaked {np.linalg.norm(got - p_anchor) * 1e3:.3f} mm at dx={dx * 1e3:.1f} mm"
    for dx in (0.010, 0.020):
        got = target_after(dx)
        want = p_anchor + np.array([dx - r, 0.0, 0.0])
        assert np.allclose(got, want, atol=1e-9), (
            f"deadband pass-through wrong at dx={dx * 1e3:.1f} mm: "
            f"{((got - p_anchor) * 1e3).round(4)} mm"
        )


def check_posture_gain_null_space() -> None:
    """The optional posture term moves the redundant swivel, not the wrist.

    Drive a seed displaced along the position Jacobian's null space with a
    fixed target: gain 0 must leave the seed where it is; gain 0.05 must pull
    it toward the preferred posture while holding the wrist point. Runs on
    every importable backend (mink's PostureTask and DLS's exact-projector
    null-space bias).
    """
    q_pref = ARM_NEUTRAL["left"].copy()
    wrist = q_pref[4:].copy()

    def fd_j4(solver, q: np.ndarray) -> np.ndarray:
        J = np.zeros((3, 4), np.float64)
        p0, _ = solver.fk(q)
        eps = 1e-7
        for i in range(4):
            qp = q.copy()
            qp[i] += eps
            J[:, i] = (solver.fk(qp)[0] - p0) / eps
        return J

    for backend in ("dls", "mink"):
        try:
            probe, _ = make_arm_solver("left", backend=backend)
        except RuntimeError:
            if backend == "dls":
                raise
            continue  # mink not installed: nothing to test on this backend
        J4 = fd_j4(probe, q_pref)
        u = np.linalg.svd(J4)[2][-1]
        seed = q_pref.copy()
        seed[:4] += 0.15 * u
        target, _ = probe.fk(seed)

        def settle(gain: float):
            solver, _ = make_arm_solver(
                "left",
                backend=backend,
                posture_gain=gain,
                q_preferred=q_pref,
            )
            q = seed.copy()
            for _ in range(60):
                q, _ik = solver.solve_position_free4(q, wrist, target, max_iters=1)
            return q, float(np.linalg.norm(solver.fk(q)[0] - target))

        q0, err0 = settle(0.0)
        q1, err1 = settle(0.05)
        d0 = float(np.linalg.norm(q0[:4] - q_pref[:4]))
        d1 = float(np.linalg.norm(q1[:4] - q_pref[:4]))
        assert d0 > 0.1, (
            f"{backend}: null-space probe did not start far enough from the "
            f"preferred posture ({d0:.4f} rad)"
        )
        assert d1 < 0.5 * d0, (
            f"{backend}: posture gain barely moved the swivel " f"({d0:.4f} -> {d1:.4f} rad)"
        )
        assert err0 < 1e-6, f"{backend}: gain-0 solve drifted off the target ({err0 * 1e3:.4f} mm)"
        assert err1 < 1e-3, (
            f"{backend}: posture term traded away the wrist point " f"({err1 * 1e3:.4f} mm)"
        )


def check_3pt_controller_frame() -> None:
    """v4 trackerless wire contract (Pico headset + two controllers, no body).

    A 3point sender fills ONLY ``t/lctrl_pose/rctrl_pose/trigger/grip``,
    forces wire ``engaged=0``, and sends placeholder ``smpl_joints`` /
    ``human_quat``. Pin: (A) grip still engages the clutch and trigger still
    drives the hand; (B) the SMPL placeholders never influence the arms (two
    tickers in lockstep, identical wires, different placeholders -> identical
    q); (C) moving the controller wire moves the commanded arm.
    """

    def make_ticker():
        return IncrementalTeleopTickerV3(
            torso_keepout=None,
            position_deadband_m=0.0,
            posture_gain=0.0,
            target_slew_m_s=0.0,
        )

    a, b = make_ticker(), make_ticker()
    now = 100.0

    def pose(x, y, z):
        out = np.zeros(7, np.float32)
        out[:3] = (x, y, z)
        out[3] = 1.0  # unit quat (wxyz): engage() refuses zero placeholders
        return out

    def frame(lx, ly, lz, rx, ry, rz, smpl, hq):
        return {
            "t": now,
            "trigger": np.array([1.0, 1.0], np.float32),
            "grip": np.array([1.0, 1.0], np.float32),
            "lctrl_pose": pose(lx, ly, lz),
            "rctrl_pose": pose(rx, ry, rz),
            "engaged": np.array([0.0]),  # 3point forces this on the wire
            "smpl_joints": smpl,
            "human_quat": hq,
        }

    zeros72 = np.zeros(72)
    identity_wxyz = np.array([1.0, 0.0, 0.0, 0.0])

    # A+B. Lockstep: same trackerless wires, different SMPL placeholders.
    for k in range(20):  # 0.4 s: past the grip/trigger LPFs, inside stale_s
        st = a.step(frame(0.20, 0.0, 0.0, -0.20, 0.0, 0.0, zeros72, identity_wxyz), now)
        st_b = b.step(
            frame(
                0.20,
                0.0,
                0.0,
                -0.20,
                0.0,
                0.0,
                np.full(72, 0.7 + 0.001 * k),
                np.array([0.0, 0.0, 0.0, 1.0]),
            ),
            now,
        )
        now += DT
    assert st["fresh"]
    assert st["held"]["left"] and st["held"]["right"]
    assert a.arms["left"].engaged and a.arms["right"].engaged
    assert st["hand"]["l"] > 0.5 and st["hand"]["r"] > 0.5
    for side in st["q"]:
        assert np.allclose(
            st["q"][side], st_b["q"][side], rtol=0.0, atol=1e-9
        ), f"{side}: placeholder SMPL fields leaked into the arm command"
    base = {side: q.copy() for side, q in st["q"].items()}

    # C. A controller-wire translation must move the commanded arm.
    for _ in range(40):  # 0.8 s
        st = a.step(frame(0.32, 0.10, 0.02, -0.20, 0.0, 0.0, zeros72, identity_wxyz), now)
        now += DT
    moved = float(np.abs(st["q"]["left"] - base["left"]).max())
    assert moved > 0.05, f"left arm ignored the wire delta ({moved:.4f} rad)"
    for q in st["q"].values():
        assert np.all(np.isfinite(q))


def check_wrist_default_signs_side_parallel() -> None:
    """Default wrist correspondence is the SAME on both sides (live under test).

    2026-10-21 v4 3-point MuJoCo sessions.  Session 1 (left signs the exact
    negation of the right, the old "left/right mirror") showed the left yaw
    and pitch reversed while the right followed the hand.  Session 2 with
    both sides at (1, -1, 1) fixed the left but left the RIGHT pitch wrong,
    so the current experiment is (1, 1, 1) on both sides: controller +y ->
    +pitch.  The X2 wrist tool frames are NOT mirrored (at the zero pose
    both ``wrist_roll`` axes are world +x and the yaw/pitch axes are world
    +z/+y, ``x2_ultra.xml``), so whatever the final triple is, it must be
    the SAME on both sides.

    Pin (zero wrist anchor, so the ZYX decomposition is single-axis): the
    same controller body-frame rotation commands the same wrist joints on
    the left and the right, and +z -> +yaw, +y -> +pitch (A/B under test),
    +x -> +roll.
    """
    theta = 0.1  # rad; below the 0.12 rad/tick slew so the sign is readable
    axes = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
    # axis -> (wrist joint index, expected sign): controller z->yaw +,
    # y->pitch + (current A/B), x->roll +
    specs = {"x": (2, 1.0), "y": (1, 1.0), "z": (0, 1.0)}
    cmds = {}
    for side in ("left", "right"):
        wc = WristClutch(side)
        assert wc.engage(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
        for name, axis in axes.items():
            dq = _rot_to_quat_wxyz(sRot.from_rotvec(theta * np.asarray(axis)))
            cmd = wc.step(dq)
            col, sign = specs[name]
            assert cmd[col] * sign > 0.5 * theta, (
                f"{side} {name}: wrist delta {np.round(cmd, 4)} has the wrong "
                f"sign on joint {col} (expected {sign:+.0f})"
            )
            cmds[(side, name)] = cmd
            wc.cmd[:] = 0.0  # back to the zero anchor for the next axis
            wc.last_delta_ypr[:] = 0.0
    for name in axes:
        assert np.allclose(
            cmds[("left", name)], cmds[("right", name)], rtol=0.0, atol=1e-12
        ), f"{name}: default wrist signs differ between sides (old mirror bug)"


def check_arm_home_long_press() -> None:
    """X/A long-press -> bounded arm home (v4 3-point contract).

    The wire carries a 2-vector ``home`` request; the sender turns one
    >=3 s face-button hold into a short pulse. Pin: a rising edge starts
    homing for that side ONLY, releases its clutch, and moves every joint
    toward ARM_NEUTRAL by at most ``home_speed_rad_s * dt`` per tick; on
    completion the pose is exactly neutral, a still-held grip re-anchors
    there without a jump, and a request level that stays high does not
    restart the motion.
    """

    def pose(x, y, z):
        out = np.zeros(7, np.float64)
        out[:3] = (x, y, z)
        out[3] = 1.0  # unit quat (wxyz): engage() refuses zero placeholders
        return out

    def frame(now, home, grip):
        return {
            "t": now,
            "trigger": np.array([0.0, 0.0]),
            "grip": np.asarray(grip, np.float64),
            "lctrl_pose": pose(0.20, 0.0, 0.10),
            "rctrl_pose": pose(-0.20, 0.0, 0.10),
            "home": np.asarray(home, np.float64),
        }

    ticker = IncrementalTeleopTickerV3(
        torso_keepout=None, position_deadband_m=0.0, posture_gain=0.0
    )
    neutral = {s: np.asarray(ARM_NEUTRAL[s], np.float64).copy() for s in ("left", "right")}
    right0 = ticker.arms["right"].q.copy()

    # Deflect the left arm and engage it, so the request must both release
    # the clutch and slew a non-neutral pose home.
    q_deflected = neutral["left"] + np.array([0.40, -0.30, 0.20, 0.10, 0.20, -0.15, 0.10])
    ticker.arms["left"].q = q_deflected.copy()
    ticker.arms["left"].wrist.cmd[:] = q_deflected[4:7]
    assert ticker.arms["left"].engage(pose(0.20, 0.0, 0.10))
    assert ticker.arms["left"].engaged

    now = 100.0
    max_step = ticker.home_speed_rad_s * ticker.dt
    # Rising edge: home starts, the clutch releases, the OTHER arm holds.
    st = ticker.step(frame(now, home=(1.0, 0.0), grip=(0.0, 0.0)), now)
    assert ticker.home["left"] and not ticker.home["right"]
    assert not ticker.arms["left"].engaged
    assert np.allclose(st["q"]["right"], right0, rtol=0.0, atol=0.0)
    q_prev = st["q"]["left"].copy()
    assert np.max(np.abs(q_prev - neutral["left"])) > 0.1  # still far from home

    # Bounded slew: every joint step <= home_speed * dt, monotone progress.
    ticks = 0
    while ticker.home["left"]:
        now += ticker.dt
        st = ticker.step(frame(now, home=(1.0, 0.0), grip=(0.0, 0.0)), now)
        q_now = st["q"]["left"]
        assert np.max(np.abs(q_now - q_prev)) <= max_step + 1e-9, (
            f"home joint step {np.max(np.abs(q_now - q_prev)):.4f} rad/tick "
            f"exceeds {max_step:.4f}"
        )
        assert np.max(np.abs(q_now - neutral["left"])) <= np.max(
            np.abs(q_prev - neutral["left"])
        ), "home moved away from ARM_NEUTRAL"
        q_prev = q_now.copy()
        ticks += 1
        assert ticks < 1000
    assert np.array_equal(st["q"]["left"], neutral["left"])  # exact snap
    assert np.allclose(st["q"]["right"], right0, rtol=0.0, atol=0.0)

    # Still-held grip re-anchors at the neutral pose; the request level is
    # still high, which must NOT restart homing or disengage again.
    for _ in range(40):  # 0.8 s: past the grip LPF
        now += ticker.dt
        st = ticker.step(frame(now, home=(1.0, 0.0), grip=(1.0, 0.0)), now)
    assert not ticker.home["left"]
    assert ticker.arms["left"].engaged
    assert np.max(np.abs(st["q"]["left"] - neutral["left"])) < 1e-6
    assert np.allclose(st["q"]["right"], right0, rtol=0.0, atol=0.0)


def check_reach_clamp_anchor_direction() -> None:
    """A long forward command saturates the reach sphere WITHOUT sliding the
    wrist sideways or up.

    Regression for the 2026-10-30 live report ("屈肘前伸左手时, 机器人手往
    右侧偏移"): the legacy shoulder-centred radial clamp traded the anchor's
    13.5 cm lateral offset for forward reach, so a straight forward command
    slid the wrist toward y_shoulder (robot's own right) and upward. The
    clamp must keep the anchor->target direction: pure +x stays pure +x.
    """
    for side in ("left", "right"):
        # v4 real-node signs (+1, +1): wire +x is operator FORWARD, so
        # pose[0] += dx below is an operator forward extension in the
        # robot's +x.
        arm = _engaged_arm(
            side,
            target_slew_m_s=0.0,
            position_deadband_m=0.0,
            operator_x_sign=1.0,
            operator_y_sign=1.0,
        )
        _a_op, _a_op_inv, a_rob, _a_rob_R = arm.anchor
        pose = np.zeros(7)
        pose[3] = 1.0
        pose[:3] = arm.solver.fk(arm.q)[0]
        q = arm.q.copy()
        for _ in range(60):  # 60 x 1 cm of operator forward travel
            pose[0] += 0.01
            q, _ik = arm.step(True, pose)
        tgt = arm._tgt_slewed
        r = float(np.linalg.norm(tgt - arm.shoulder))
        assert r <= arm.max_reach + 1e-9, (
            f"{side}: saturated target escaped the reach sphere ({r:.4f} > {arm.max_reach:.4f})"
        )
        fwd = float(tgt[0] - a_rob[0])
        assert fwd > 0.10, f"{side}: saturated target barely advanced ({fwd:.3f} m)"
        dy = float(tgt[1] - a_rob[1])
        dz = float(tgt[2] - a_rob[2])
        assert abs(dy) < 2e-3, (
            f"{side}: forward command slid the target laterally by {dy * 100:+.1f} cm"
        )
        assert abs(dz) < 2e-3, (
            f"{side}: forward command slid the target vertically by {dz * 100:+.1f} cm"
        )
        fk = arm.solver.fk(q)[0]
        assert abs(float(fk[1] - a_rob[1])) < 5e-3, (
            f"{side}: FK wrist drifted {float(fk[1] - a_rob[1]) * 100:+.1f} cm in y"
        )


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
    # appended, never inserted: gear_sonic/tests/test_arm_teleop_v3.py
    # indexes CHAIN_V3_CHECKS by position
    ("full_pose_ik_tracks_orientation", check_full_pose_ik_tracks_orientation),
    ("trigger_hand_grip_clutch", check_trigger_hand_grip_clutch),
    ("default_4dof_equal_full_pose_weights", check_default_4dof_and_equal_full_pose_weights),
    ("ctrl_lever_pure_wrist_rotation", check_ctrl_lever_pure_wrist_rotation),
    ("position_deadband", check_position_deadband),
    ("posture_gain_null_space", check_posture_gain_null_space),
    ("wrist_nonzero_anchor_tracks_tool_rotation", check_wrist_nonzero_anchor_tracks_tool_rotation),
    ("3pt_controller_frame", check_3pt_controller_frame),
    ("wrist_default_signs_side_parallel", check_wrist_default_signs_side_parallel),
    ("arm_home_long_press", check_arm_home_long_press),
    ("operator_signs", check_operator_signs),
    ("reach_clamp_anchor_direction", check_reach_clamp_anchor_direction),
]


def run_self_test(checks=None) -> bool:
    """Run this module's pure checks -- or a caller-composed list (the sim
    harness passes CHECKS + its plant-dependent ones). True iff all pass."""
    checks = CHECKS if checks is None else checks
    return _v2_run_self_test(checks)


if __name__ == "__main__":
    raise SystemExit(0 if run_self_test() else 1)
