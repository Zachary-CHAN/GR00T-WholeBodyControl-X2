"""mink-backed IK solver for the X2 arm.

Same public surface as :class:`~gear_sonic.utils.teleop.solver.arm.x2_arm_ik.ArmIKSolver`
(``solve`` / ``solve_position_free4`` / ``fk`` / ``reset_rotation_err_state``,
same ``(q7, IKResult)`` returns), but the per-tick step is a QP built out of
declarative mink tasks and limits instead of a hand-written damped
least-squares update plus a shell of guards.

Why this is a faithful port, and where it is not
-----------------------------------------------
mink's QP solves for the joint step ``dq`` (``solve_ik`` returns
``result.x / dt``) minimising, per task,

    ``|| cost .* (J dq - gain * e) ||^2``   plus ``damping * ||dq||^2``

which assembles to ``dq = (J^T W J + damping I)^-1 J^T W (gain e)`` with
``W = diag(cost^2)``. The legacy step is
``dq = J^T (J J^T + lambda^2 I)^-1 err`` with ``err = W' e``, so the weight has
to ride on the *residual* rather than on the cost matrix: ``gain=weight`` with
``cost=1``, never ``cost=weight``, which would also scale ``J^T W J`` against
the damping (measured against legacy on a 0.1 rad rotation step: 1.6e-3 rad
with the weight on the gain, 8.4e-2 rad with it on the cost). The mapping is
then:

    position task   : ``position_cost=1.0, orientation_cost=0.0, gain=position_weight``
    orientation task: ``position_cost=0.0, orientation_cost=1.0, gain=rotation_weight``
    damping         : ``damping=lambda^2`` (one joint-space diagonal term)
    ``lm_damping``  : 0 on both (mink's other, error-norm-adaptive damping law)

What does *not* map exactly is the error itself. mink's is a single SE(3) log
(a body twist, with the logarithm's left Jacobian premultiplied onto the task
Jacobian); the legacy error is decoupled -- ``[target_pos - p, w * axis * angle]``
in the root frame. The two share a fixed point, not a path: both drive the same
residuals to zero, but they take different steps while the residual is large.
Over the residuals the teleop actually sees the gap is small -- measured
per-step ``max |dq_mink - dq_dls|``: 1.9e-6 rad for a 10 mm pure translation,
1.8e-5 rad for a 0.1 rad pure rotation, 1.1e-4 rad at 2 mm + 0.05 rad (the
tracking regime the A/B gate checks), reaching 1.8e-2 rad only for a 50 mm
residual, i.e. a transient after the freeze guard has latched and a branch is
re-forming. That difference is the price of the migration; it is measured on a
real tape by ``tools/ab_arm_ik_backends.py`` rather than argued here.

One trap that follows from the error being an SE(3) log: a task's *rotation*
matters even when its cost is zero, because the left Jacobian mixes the angular
rows into the linear ones at that rotation's angle. Both tasks are therefore
given the same target pose, and the position-only path the arm's own rotation
-- otherwise the delivered rotation (identity, say) shows up as a first-order
error in the step: 3.0e-3 rad on a 10 mm translation versus 1.9e-6 rad with the
shared target, i.e. the difference would be the artifact rather than the solver.

``RelativeFrameTask``'s root is ``torso_link``, which is exactly the frame
``x2_arm_fk`` reports the wrist in -- the same frame the teleop's
``tgt_pos``/``tgt_rot`` already live in. No frame bridging code.

The model
---------
``VelocityLimit`` refuses a model with a free joint, so the solver prunes one
in memory from the same MJCF the plant uses (no second kinematic source):
drop the free joint, the waist joints (welding ``pelvis -> torso_link``), the
legs, the head and the other arm, plus the actuators/sensors that reference
them. That leaves ``nq = nv = 7``: one 7-DoF arm, nothing else to freeze, a
7-variable QP with 14 inequality rows. The two arms share one read-only
``MjModel`` (``lru_cache``ed) and each holds its own ``Configuration``.

Limits: the step cap goes to mink, the joint ranges stay a post-step clip
------------------------------------------------------------------------
The per-tick step cap moves to ``VelocityLimit`` at
``max_per_tick_step_rad / DT`` rad/s -- a *risk inherited on purpose*: live
round 5 concluded that clipping the step per joint twists its direction near
the elbow-extended singularity (different joints saturate at different times)
and feeds a bang-bang oscillation, which is why the legacy solver grew
``uniform_step_clip``. ``VelocityLimit`` is a per-joint box, the same shape as
the clip that was rejected. It is used anyway because as a *constraint inside
the QP* the violated joints are re-allocated over the rest of the arm instead
of the whole step being scaled down, and because 0.30 rad/tick barely
activates in normal motion (measured peak 0.167 rad/tick on a legal
turn-around) -- it only sits at the bound while a frozen branch re-forms. The
behaviour difference is measured by ``tools/ab_arm_ik_backends.py`` rather
than argued here.

Joint *ranges* were handed to ``ConfigurationLimit`` first -- it reads the
same ranges out of the MJCF that ``_check_model`` asserts equal to
``X2_ARM_JOINT_LIMITS_RAD`` -- and then taken back out again, on measurement.
Driving one start state through a 3 s / +90 deg wrist-rotation target, the two
resolutions part ways the moment ``wrist_pitch`` reaches its -0.558 rad bound:

* unconstrained step + post-clip (the legacy loop): the step lands almost
  entirely on ``wrist_pitch`` itself, the cheapest joint for a pure wrist
  rotation, and the clip discards precisely that joint. Position is held
  (2.5 mm), the saturated rotation residual is left standing (22.5 deg), peak
  step 0.0072 rad/tick.
* ``ConfigurationLimit``: ``wrist_pitch`` is pinned *inside* the QP, which
  re-allocates the same request over the arm instead
  (``dq = [0.018, -0.012, 0.010, -0.058, -0.013, 0, 0.003]``, ``|dq|`` 0.0645
  against 0.0396 unconstrained). Orientation is bought with position: the hand
  leaves the operator's target by up to 50.9 mm over the drive, peak step
  0.0230 rad/tick, and ``max|dq|`` against the DLS backend 0.562 rad -- 11x the
  A/B budget this migration is held to.

Position is the operator-visible quantity here (``position_weight`` 1.0
against 0.5) and the torso keep-out upstream is enforced on the *target*, so a
wrist lagging its target walks straight into the ellipse. (The v3 opt-in
full-pose path raises both gains to 1.0; on the same saturated drive the split
becomes ~11.8 deg rotation / ~12 mm position, both bounded. It still keeps the
post-step-clip choice below.) The QP therefore gets the step cap only, and
joint ranges are clipped after the step exactly as the legacy loop did; with
that, the two backends agree to 0.00122 rad max / 1.3e-4 RMS on the same
drive.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

try:
    import mink
    import mujoco
except ImportError as exc:  # pragma: no cover - reported by make_arm_solver
    raise ImportError(
        "MinkArmIK needs mujoco and mink (plus a QP backend such as daqp). "
        "Install them with `pip install -e 'gear_sonic[sim]'`, or run the "
        "legacy solver instead with X2_IK_BACKEND=dls."
    ) from exc

from gear_sonic.utils.teleop.solver.arm.x2_arm_fk import (
    chain_fk_full,
    left_arm_chain,
    right_arm_chain,
)
from gear_sonic.utils.teleop.solver.arm.x2_arm_ik import (
    X2_ARM_JOINT_LIMITS_RAD,
    IKResult,
    _quat_wxyz_to_matrix,
    _rotation_error_axis_angle,
)

# The 50 Hz teleop tick. Only used as the dt mink integrates with and as the
# divisor that turns the per-tick step cap into a velocity limit: the step
# itself is dt-independent (``solve_ik`` returns ``result.x / dt`` and the
# caller integrates with the same dt).
DT = 0.02

# Joint names inside one arm, in chain order. Prefix with the side.
_ARM_JOINTS: tuple[str, ...] = (
    "shoulder_pitch_joint",
    "shoulder_roll_joint",
    "shoulder_yaw_joint",
    "elbow_joint",
    "wrist_yaw_joint",
    "wrist_pitch_joint",
    "wrist_roll_joint",
)

# Body of each joint, in the same order (the chain link that owns it).
_ARM_LINKS: tuple[str, ...] = tuple(f"{n[: -len('_joint')]}_link" for n in _ARM_JOINTS)

# Trunk bodies kept (welded) so torso_link stays the task root.
_KEEP_BODIES: frozenset[str] = frozenset(
    {"world", "pelvis", "waist_yaw_link", "waist_pitch_link", "torso_link"}
)

_DEFAULT_MJCF = (
    Path(__file__).resolve().parents[4]
    / "data"
    / "assets"
    / "robot_description"
    / "mjcf"
    / "x2_ultra.xml"
)


def _mjcf_path() -> Path:
    path = _DEFAULT_MJCF
    if not path.is_file():
        raise FileNotFoundError(
            f"X2 MJCF not found at {path}. The mink backend builds its model "
            "from it in memory; pass model_path= to point somewhere else, or "
            "run the legacy solver with X2_IK_BACKEND=dls."
        )
    return path


@functools.lru_cache(maxsize=None)
def arm_only_model(side: str, model_path: str) -> mujoco.MjModel:
    """The X2 MJCF with everything but ``side``'s arm removed.

    One movable 7-DoF chain on a welded trunk: no floating base (which
    ``VelocityLimit`` rejects), no waist, no second arm, nothing else that
    could wander while the QP is busy. Cached per (side, path); the returned
    model is read-only -- solvers share it and keep their own MjData.
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    spec = mujoco.MjSpec.from_file(model_path)

    keep_bodies = set(_KEEP_BODIES) | {f"{side}_{link}" for link in _ARM_LINKS}
    # Delete only whole subtrees whose parent we keep -- legs, head, and the
    # other arm are each one child of the (kept) pelvis/torso. Walking the
    # body list and deleting every non-kept body instead leaves the wrappers of
    # already-freed descendants alive, which segfaults when they are collected.
    doomed = [
        body.name
        for body in spec.bodies
        if body.name not in keep_bodies and body.parent.name in keep_bodies
    ]
    for name in doomed:
        spec.delete(spec.body(name))

    keep_joints = {f"{side}_{name}" for name in _ARM_JOINTS}
    for joint in list(spec.joints):
        if joint.name not in keep_joints:
            # The free joint and the three waist joints. Dropping a joint
            # welds its body to its parent, which is what pins the trunk.
            spec.delete(joint)

    # Everything below references the deleted elements (or is simply unused).
    for attr in ("actuators", "sensors", "cameras", "lights", "equalities", "tendons"):
        collection = getattr(spec, attr, None)
        if collection is None:
            continue
        for element in list(collection):
            spec.delete(element)
    return spec.compile()


class _SaturatedRelativeFrameTask(mink.RelativeFrameTask):
    """``RelativeFrameTask`` whose angular residual is capped in magnitude.

    Round 8.1's ``max_rotation_err_rad``: the operator's wrist re-orients far
    past what the three wrist joints can reach, so the unreachable part of the
    orientation error persists tick after tick, and answering a large
    persistent residual with a proportionally large step turns into grinding
    at the wrist limit. Clamping the residual's *magnitude* while keeping its
    direction bounds the pull without a gate or any state.

    mink hands both seams the raw task error (the ``gain``/``cost`` scaling
    happens afterwards, inside ``_weighted_residual``), so ``error[3:]`` is the
    angular residual in radians and can be compared against the cap directly.
    Below the cap this is a no-op that forwards the same array object -- the
    solver stays bit-for-bit the unsaturated one, which is what
    ``incremental_chain.check_pose_err_cap`` asserts.

    Both seams are overridden because the first version of this class only
    overrode ``_assemble_qp``, which mink never calls for a FrameTask:
    ``solve_ik._compute_qp_objective`` asks for ``compute_qp_residual`` first,
    and ``RelativeFrameTask`` answers with a ``_weighted_residual`` triple, so
    the objective path (and with it ``_assemble_qp``) is only reached by tasks
    that decline to expose a residual. The saturation was silently inert until
    ``check_pose_err_cap``'s per-tick step bound caught it.
    """

    def __init__(self, *args, max_rotation_err_rad: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_rotation_err_rad = float(max_rotation_err_rad)

    def _capped(self, error: np.ndarray) -> np.ndarray:
        cap = self.max_rotation_err_rad
        if cap > 0.0:
            norm = float(np.linalg.norm(error[3:]))
            if norm > cap:
                return np.concatenate([error[:3], error[3:] * (cap / norm)])
        return error

    def _weighted_residual(self, error: np.ndarray, jacobian: np.ndarray):
        return super()._weighted_residual(self._capped(error), jacobian)

    def _assemble_qp(self, error: np.ndarray, jacobian: np.ndarray, eye_nv: np.ndarray):
        return super()._assemble_qp(self._capped(error), jacobian, eye_nv)


class MinkArmIK:
    """Single-arm mink solver. Side-symmetric (constructed per arm).

    Args:
        side: ``"left"`` or ``"right"``.
        damping: DLS damping coefficient, same meaning and default as
            :class:`ArmIKSolver`; enters the QP as ``damping ** 2``.
        position_weight: gain on the linear residual.
        rotation_weight: gain on the angular residual. 0 disables
            orientation tracking (position-only IK).
        max_per_tick_step_rad: per-joint bound on one update step, enforced by
            ``VelocityLimit``. Joint ranges are clipped after the step instead
            of constrained in the QP (see the module docstring).
        max_rotation_err_rad: saturation cap on the angular residual (0 = off).
        posture_gain: cost of the optional null-space ``PostureTask`` (0 =
            off). The task has no position/orientation Jacobian rows of its
            own: on the 4-DoF position path it can only move the arm's
            one-dimensional swivel, so it regularises *which* upstream
            configuration tracks the target without moving the wrist point
            (to first order). It is a soft task, so a tiny position trade is
            possible; keep the cost low (1e-3 .. 1e-2) and verify with
            ``check_posture_gain_null_space``.
        q_preferred: posture the ``PostureTask`` pulls toward. Shape (7,);
            ``ClutchArm`` re-targets it to the engage configuration.
        solver: QP backend name handed to :func:`mink.solve_ik`.
        model_path: MJCF to build the arm-only model from; defaults to the
            X2 description in this repo.
    """

    backend = "mink"

    def __init__(
        self,
        side: str,
        damping: float = 0.08,
        position_weight: float = 1.0,
        rotation_weight: float = 0.5,
        max_per_tick_step_rad: float = 0.30,
        max_rotation_err_rad: float = 0.0,
        posture_gain: float = 0.0,
        q_preferred: np.ndarray | None = None,
        solver: str = "daqp",
        model_path: str | Path | None = None,
    ):
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        self.side = side
        self.damping = float(damping)
        self.position_weight = float(position_weight)
        self.rotation_weight = float(rotation_weight)
        self.max_per_tick_step_rad = float(max_per_tick_step_rad)
        self.max_rotation_err_rad = float(max_rotation_err_rad)
        self.posture_gain = float(posture_gain)
        self.solver_name = str(solver)
        self._chain = left_arm_chain() if side == "left" else right_arm_chain()

        limits = X2_ARM_JOINT_LIMITS_RAD[:7] if side == "left" else X2_ARM_JOINT_LIMITS_RAD[7:]
        self._lo = np.array([lo for lo, _ in limits], dtype=np.float64)
        self._hi = np.array([hi for _, hi in limits], dtype=np.float64)

        self.model = arm_only_model(side, str(model_path or _mjcf_path()))
        self._check_model()

        self._configuration = mink.Configuration(self.model)
        self._position_task = mink.RelativeFrameTask(
            f"{side}_wrist_roll_link",
            "body",
            "torso_link",
            "body",
            position_cost=1.0,
            orientation_cost=0.0,
            gain=self.position_weight,
            lm_damping=0.0,
        )
        self._orientation_task = _SaturatedRelativeFrameTask(
            f"{side}_wrist_roll_link",
            "body",
            "torso_link",
            "body",
            position_cost=0.0,
            orientation_cost=1.0,
            gain=self.rotation_weight,
            lm_damping=0.0,
            max_rotation_err_rad=self.max_rotation_err_rad,
        )
        # Step cap only. Joint ranges are clipped after the step instead of
        # being handed to ConfigurationLimit: see the module docstring for the
        # 50.9 mm position sacrifice that decision came from.
        self._limits = [
            mink.VelocityLimit(
                self.model,
                {f"{side}_{name}": self.max_per_tick_step_rad / DT for name in _ARM_JOINTS},
            ),
        ]
        # solve_position_free4 pins the wrist exactly rather than with a limit.
        wrist_dofs = [
            int(self.model.jnt_dofadr[self.model.joint(f"{side}_{name}").id])
            for name in _ARM_JOINTS[4:]
        ]
        self._freeze_wrist = mink.DofFreezingTask(model=self.model, dof_indices=wrist_dofs)

        # Optional null-space posture. On a 3-D position task over the
        # shoulder-3 + elbow, the null space is the single elbow-swivel
        # direction; a low-cost PostureTask picks the swivel nearest the
        # preferred configuration instead of letting numerical drift (or a
        # moving operator wrist) choose it. The three wrist DOFs are frozen
        # by ``_freeze_wrist``, so the task is inert there.
        if q_preferred is None:
            q_preferred = np.zeros(7, dtype=np.float64)
        q_preferred = np.asarray(q_preferred, dtype=np.float64).reshape(-1)
        if q_preferred.shape != (self.model.nq,):
            raise ValueError(f"q_preferred must be ({self.model.nq},); got {q_preferred.shape}")
        self.q_preferred = q_preferred.copy()
        self._posture_task = None
        if self.posture_gain > 0.0:
            self._posture_task = mink.PostureTask(
                model=self.model, cost=self.posture_gain, gain=1.0, lm_damping=0.0
            )
            self._posture_task.set_target(self.q_preferred)

    def set_posture_target(self, q: np.ndarray) -> None:
        """Move the null-space posture target (no-op when posture is off)."""
        if self._posture_task is None:
            return
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if q.shape != (self.model.nq,):
            raise ValueError(f"posture target must be ({self.model.nq},); got {q.shape}")
        self.q_preferred = q.copy()
        self._posture_task.set_target(self.q_preferred)

    def _check_model(self) -> None:
        """Fail at build time if the MJCF stops agreeing with the catalog.

        The joint ranges are the clip bounds the *robot* clips against, so a
        regenerated description that quietly moved them would otherwise show up
        as the sim chasing a reach the hardware does not have.
        """
        for index, name in enumerate(_ARM_JOINTS):
            joint = self.model.joint(f"{self.side}_{name}")
            if joint.type != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f"{self.side}_{name} is not a hinge joint")
            if not self.model.jnt_limited[joint.id]:
                raise ValueError(f"{self.side}_{name} has no joint range in the MJCF")
            got = np.asarray(self.model.jnt_range[joint.id], dtype=np.float64)
            want = np.array([self._lo[index], self._hi[index]], dtype=np.float64)
            if not np.allclose(got, want, rtol=0.0, atol=1e-9):
                raise ValueError(
                    f"{self.side}_{name} range {got} != catalog {want}; the MJCF "
                    "and X2_ARM_JOINT_LIMITS_RAD have drifted apart"
                )

    # ── same surface as ArmIKSolver ───────────────────────────────────────

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward kinematics. Returns (xyz, R_3x3) of the wrist.

        Deliberately the plain chain FK, not mink: ``ClutchArm`` builds its
        shoulder point and reachable sphere from it, telemetry reports
        through it, and the numbers have to stay comparable with the legacy
        backend and with the robot.
        """
        q = np.asarray(q, dtype=np.float64)
        T, _, _ = chain_fk_full(self._chain, q)
        return T[:3, 3].copy(), T[:3, :3].copy()

    def reset_rotation_err_state(self) -> None:
        """API compatibility: saturation is stateless, nothing to reset."""

    def solve(
        self,
        q_seed: np.ndarray,
        target_pos: np.ndarray,
        target_quat_wxyz: np.ndarray | None = None,
        max_iters: int = 1,
    ) -> tuple[np.ndarray, IKResult]:
        """Solve one (or a few) QP step(s). Returns ``(q_next, info)``."""
        q_seed = np.asarray(q_seed, dtype=np.float64).copy()
        target_pos = np.asarray(target_pos, dtype=np.float64)
        R_target: np.ndarray | None = None
        if target_quat_wxyz is not None and self.rotation_weight > 0:
            R_target = _quat_wxyz_to_matrix(np.asarray(target_quat_wxyz, dtype=np.float64))

        if R_target is None:
            # Position-only. The task still carries a rotation, and mink's
            # error is a single SE(3) log whose left Jacobian mixes the angular
            # rows into the linear ones *at that rotation's angle* -- so the
            # rotation it is given is not cosmetic even though its cost is
            # zero. Seeding it with the arm's own rotation keeps the log a
            # small position twist; handing it an unrelated rotation (identity,
            # say) adds a first-order error to the step, which is what the
            # A/B against the legacy position-only branch would otherwise
            # measure instead of the solver.
            tasks = [self._position_task]
            self._position_task.set_target(_target_se3(self.fk(q_seed)[1], target_pos))
        else:
            # Both tasks get the same target pose, for the same reason: the
            # position task's log then turns at the (small) residual rotation
            # rather than at the target's absolute orientation.
            tasks = [self._position_task, self._orientation_task]
            target = _target_se3(R_target, target_pos)
            self._position_task.set_target(target)
            self._orientation_task.set_target(target)

        q = q_seed
        clamped_count = 0
        pos_err, rot_err = self._residual(q, target_pos, R_target)
        iters = 0
        for _ in range(max(1, max_iters)):
            # The legacy loop's convergence test, moved from after the step to
            # before it. mink's model FK and the hand-copied chain FK differ by
            # ~1e-6 m (MJCF print precision), so a target sampled from the
            # chain is never exactly reached and the QP would answer every
            # "hold still" tick with a ~1e-7 rad correction that means nothing
            # physically while making "the commanded q does not move" untrue --
            # which is what the freeze/re-anchor/leak invariants assert at
            # 1e-9. Below the tolerance there is nothing to correct: skip.
            # The thresholds are the legacy ones, so the two backends agree on
            # when to stop.
            if pos_err < 5e-4 and rot_err < 5e-3:
                break
            iters += 1
            self._configuration.update(q)
            try:
                v = mink.solve_ik(
                    self._configuration,
                    tasks,
                    dt=DT,
                    solver=self.solver_name,
                    damping=self.damping**2,
                    safety_break=False,
                    limits=self._limits,
                )
            except Exception:  # mink.NoSolutionFound, or a backend blow-up
                # Report the arm as unsolved rather than letting the exception
                # escape into the 50 Hz loop: inf > freeze_pos_err_m latches
                # the freeze guard, which holds the last command -- the same
                # place a diverging DLS ends up.
                return q_seed, IKResult(float("inf"), float("inf"), 0, iters - 1)
            self._configuration.integrate_inplace(v, DT)
            q_new = np.asarray(self._configuration.q, dtype=np.float64).copy()

            # The legacy joint-range clip, still the mechanism in charge (the
            # QP is given the step cap only -- see the module docstring).
            clamped = (q_new < self._lo) | (q_new > self._hi)
            if np.any(clamped):
                clamped_count += int(np.sum(clamped))
                q_new = np.clip(q_new, self._lo, self._hi)
            q = q_new
            pos_err, rot_err = self._residual(q, target_pos, R_target)

        return q, IKResult(
            pos_err_m=pos_err,
            rot_err_rad=rot_err,
            clamped_count=clamped_count,
            iterations=iters,
        )

    def _residual(
        self, q: np.ndarray, target_pos: np.ndarray, R_target: np.ndarray | None
    ) -> tuple[float, float]:
        """Position / rotation error against the UNsaturated target.

        Measured through the plain chain FK (not mink's task error) so the
        numbers stay comparable with the legacy backend, with the tolerances
        above, and with ``freeze_pos_err_m``.
        """
        p_e, R_e = self.fk(q)
        pos_err = float(np.linalg.norm(target_pos - p_e))
        rot_err = (
            float(np.linalg.norm(_rotation_error_axis_angle(R_e, R_target)))
            if R_target is not None
            else 0.0
        )
        return pos_err, rot_err

    def solve_position_free4(
        self,
        q_seed: np.ndarray,
        wrist_q3: np.ndarray,
        target_pos: np.ndarray,
        max_iters: int = 1,
    ) -> tuple[np.ndarray, IKResult]:
        """Position-only solve over shoulder-3 + elbow with the wrist pinned.

        The v3 teleop path: the three wrist joints are commanded by the
        per-axis controller mapping, so the solve must leave them exactly
        where it was told. mink expresses that as an equality constraint
        (``DofFreezingTask``) rather than as a smaller Jacobian, which keeps
        the whole 7-DoF model and its step cap in play.
        """
        q_seed = np.asarray(q_seed, dtype=np.float64)
        wrist_q3 = np.asarray(wrist_q3, dtype=np.float64).reshape(3)
        q = np.concatenate([q_seed[:4], wrist_q3])
        target_pos = np.asarray(target_pos, dtype=np.float64)
        self._position_task.set_target(_target_se3(self.fk(q)[1], target_pos))

        pos_err = float(np.linalg.norm(target_pos - self.fk(q)[0]))
        clamped_count = 0
        iters = 0
        tasks = [self._position_task]
        if self._posture_task is not None:
            # The posture task has no end-effector rows: it lives entirely in
            # the position task's null space. Keeping it in the task list is
            # what lets it act on a tick whose position error is already
            # converged, so the pre-step break below must not fire then.
            tasks.append(self._posture_task)
        for _ in range(max(1, max_iters)):
            # Same pre-step convergence test as solve(); see the note there.
            # With posture ON we keep iterating: position error 0 does not
            # mean the null-space task has converged.
            if pos_err < 5e-4 and self._posture_task is None:
                break
            iters += 1
            self._configuration.update(q)
            try:
                v = mink.solve_ik(
                    self._configuration,
                    tasks,
                    dt=DT,
                    solver=self.solver_name,
                    damping=self.damping**2,
                    safety_break=False,
                    limits=self._limits,
                    constraints=[self._freeze_wrist],
                )
            except Exception:
                return q, IKResult(float("inf"), float("inf"), 0, iters - 1)
            self._configuration.integrate_inplace(v, DT)
            q_new = np.asarray(self._configuration.q, dtype=np.float64).copy()

            clamped = (q_new[:4] < self._lo[:4]) | (q_new[:4] > self._hi[:4])
            if np.any(clamped):
                clamped_count += int(np.sum(clamped))
                q_new[:4] = np.clip(q_new[:4], self._lo[:4], self._hi[:4])
            # The constraint holds the wrist, but pin it again so the contract
            # is exact even if a future limit ever touches those DOFs.
            q_new[4:7] = wrist_q3
            q = q_new
            pos_err = float(np.linalg.norm(target_pos - self.fk(q)[0]))

        return q, IKResult(
            pos_err_m=pos_err,
            rot_err_rad=0.0,
            clamped_count=clamped_count,
            iterations=iters,
        )


def _target_se3(R: np.ndarray, p: np.ndarray) -> mink.SE3:
    """A target pose in the torso frame, which is the frame the tasks use."""
    return mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(R), p)
