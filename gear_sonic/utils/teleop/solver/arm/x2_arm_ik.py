"""Damped Least-Squares (DLS) IK solver for the X2 arm.

A pure-numpy single-step Jacobian-pseudo-inverse solver. Intended for
the VR teleop dataset recorder
(:mod:`gear_sonic.scripts.record_x2_dataset`): one DLS step per 50 Hz
tick per arm, with the previous q as the seed.

Why DLS and not pinocchio + qpsolvers (the path taken by
``decoupled_wbc.control.teleop.solver.body.body_ik_solver``):

* The recorder loop already runs at 50 Hz and the operator's wrist
  only moves a few mm per tick. A single DLS update with the previous
  q as the seed tracks easily while staying numerically stable through
  the X2's elbow-extended singularity (where the elbow joint
  approaches 0).
* Pulling pinocchio + URDF parsing into the recorder's runtime adds
  ~30 MB of native deps for what amounts to ~50 lines of analytical
  FK + a 6x7 SVD per arm.

Vendored from
``agitbot-x2-record-and-replay/src/x2_recorder/teleop/arm_ik.py``
(branch ``quest3-bare-hand-control``). The joint names + limits +
default PD gains here mirror the X2 hardware deploy's PD controller
(``ARM_JOINTS`` in the sister repo's ``constants.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gear_sonic.utils.teleop.solver.arm.x2_arm_fk import (
    chain_fk_full,
    left_arm_chain,
    right_arm_chain,
)


# ── X2 arm joint catalog (matches the deploy's PD config) ─────────────────


X2_ARM_JOINT_NAMES: tuple[str, ...] = (
    # Left arm [0..6]
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    # Right arm [7..13]
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
)


# (lower, upper) in radians, matching ``ARM_JOINTS`` in the sister
# repo. The shoulder-roll asymmetry between left/right is real (URDF
# joint limits are mirrored, not symmetric).
X2_ARM_JOINT_LIMITS_RAD: tuple[tuple[float, float], ...] = (
    (-3.080,  2.040),  # left_shoulder_pitch
    (-0.061,  2.993),  # left_shoulder_roll
    (-2.556,  2.556),  # left_shoulder_yaw
    (-2.3556, 0.000),  # left_elbow
    (-2.556,  2.556),  # left_wrist_yaw
    (-0.558,  0.558),  # left_wrist_pitch
    (-1.571,  0.724),  # left_wrist_roll
    (-3.080,  2.040),  # right_shoulder_pitch
    (-2.993,  0.061),  # right_shoulder_roll
    (-2.556,  2.556),  # right_shoulder_yaw
    (-2.3556, 0.000),  # right_elbow
    (-2.556,  2.556),  # right_wrist_yaw
    (-0.558,  0.558),  # right_wrist_pitch
    (-0.724,  1.571),  # right_wrist_roll
)


# ── Quaternion / rotation helpers ─────────────────────────────────────────


def _quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert a [w, x, y, z] quaternion to a 3x3 rotation matrix."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = w * w + x * x + y * y + z * z
    if n == 0.0:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def _rotation_error_axis_angle(R_current: np.ndarray, R_target: np.ndarray) -> np.ndarray:
    """Return a 3-vector axis*angle (radians) rotating R_current to R_target.

    Used as the orientation residual feeding the Jacobian's angular
    rows. We deliberately drop the per-axis sign-disambiguation that a
    proper SO(3) log would do for >pi rotations -- DLS will iterate
    toward the right sign on the next tick, and per-tick the operator
    rarely commands more than 5 deg of wrist twist.
    """
    R = R_target @ R_current.T
    cos_theta = (np.trace(R) - 1.0) * 0.5
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta = float(np.arccos(cos_theta))
    if theta < 1e-6:
        return np.zeros(3, dtype=np.float64)
    rx = R[2, 1] - R[1, 2]
    ry = R[0, 2] - R[2, 0]
    rz = R[1, 0] - R[0, 1]
    axis = np.array([rx, ry, rz], dtype=np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        diag = np.array([R[0, 0], R[1, 1], R[2, 2]])
        i = int(np.argmax(diag))
        col = (R[:, i] + np.eye(3)[:, i])
        n = np.linalg.norm(col)
        if n < 1e-12:
            return np.zeros(3, dtype=np.float64)
        return col / n * theta
    return axis / n * theta


def matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a [w, x, y, z] quaternion."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return np.array([qw, qx, qy, qz], dtype=np.float64)


# ── Solver ────────────────────────────────────────────────────────────────


@dataclass
class IKResult:
    """Diagnostic info returned alongside the solved joint vector."""

    pos_err_m: float
    rot_err_rad: float
    clamped_count: int
    iterations: int


class ArmIKSolver:
    """Single-arm DLS solver. Side-symmetric (constructed per arm).

    Args:
        side: ``"left"`` or ``"right"``.
        damping: DLS damping coefficient. Larger = more stable near
            singularities but slower convergence. The classic value
            for human-scale 7-DoF arms is 0.05; we use 0.08 because
            the X2's elbow-straight pose is right in the middle of
            the comfortable teleop workspace.
        position_weight: gain on the linear residual.
        rotation_weight: gain on the angular residual. 0 disables
            orientation tracking (position-only IK).
        max_per_tick_step_rad: per-joint upper bound on a single
            update step. Acts as a final safety belt; a runaway DLS
            iteration (e.g. near a singularity with a huge target
            error) cannot move any joint more than this in one solve.
        uniform_step_clip: scale the WHOLE dq vector uniformly down to
            the max step instead of clipping per-joint. Near a
            singularity (elbow straight) the raw dq is huge; per-joint
            clipping distorts its direction (different joints saturate
            at different times), which feeds a bang-bang oscillation --
            the live "arm shakes once extended" symptom. Uniform
            scaling keeps the descent direction and only caps the
            magnitude. Opt-in; default keeps the legacy clip.
        max_rotation_err_rad: SOFT-envelope threshold on the angular
            residual fed to the step (radians; 0 = off, legacy
            behaviour). Rationale (live round 8, pose mode): the
            operator wrist re-orients over a much wider range than the
            robot's three wrist joints (+-0.558 rad pitch etc.), so the
            unreachable part of the orientation error persists tick
            after tick -- and DLS answers a large persistent residual
            with a large per-tick step. Measured on a 90-deg twist:
            the commanded dq scales ~linearly with the residual
            (5 deg -> 0.006 rad/tick, 34 deg -> 0.14, 90 deg -> 0.26),
            and at the wrist-limit boundary it turns into a symmetric
            +-40 deg wrist_roll limit cycle (~1 Hz) that also wrecks
            position tracking. A hard cap only shrinks the cycle. The
            envelope instead FADES the residual: full weight below
            ``max_rotation_err_rad`` (any trackable pose keeps exact
            accuracy), linearly down to ZERO at ``2.5x`` the threshold
            (an unreachable pose stops PUSHING entirely -- the
            orientation task yields, position tracking wins, and the
            cycle's driving force disappears).
    """

    def __init__(
        self,
        side: str,
        damping: float = 0.08,
        position_weight: float = 1.0,
        rotation_weight: float = 0.5,
        max_per_tick_step_rad: float = 0.30,
        null_space_gain: float = 0.0,
        q_preferred: np.ndarray | None = None,
        uniform_step_clip: bool = False,
        max_rotation_err_rad: float = 0.0,
    ):
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        self.side = side
        self.damping = float(damping)
        self.position_weight = float(position_weight)
        self.rotation_weight = float(rotation_weight)
        self.max_per_tick_step_rad = float(max_per_tick_step_rad)
        self.uniform_step_clip = bool(uniform_step_clip)
        self.max_rotation_err_rad = float(max_rotation_err_rad)
        # Residual above which the orientation task fully yields (rad).
        # 2.5x keeps the fade band wide (>= e.g. 0.2 rad at the 0.08
        # default) so the hand-off is gradual, not a switch.
        self._rot_err_fade_rad = 2.5 * self.max_rotation_err_rad
        # Per-tick trend of the angular residual (EMA, rad/tick), used to
        # tell "operator is turning BACK" (closing) from "operator keeps
        # pushing past the limit" (opening) while the residual is inside
        # the dead band. Closing re-arms full pull-in so the wrist
        # follows home instead of locking at the frozen pose.
        self._last_rot_err: float | None = None
        self._rot_err_trend = 0.0
        # Trend below which the residual is considered "moving" (rad/tick).
        # Sits above static noise (~2e-3) and below a deliberate return.
        self._rot_err_trend_eps = 5.0e-3
        # True when a wrist joint sits within ``_rot_err_jam_margin`` of
        # its limit. Freezing a large residual is only correct then --
        # a free wrist with a big residual is lag on a reachable target.
        self._wrist_jammed = False
        self._rot_err_jam_margin = 0.04
        # Whether the orientation target itself is moving (operator is
        # still rotating). The limit cycle is a property of a
        # *stationary* unreachable target; while the operator keeps
        # turning, the loop is driven, never resonant -- the unfrozen
        # baseline tracks a moving target cleanly.
        #
        # Debounced over 2 consecutive ticks: live Pico streams show
        # isolated input spikes up to ~3 deg/frame, and a one-tick
        # "moving" blip would re-arm full pull-in for a single tick
        # (a visible twitch). A real rotation stays above the
        # threshold for many ticks, so the 1-tick confirmation delay
        # costs nothing.
        self._last_target_R: np.ndarray | None = None
        self._move_votes = 0
        self._target_moving = False
        self._target_move_eps = 3.0e-3
        self._chain = left_arm_chain() if side == "left" else right_arm_chain()

        if side == "left":
            limits = X2_ARM_JOINT_LIMITS_RAD[:7]
        else:
            limits = X2_ARM_JOINT_LIMITS_RAD[7:]
        self._lo = np.array([lo for lo, _ in limits], dtype=np.float64)
        self._hi = np.array([hi for _, hi in limits], dtype=np.float64)

        # Null-space bias toward a preferred posture. For 7-DOF arms,
        # there is at least one redundant DOF (the elbow-swivel about
        # the shoulder->wrist axis) even when both position AND
        # orientation are tracked. Without a bias, the DLS solver picks
        # whichever swivel minimises ``||dq||``, which produces visible
        # branch flips ("elbow flips backward") whenever the IK target
        # is unreachable or the operator brushes a singularity.
        #
        # The standard fix is task-priority kinematic control: project
        # the bias ``(q_pref - q)`` into the null space of J and add it
        # to the primary task velocity. ``null_space_gain == 0`` (the
        # default) disables it for backward compatibility with the
        # legacy engage-anchor solver path; the calibrated v2 solver
        # opts in with ~0.1.
        self.null_space_gain = float(null_space_gain)
        if q_preferred is None:
            q_preferred = np.zeros(7, dtype=np.float64)
        self.q_preferred = np.asarray(q_preferred, dtype=np.float64).copy()
        if self.q_preferred.shape != (7,):
            raise ValueError(
                f"q_preferred must be (7,); got {self.q_preferred.shape}"
            )

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward kinematics. Returns (xyz, R_3x3) of the wrist."""
        q = np.asarray(q, dtype=np.float64)
        T, _, _ = chain_fk_full(self._chain, q)
        return T[:3, 3].copy(), T[:3, :3].copy()

    def _jacobian(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the 6x7 geometric Jacobian and current EE pose."""
        T, origins, axes = chain_fk_full(self._chain, q)
        p_e = T[:3, 3]
        R_e = T[:3, :3]
        J = np.zeros((6, 7), dtype=np.float64)
        for i in range(7):
            z_i = axes[i]
            p_i = origins[i]
            J[:3, i] = np.cross(z_i, p_e - p_i)
            J[3:, i] = z_i
        return J, p_e, R_e

    def reset_rotation_err_state(self) -> None:
        """Forget the angular-residual trend (call on re-anchor/clutch)."""
        self._last_rot_err = None
        self._rot_err_trend = 0.0
        self._wrist_jammed = False
        self._last_target_R = None
        self._move_votes = 0
        self._target_moving = False

    def solve(
        self,
        q_seed: np.ndarray,
        target_pos: np.ndarray,
        target_quat_wxyz: np.ndarray | None = None,
        max_iters: int = 1,
    ) -> tuple[np.ndarray, IKResult]:
        """Solve one (or a few) DLS step(s). Returns ``(q_next, info)``.

        ``max_iters`` defaults to 1 because the recorder iterates the
        solver tick-by-tick at 50 Hz. Set higher (e.g. 3) for warm-up
        or bring-the-arm-to-target use cases that don't have a
        per-tick cadence behind them.
        """
        q = np.asarray(q_seed, dtype=np.float64).copy()
        target_pos = np.asarray(target_pos, dtype=np.float64)
        R_target: np.ndarray | None = None
        if target_quat_wxyz is not None and self.rotation_weight > 0:
            R_target = _quat_wxyz_to_matrix(np.asarray(target_quat_wxyz, dtype=np.float64))

        clamped_count = 0
        pos_err = 0.0
        rot_err = 0.0
        iters = 0
        for _ in range(max(1, max_iters)):
            iters += 1
            J, p_e, R_e = self._jacobian(q)

            err_lin = self.position_weight * (target_pos - p_e)
            if R_target is not None:
                e_ang = _rotation_error_axis_angle(R_e, R_target)
                n_ang = float(np.linalg.norm(e_ang))
                if iters == 1:
                    # Per-tick state update (solve runs once per tick,
                    # the inner iterations share the tick's entry state).
                    if self._last_rot_err is not None:
                        self._rot_err_trend = 0.5 * self._rot_err_trend + 0.5 * (
                            n_ang - self._last_rot_err
                        )
                    self._last_rot_err = n_ang
                    if self._last_target_R is not None:
                        tr = float(np.trace(self._last_target_R.T @ R_target))
                        step = float(np.arccos(np.clip(0.5 * (tr - 1.0), -1.0, 1.0)))
                        self._move_votes = (
                            self._move_votes + 1 if step > self._target_move_eps else 0
                        )
                        self._target_moving = self._move_votes >= 2
                    self._last_target_R = R_target
                    m = np.minimum(q[4:] - self._lo[4:], self._hi[4:] - q[4:])
                    self._wrist_jammed = bool(m.min() < self._rot_err_jam_margin)
                if self.max_rotation_err_rad > 0.0:
                    lo = self.max_rotation_err_rad
                    hi = self._rot_err_fade_rad
                    closing = self._rot_err_trend < -self._rot_err_trend_eps
                    if n_ang >= hi:
                        # Unreachable residual. Yield the orientation task
                        # entirely (position tracking owns the arm) ONLY
                        # in the situation that sustains the limit cycle:
                        # a stationary far target AND a wrist jammed
                        # against its limit. Anything else keeps pulling
                        # -- the operator may be still turning (driven,
                        # never resonant), may be turning back, or the
                        # wrist may simply be lagging a reachable target.
                        if self._wrist_jammed and not self._target_moving and not closing:
                            e_ang = np.zeros(3)
                    elif n_ang > lo:
                        e_ang = e_ang * ((hi - n_ang) / (hi - lo))
                err_ang = self.rotation_weight * e_ang
                err = np.concatenate([err_lin, err_ang])
            else:
                # Position-only: drop the angular rows of J so the
                # solver doesn't waste DoF tracking arbitrary
                # orientation. The Jacobian-pseudoinv with `damping`
                # naturally regularises the under-determined case.
                J = J[:3, :]
                err = err_lin

            n_rows = J.shape[0]
            JJt = J @ J.T
            damp = (self.damping ** 2) * np.eye(n_rows)
            try:
                inv = np.linalg.solve(JJt + damp, err)
            except np.linalg.LinAlgError:
                inv = np.linalg.lstsq(JJt + damp, err, rcond=None)[0]
            dq = J.T @ inv

            # Null-space task: bias toward ``q_preferred`` in the
            # subspace that does NOT affect the end-effector. For a
            # 7-DOF arm this is at minimum a 1-D swivel; for
            # position-only IK it's 4-D. Standard task-priority form:
            #     dq_total = dq_primary + (I - J^+ J) * dq_null
            # where ``dq_primary = J.T @ (J J^T + λ²I)^-1 * err`` and
            # ``J^+ = J^T (J J^T + λ²I)^-1``. We project the bias by
            # subtracting the J-image of dq_null, which is equivalent
            # and avoids forming the explicit (7,7) projector.
            if self.null_space_gain != 0.0:
                dq_null = self.null_space_gain * (self.q_preferred - q)
                try:
                    null_inv = np.linalg.solve(JJt + damp, J @ dq_null)
                except np.linalg.LinAlgError:
                    null_inv = np.linalg.lstsq(JJt + damp, J @ dq_null, rcond=None)[0]
                dq_null_proj = dq_null - J.T @ null_inv
                dq = dq + dq_null_proj

            if self.uniform_step_clip:
                mx = float(np.max(np.abs(dq)))
                if mx > self.max_per_tick_step_rad:
                    dq *= self.max_per_tick_step_rad / mx
            else:
                np.clip(dq, -self.max_per_tick_step_rad,
                        self.max_per_tick_step_rad, out=dq)

            q_new = q + dq

            clamped_idx = (q_new < self._lo) | (q_new > self._hi)
            if np.any(clamped_idx):
                clamped_count += int(np.sum(clamped_idx))
                q_new = np.clip(q_new, self._lo, self._hi)

            q = q_new

            p_e_check, R_e_check = self.fk(q)
            pos_err = float(np.linalg.norm(target_pos - p_e_check))
            if R_target is not None:
                rot_err = float(np.linalg.norm(
                    _rotation_error_axis_angle(R_e_check, R_target)
                ))
            else:
                rot_err = 0.0

            if pos_err < 5e-4 and rot_err < 5e-3:
                break

        return q, IKResult(
            pos_err_m=pos_err,
            rot_err_rad=rot_err,
            clamped_count=clamped_count,
            iterations=iters,
        )
