"""Headless tests for the mink arm IK backend (``solver/arm/x2_arm_mink.py``).

Three families:

* the hand-copied ``x2_arm_fk`` chain against the two descriptions it has to
  agree with -- the MJCF the mink model is pruned from, and the URDF the chain
  was extracted from. The URDF audit is the one that found the right wrist
  1.22 mm off the deploy's kinematics (wrist_yaw/wrist_pitch had been the left
  arm's values mirrored and rounded); it is here so a future "cleanup" that
  restores them fails loudly instead of silently moving the right hand.
* solver contracts: the position-only branch, ``max_iters`` composition, the
  joint-range clip, and the QP-failure fallback.
* the rotation-residual saturation, including the non-vacuity check. An
  earlier revision capped the error by overriding a seam that mink's
  ``compute_qp_residual`` fast path never calls, so the cap was inert while
  every "saturation" assertion still passed; ``test_saturation_reaches_the_qp``
  drives the public ``solve()`` so it cannot pass that way again.

Run: ``.venv/bin/python -m pytest gear_sonic/tests/test_arm_mink_ik.py -q``
"""

from __future__ import annotations

from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as sRot

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

pytest.importorskip("mink", reason="the mink backend needs mink/mujoco; X2_IK_BACKEND=dls skips it")

import mink  # noqa: E402

from gear_sonic.utils.teleop.incremental_chain import (  # noqa: E402
    _rot_to_quat_wxyz,
    make_arm_solver,
    resolve_arm_backend,
)
from gear_sonic.utils.teleop.solver.arm.x2_arm_fk import (  # noqa: E402
    left_arm_chain,
    right_arm_chain,
)
from gear_sonic.utils.teleop.solver.arm.x2_arm_ik import (  # noqa: E402
    X2_ARM_JOINT_LIMITS_RAD,
)
from gear_sonic.utils.teleop.solver.arm.x2_arm_mink import (  # noqa: E402
    MinkArmIK,
)

URDF = (
    REPO
    / "gear_sonic"
    / "data"
    / "assets"
    / "robot_description"
    / "urdf"
    / "x2_ultra"
    / "x2_ultra.urdf"
)

JOINT_NAMES = (
    "shoulder_pitch_joint",
    "shoulder_roll_joint",
    "shoulder_yaw_joint",
    "elbow_joint",
    "wrist_yaw_joint",
    "wrist_pitch_joint",
    "wrist_roll_joint",
)

FK_GATE_M = 1e-5
FK_GATE_RAD = 1e-5


def _chain_and_bounds(side: str):
    chain = left_arm_chain() if side == "left" else right_arm_chain()
    limits = np.asarray(X2_ARM_JOINT_LIMITS_RAD, dtype=np.float64)
    off = 0 if side == "left" else 7
    return chain, limits[off : off + 7, 0], limits[off : off + 7, 1]


def _rotation_angle(R: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))


def _rotated_quat(R0: np.ndarray, axis, angle: float) -> np.ndarray:
    """Quaternion (w, x, y, z) of ``R0`` rotated by ``angle`` about ``axis``."""
    return _rot_to_quat_wxyz(
        sRot.from_rotvec(np.asarray(axis, float) * angle) * sRot.from_matrix(R0)
    )


# ── the two descriptions ──────────────────────────────────────────────────


@pytest.mark.parametrize("side", ["left", "right"])
def test_fk_frame_agreement(side: str) -> None:
    """Chain FK == mink's own FK, same frame, over the whole joint box.

    The mink model is pruned from the MJCF, the chain is hand-copied from the
    URDF, and every telemetry path, the keep-out projection and the reachable
    sphere are built on the chain while the solve runs on the MJCF. They have
    to be the same kinematics.
    """
    solver = MinkArmIK(side)
    _chain, lo, hi = _chain_and_bounds(side)
    rng = np.random.RandomState(0)
    worst_p = worst_r = 0.0
    for _ in range(2000):
        q = rng.uniform(lo, hi)
        solver._configuration.update(q)
        T_mink = np.asarray(
            solver._configuration.get_transform(
                f"{side}_wrist_roll_link", "body", "torso_link", "body"
            ).as_matrix(),
            dtype=np.float64,
        )
        p_chain, R_chain = solver.fk(q)
        worst_p = max(worst_p, float(np.linalg.norm(T_mink[:3, 3] - p_chain)))
        worst_r = max(worst_r, _rotation_angle(T_mink[:3, :3].T @ R_chain))
    assert worst_p < FK_GATE_M, f"{side}: chain vs mink FK differs by {worst_p * 1e3:.4f} mm"
    assert worst_r < FK_GATE_RAD, f"{side}: chain vs mink FK rotation differs by {worst_r:.3e} rad"


@pytest.mark.parametrize("side", ["left", "right"])
def test_chain_matches_urdf(side: str) -> None:
    """Every one of the 7 joints: origin xyz, rpy and axis, straight from the URDF.

    Walked link by link from ``torso_link`` rather than matched by name, so the
    order is the URDF's own chain order.
    """
    root = ET.parse(URDF).getroot()
    children: dict[str, list] = {}
    for joint in root.findall("joint"):
        children.setdefault(joint.find("parent").get("link"), []).append(joint)

    chain = left_arm_chain() if side == "left" else right_arm_chain()
    link = "torso_link"
    for index, entry in enumerate(chain):
        on_side = [j for j in children.get(link, []) if j.get("name", "").startswith(f"{side}_")]
        assert len(on_side) == 1, f"expected exactly one {side} joint under {link}, got {on_side}"
        joint = on_side[0]
        origin = joint.find("origin")
        xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]
        rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
        axis = [float(v) for v in joint.find("axis").get("xyz").split()]
        name = joint.get("name")
        # 1e-9: the chain stores exactly what the URDF prints (15 digits).
        np.testing.assert_allclose(entry.xyz, xyz, atol=1e-9, err_msg=f"{side} {name} xyz")
        np.testing.assert_allclose(entry.rpy, rpy, atol=1e-9, err_msg=f"{side} {name} rpy")
        np.testing.assert_allclose(entry.axis, axis, atol=1e-9, err_msg=f"{side} {name} axis")
        link = joint.find("child").get("link")
    assert link == f"{side}_wrist_roll_link"


@pytest.mark.parametrize("side", ["left", "right"])
def test_model_limits_match_catalog(side: str) -> None:
    """The MJCF ranges the solver clips against are the deploy's catalog."""
    solver = MinkArmIK(side)
    _chain, lo, hi = _chain_and_bounds(side)
    ids = [solver.model.joint(f"{side}_{name}").id for name in JOINT_NAMES]
    np.testing.assert_allclose(solver.model.jnt_range[ids, 0], lo, atol=1e-9)
    np.testing.assert_allclose(solver.model.jnt_range[ids, 1], hi, atol=1e-9)
    np.testing.assert_allclose(solver._lo, lo, atol=1e-9)
    np.testing.assert_allclose(solver._hi, hi, atol=1e-9)


def test_joint_ranges_are_not_a_qp_constraint() -> None:
    """Joint ranges stay a post-step clip, deliberately.

    Handing them to ``ConfigurationLimit`` was tried and measured: once
    ``wrist_pitch`` binds, the QP re-allocates the saturated rotation request
    over the whole arm and the hand leaves the operator's target by up to
    50.9 mm on a 3 s rotation drive (0.562 rad max step difference against
    DLS, 11x this migration's A/B budget). See the module docstring.
    """
    solver = MinkArmIK("right")
    assert not any(isinstance(limit, mink.ConfigurationLimit) for limit in solver._limits)
    assert any(isinstance(limit, mink.VelocityLimit) for limit in solver._limits)


def test_out_of_range_step_is_clipped() -> None:
    """Whatever the QP returns, the commanded q is inside the catalog."""
    solver = MinkArmIK("right")
    _chain, lo, hi = _chain_and_bounds("right")
    # A target far outside the reachable sphere pulls every joint to a bound.
    q_out, _ = solver.solve(np.zeros(7), np.array([0.9, -0.9, 0.9]), None, max_iters=4)
    assert np.all(q_out >= lo - 1e-12) and np.all(q_out <= hi + 1e-12)
    assert np.any(np.isclose(q_out, lo) | np.isclose(q_out, hi)), "nothing reached a bound"


# ── solver contracts ──────────────────────────────────────────────────────


def test_position_only_branch() -> None:
    """``target_quat_wxyz=None`` drops the angular rows, reports rot_err 0, and
    lands on the legacy solver's answer.

    The seed is bent (elbow -0.8) on purpose. The right elbow's range is
    [-2.3556, 0], so q = 0 sits exactly *on* its upper bound and the arm is
    straight: from there, shortening the arm is a second-order motion (the
    elbow column is 0.2025 long in x but 0.0137 in z, the extension axis), so
    a target that needs the arm to come in converges at ~3% a step and parks
    on the singular direction -- identically in both backends, which is why
    the legacy suite calls that a documented limit rather than a bug. A
    lateral target from a bent seed is the well-conditioned case.
    """
    from gear_sonic.utils.teleop.solver.arm.x2_arm_ik import ArmIKSolver

    solver = MinkArmIK("right")
    legacy = ArmIKSolver("right")
    q_seed = np.zeros(7)
    q_seed[3] = -0.8
    p0, _ = solver.fk(q_seed)
    target = p0 - np.array([0.03, 0.0, 0.0])

    q, result = solver.solve(q_seed, target, None, max_iters=10)
    p1, _ = solver.fk(q)
    assert np.linalg.norm(target - p1) < 1e-3
    assert result.pos_err_m < 1e-3
    assert result.rot_err_rad == 0.0  # no orientation target -> no rotation error
    assert 1 <= result.iterations <= 10

    q_legacy, r_legacy = legacy.solve(q_seed, target, None, max_iters=10)
    p_legacy, _ = legacy.fk(q_legacy)
    assert np.linalg.norm(p1 - p_legacy) < 1e-5, "position-only solves land in different places"
    # The joint vectors agree except on wrist_pitch, where the two error forms
    # differ structurally: its origin IS the wrist centre, so the legacy
    # decoupled Jacobian has an exactly-zero position column there, while
    # mink's SE(3)-log mixes a sliver of the angular row into the linear ones
    # (0.6 mrad here -- a joint with no position authority).
    assert np.abs(q - q_legacy).max() < 1e-3
    assert np.abs((q - q_legacy)[[0, 1, 2, 3, 4, 6]]).max() < 1e-5


def test_max_iters_composition() -> None:
    """Two one-iteration solves == one two-iteration solve, bit for bit.

    The 50 Hz recorder calls the solver per tick with ``max_iters=1`` and the
    warm-up paths ask for several; the two have to mean the same thing.
    """
    solver = MinkArmIK("right")
    q_seed = np.zeros(7)
    p0, R0 = solver.fk(q_seed)
    target = p0 + np.array([0.02, 0.03, -0.03])
    quat = _rotated_quat(R0, [0.0, 0.0, 1.0], 0.087)  # 5 deg, inside the cap
    q_once, r_once = solver.solve(q_seed, target, quat, max_iters=1)
    q_twice, r_twice = solver.solve(q_once, target, quat, max_iters=1)
    q_combined, r_combined = solver.solve(q_seed, target, quat, max_iters=2)
    np.testing.assert_array_equal(q_twice, q_combined)
    assert (r_once.iterations, r_twice.iterations, r_combined.iterations) == (1, 1, 2)


def test_qp_failure_falls_back_to_the_seed() -> None:
    """A dead QP backend must not raise into the 50 Hz loop.

    ``inf`` error latches the freeze guard, which holds the last command --
    the same place a diverging DLS ends up.
    """
    solver = MinkArmIK("right", solver="not-a-qp-backend")
    q_seed = np.zeros(7)
    p0, _ = solver.fk(q_seed)
    q, result = solver.solve(q_seed, p0 + np.array([0.03, 0.0, 0.0]), None, max_iters=3)
    np.testing.assert_array_equal(q, q_seed)
    assert np.isinf(result.pos_err_m) and np.isinf(result.rot_err_rad)
    assert result.iterations == 0


def test_saturation_below_cap_is_a_noop() -> None:
    """Below the cap the two solvers are bit-identical (the round-8.1 bound)."""
    q_seed = np.zeros(7)
    capped = MinkArmIK("right", max_rotation_err_rad=0.08)
    uncapped = MinkArmIK("right", max_rotation_err_rad=0.0)
    p0, R0 = capped.fk(q_seed)
    quat = _rotated_quat(R0, [1.0, 0.0, 0.0], 0.05)  # 0.05 rad, under the 0.08 cap
    target = p0 + np.array([0.01, 0.0, 0.0])
    q_a, r_a = capped.solve(q_seed, target, quat, max_iters=1)
    q_b, r_b = uncapped.solve(q_seed, target, quat, max_iters=1)
    np.testing.assert_array_equal(q_a, q_b)
    assert r_a.rot_err_rad == r_b.rot_err_rad


def test_saturation_reaches_the_qp() -> None:
    """Above the cap the two solvers MUST differ: proof the cap is not inert.

    mink's ``solve_ik`` prefers ``task.compute_qp_residual``; a cap wired into
    a seam that path skips is dead code that still passes every
    below-threshold assertion.
    """
    q_seed = np.zeros(7)
    capped = MinkArmIK("right", max_rotation_err_rad=0.05)
    uncapped = MinkArmIK("right", max_rotation_err_rad=0.0)
    p0, R0 = capped.fk(q_seed)
    quat = _rotated_quat(R0, [0.0, 1.0, 0.0], np.radians(60.0))  # far above any cap
    q_a, _ = capped.solve(q_seed, p0, quat, max_iters=1)
    q_b, _ = uncapped.solve(q_seed, p0, quat, max_iters=1)
    assert not np.allclose(q_a, q_b), "the saturation cap never reached the QP"
    # And the capped request is the smaller one: it asks for less rotation.
    assert np.linalg.norm(q_a - q_seed) < np.linalg.norm(q_b - q_seed)


# ── backend selection ─────────────────────────────────────────────────────


def test_backend_resolution_order(monkeypatch) -> None:
    monkeypatch.delenv("X2_IK_BACKEND", raising=False)
    assert resolve_arm_backend() == "mink"  # mink is importable here
    assert resolve_arm_backend("dls") == "dls"  # argument wins
    monkeypatch.setenv("X2_IK_BACKEND", "DLS")
    assert resolve_arm_backend() == "dls"  # env, case-insensitive
    assert resolve_arm_backend("mink") == "mink"  # argument still wins
    monkeypatch.setenv("X2_IK_BACKEND", "bogus")
    with pytest.raises(ValueError, match="unknown arm IK backend"):
        resolve_arm_backend()


def test_legacy_only_kwargs_are_dropped_for_mink() -> None:
    """``uniform_step_clip`` has no mink meaning; passing it is not an error.

    It is in the shared call site because the DLS backend needs it, so the
    factory filters rather than the call site branching.
    """
    solver, backend = make_arm_solver(
        "right",
        backend="mink",
        uniform_step_clip=True,
        null_space_gain=0.0,
        q_preferred=[0.0] * 7,
    )
    assert backend == "mink"
    assert solver.backend == "mink"
    assert not hasattr(solver, "uniform_step_clip")
