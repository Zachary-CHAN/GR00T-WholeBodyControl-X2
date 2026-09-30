"""Operator root-tilt guard: clamp invariants + the live insertion point.

The guard exists because both bundled Pico tapes make the X2 fall
deterministically: a recorded operator root tilt past ~30 deg held for longer
than ~1.5 s makes the policy track the lean into the deploy tilt watchdog
4.22 +/- 0.11 s after the event onset (2 tapes / 3 cut points / 4 runs,
residual < 0.13 s).  See gear_sonic/utils/teleop/root_tilt_guard.py for the
frame proof and why the cap is a swing (not Euler) clamp.

Three things must never regress:

1. THE ROOT CLAMP.  Over-cap frames land exactly on the cap; under-cap frames
   pass through BIT-IDENTICAL (a clean session must see zero disturbance); the
   lean AZIMUTH and the heading (twist) do not move -- at the critical moment a
   rotated azimuth would change which way the robot falls.
   ⚠ When writing these, split frames on the INPUT tilt, not the output: a
   frame clamped to exactly LIM reads as "under cap" and gets misclassified.
   ⚠ Do NOT measure heading with as_euler("ZYX")[0] -- gimbal lock near
   pitch=+-90 reports a phantom 180 deg error.

2. THE TORSO CLAMP.  The root channel does NOT carry the whole lean: with the
   root pinned perfectly upright the world pelvis->head direction still
   reaches 28.9 deg on seg01 (the operator's own spine bend) against a 30 deg
   fatal line, and 44.7 deg at the 25 deg default.  So cap_torso_lean rotates
   the upper-body subtree to bound every spine joint in the WORLD, and must
   leave the pelvis and both legs byte-identical -- the stance the policy is
   asked to hold does not change, only how far the operator is folded.

3. THE INSERTION POINT.  LiveSmplSource.window() / SmplFileSource.window() are
   the single place both consumers read from (pico_intent_sender.py's wire
   quat, build_smpl_obs' 6-D root block AND 72-value joint block), and
   --tape-replay goes through the same call -- so guarding there protects live
   teleop and tape replay at once.  If someone moves the guard out of
   window(), or drops either channel from it, test_window_insertion fails.

    .venv/bin/python -m pytest gear_sonic/tests/test_root_tilt_clamp.py -q
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as sRot

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "gear_sonic" / "scripts"
for p in (str(REPO_ROOT), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from gear_sonic.utils.teleop.root_tilt_guard import (  # noqa: E402
    SPINE_CHAIN, TORSO_SUBTREE, UP, _at_floor, cap_root_tilt, cap_torso_lean,
    clamp_stats, guard_window_pose, guard_window_quats, ramp_seconds,
    ramp_strength, set_root_tilt_cap, set_root_tilt_ramp, tilt_deg,
)

LIM = 25.0


# --------------------------------------------------------------------------
# vectorised helpers (the module's own decomposition, recomputed independently)
# --------------------------------------------------------------------------
def _up_axis(R: sRot) -> np.ndarray:
    return np.atleast_2d(R.apply(UP))


def _swing(R: sRot) -> sRot:
    """The tilt part of R = swing * twist (twist fixes UP)."""
    u = _up_axis(R)
    ax = np.cross(UP, u)
    n = np.linalg.norm(ax, axis=1, keepdims=True)
    th = np.arccos(np.clip(u @ UP, -1.0, 1.0))
    degen = n[:, 0] < 1e-9
    ax = np.where(degen[:, None], np.array([1.0, 0.0, 0.0]), ax)
    n = np.where(degen[:, None], 1.0, n)
    return sRot.from_rotvec(th[:, None] * ax / n)


def _azimuth(R: sRot) -> np.ndarray:
    """Direction the up-axis leans toward, as an angle in the horizontal plane."""
    h = _up_axis(R)
    h = h - (h @ UP)[:, None] * UP
    return np.arctan2(h[:, 1], h[:, 0])


def _angle_between(a: sRot, b: sRot) -> np.ndarray:
    return np.degrees(np.linalg.norm((a.inv() * b).as_rotvec(), axis=1))


def _wxyz_to_rot(q: np.ndarray) -> sRot:
    return sRot.from_quat(np.asarray(q, np.float64)[:, [1, 2, 3, 0]])


DISTS = [
    ("uniform SO(3)", lambda r: sRot.random(random_state=r)),
    ("N(0,12deg) live-like", lambda r: sRot.from_rotvec(r.normal(0, np.radians(12), 3))),
    ("N(0,40deg) fall-like", lambda r: sRot.from_rotvec(r.normal(0, np.radians(40), 3))),
]


@pytest.mark.parametrize("name,gen", DISTS)
def test_clamp_invariants(name, gen):
    """Hard cap, bit-exact pass-through below it, azimuth and heading preserved."""
    rng = np.random.default_rng(1)
    Q = np.stack([gen(rng).as_quat() for _ in range(4000)])[:, [3, 0, 1, 2]].astype(np.float32)

    out, hit = cap_root_tilt(Q, LIM, return_mask=True)
    Rin, Rout = _wxyz_to_rot(Q), _wxyz_to_rot(out)

    # 1. hard cap.  Note this counts frames the clamp produced; the raw input
    #    of course still exceeds it.
    assert tilt_deg(Rout).max() <= LIM + 1e-4, f"{name}: cap not hard"

    # 2. under-cap frames pass through bit-identical -- split on the INPUT tilt,
    #    or a frame clamped to exactly LIM is misread as "under cap".
    under = tilt_deg(Rin) <= LIM
    assert np.array_equal(out[under], Q[under]), f"{name}: disturbed an under-cap frame"
    assert hit.sum() == int((~under).sum()), f"{name}: hit mask disagrees with input tilt"

    # 3. the cap lands ON the limit, not near it
    assert np.abs(tilt_deg(Rout)[hit] - LIM).max() < 1e-3, f"{name}: clamped off the limit"

    # 4. lean azimuth unmoved; heading (twist) unmoved
    mv = np.flatnonzero(hit)
    if len(mv):
        d_az = np.degrees(np.abs(np.angle(np.exp(1j * (_azimuth(Rout[mv]) - _azimuth(Rin[mv]))))))
        assert d_az.max() < 1e-3, f"{name}: azimuth drifted {d_az.max():.2e} deg"
        want = _swing(Rout[mv]) * _swing(Rin[mv]).inv() * Rin[mv]
        d_hd = _angle_between(Rout[mv], want)
        assert d_hd.max() < 1e-3, f"{name}: heading drifted {d_hd.max():.2e} deg"


def test_cap_off_is_a_true_off_switch():
    """--root-tilt-limit-deg 0 must be bit-identical, and must not alias input."""
    rng = np.random.default_rng(3)
    # float32, like the ring the live path feeds it; the function is documented
    # to return float32, so a float64 input would compare unequal on dtype.
    Q = np.stack([sRot.random(random_state=rng).as_quat()
                  for _ in range(200)])[:, [3, 0, 1, 2]].astype(np.float32)
    assert np.array_equal(cap_root_tilt(Q, 0.0), Q)
    assert np.array_equal(cap_root_tilt(Q, -1.0), Q)

    # float64 input must survive: cap_root_tilt writes clamped rows in place.
    Q64 = Q.astype(np.float64)
    before = Q64.copy()
    cap_root_tilt(Q64, LIM)
    assert np.array_equal(Q64, before), "clamp mutated the caller's array"

    assert cap_root_tilt(Q[0], LIM).shape == (4,), "single-quat input lost its shape"


def _lean(R: sRot, v: np.ndarray) -> float:
    """Angle of R @ v from the world up axis (the frame the wire quats live in)."""
    w = R.apply(v)
    return float(np.degrees(np.arccos(np.clip(w @ UP / np.linalg.norm(w), -1.0, 1.0))))


def _torso_frames(rng, n=400, fold_deg=85.0):
    """(n,24,3) root-local skeletons with a random spine bend added.

    Proportions follow a real upright frame off seg02 (`smpl_joints_local`):
    the root-local up axis is +Z (the pelvis->head unit vector measures
    [0.003, 0.017, 0.9998] on upright frames), the spine climbs +Z and the legs
    hang -Z.
    """
    base = np.zeros((n, 24, 3))
    base[:, 0] = np.array([0.237, -0.260, 0.0])                 # pelvis
    for j, z in ((3, 0.11), (6, 0.24), (9, 0.29), (12, 0.45), (15, 0.61)):
        base[:, j] = np.array([0.23, -0.26, z])
    for j, (x, z) in ((13, (0.03, 0.37)), (14, (-0.03, 0.37)),
                      (16, (0.05, 0.37)), (17, (-0.05, 0.37)),
                      (18, (0.15, 0.37)), (19, (-0.15, 0.37)),
                      (20, (0.27, 0.37)), (21, (-0.27, 0.37)),
                      (22, (0.33, 0.37)), (23, (-0.33, 0.37))):
        base[:, j] = np.array([0.23 + x, -0.26, z])
    for j, (x, z) in ((1, (0.09, -0.09)), (2, (-0.09, -0.09)),
                      (4, (0.10, -0.45)), (5, (-0.10, -0.45)),
                      (7, (0.11, -0.87)), (8, (-0.11, -0.87)),
                      (10, (0.12, -0.92)), (11, (-0.12, -0.92))):
        base[:, j] = np.array([0.23 + x, -0.26, z])
    # fold the spine about the pelvis by a random angle per frame, about a
    # random HORIZONTAL axis (a rotation about +Z would be a twist, not a lean)
    ang = rng.uniform(0.0, np.radians(fold_deg), n)
    ax = rng.normal(size=(n, 3))
    ax[:, 2] = 0.0
    ax /= np.maximum(np.linalg.norm(ax, axis=1, keepdims=True), 1e-9)
    Rf = sRot.from_rotvec(ang[:, None] * ax)
    sub = base[:, list(TORSO_SUBTREE), :] - base[:, :1, :]
    flat = sRot.from_quat(np.repeat(Rf.as_quat(), len(TORSO_SUBTREE), axis=0))
    base[:, list(TORSO_SUBTREE), :] = (
        base[:, :1, :] + flat.apply(sub.reshape(-1, 3)).reshape(n, len(TORSO_SUBTREE), 3))
    return base


def test_torso_cap_bounds_the_world_lean():
    """The torso clamp must bound EVERY spine joint, and move nothing else.

    A rigid rotation moves the whole chain, so clipping one joint does not
    bound the rest -- the loop in cap_torso_lean re-picks the worst until all
    of them are inside.  That is the property under test.
    """
    rng = np.random.default_rng(11)
    J = _torso_frames(rng)
    # a random world root per frame, then root-clamped exactly as production does
    Q = np.stack([sRot.random(random_state=rng).as_quat() for _ in range(len(J))])[:, [3, 0, 1, 2]]
    Q = cap_root_tilt(Q.astype(np.float32), LIM)
    R = _wxyz_to_rot(Q)

    Jc, hit = cap_torso_lean(J.astype(np.float32), Q, LIM, return_mask=True)

    # 1. every spine joint, in the world, is inside the cap
    worst_o = worst_c = 0.0
    for j in SPINE_CHAIN:
        v_o, v_c = J[:, j] - J[:, 0], Jc[:, j] - Jc[:, 0]
        worst_o = max(worst_o, max(_lean(R[k], v_o[k]) for k in range(len(J))))
        worst_c = max(worst_c, max(_lean(R[k], v_c[k]) for k in range(len(J))))
    assert worst_c <= LIM + 1e-3, f"torso cap not hard: {worst_c:.2f} deg"
    assert worst_o > LIM + 5.0, f"test data never exceeded the cap ({worst_o:.1f} deg)"

    # 2. pelvis and legs are bit-identical -- the stance is untouched
    still = [0, 1, 2, 4, 5, 7, 8, 10, 11]
    assert np.array_equal(Jc[:, still], J.astype(np.float32)[:, still]), \
        "the torso clamp moved the pelvis or a leg"

    # 3. the skeleton is undeformed: a rigid rotation preserves every length
    #    inside the subtree, and every distance from the pelvis
    for j in TORSO_SUBTREE:
        d_o = np.linalg.norm(J[:, j] - J[:, 0], axis=1)
        d_c = np.linalg.norm(Jc[:, j] - Jc[:, 0], axis=1)
        assert np.abs(d_o - d_c).max() < 1e-5, f"joint {j} changed its distance from the pelvis"
    a, b = TORSO_SUBTREE[0], TORSO_SUBTREE[-1]
    assert np.abs(np.linalg.norm(J[:, a] - J[:, b], axis=1)
                  - np.linalg.norm(Jc[:, a] - Jc[:, b], axis=1)).max() < 1e-5, \
        "the torso was deformed, not rotated"

    # 4. a frame already inside the cap passes through bit-identical
    under = np.array([max(_lean(R[k], (J[k, j] - J[k, 0])) for j in SPINE_CHAIN)
                      for k in range(len(J))]) <= LIM
    assert np.array_equal(Jc[under], J.astype(np.float32)[under]), \
        "disturbed a frame that was already inside the cap"
    assert hit.sum() == int((~under).sum()), "hit mask disagrees with the input lean"


def test_torso_cap_off_is_a_true_off_switch():
    """cap 0 must be bit-identical and must not alias the caller's array."""
    rng = np.random.default_rng(13)
    J = _torso_frames(rng, 60).astype(np.float32)
    Q = np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (len(J), 1))
    assert np.array_equal(cap_torso_lean(J, Q, 0.0), J)
    assert np.array_equal(cap_torso_lean(J, Q, -1.0), J)

    J64 = J.astype(np.float64)
    before = J64.copy()
    cap_torso_lean(J64, Q.astype(np.float64), LIM)
    assert np.array_equal(J64, before), "the torso clamp mutated the caller's array"

    # the combined live entry point must be the identity in BOTH channels
    # (read the old cap from clamp_stats -- set_root_tilt_cap returns the NEW
    # one, so `prev = set_root_tilt_cap(0)` would restore 0 and leak it into
    # every later test in the session)
    prev = clamp_stats()[-1]
    set_root_tilt_cap(0)
    try:
        j2, q2 = guard_window_pose(J, Q)
        assert j2 is J and q2 is Q, "guard_window_pose copied or rewrote an input while off"
    finally:
        set_root_tilt_cap(prev)


def _spine_at(tilts_deg):
    """One skeleton whose 5 spine joints point at these world tilts.

    The root quat is identity, so root-local == world and the tilts are exactly
    what cap_torso_lean measures.  Positive leans toward +X, negative toward -X,
    so mixing signs puts two active joints in opposite azimuths -- the geometry
    that makes the correction a min-max instead of a single-joint fix.
    """
    J = _torso_frames(np.random.default_rng(0), 1, fold_deg=0.0)
    for j, t in zip(SPINE_CHAIN, tilts_deg):
        a = np.radians(t)
        J[0, j] = J[0, 0] + np.array([np.sin(a), 0.0, np.cos(a)]) * 0.15
    return J


def _worst(Jc):
    v = Jc[0, list(SPINE_CHAIN)] - Jc[0, 0]
    return max(_lean(sRot.identity(), v[k]) for k in range(len(SPINE_CHAIN)))


def _min_cone_deg(tilts_deg):
    """Smallest half-angle cone containing the spine directions, by brute force.

    This is the floor cap_torso_lean is trying to reach: rotating the chain
    rigidly cannot beat it, because a rotation preserves every angle between
    the joints.
    """
    dirs = np.stack([np.array([np.sin(np.radians(t)), 0.0, np.cos(np.radians(t))])
                     for t in tilts_deg])

    def cost(a):
        a = a / np.linalg.norm(a)
        return float(np.degrees(np.arccos(np.clip(dirs @ a, -1.0, 1.0))).max())

    best = min(cost(np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)]))
               for t in np.linspace(0.0, np.pi, 181)
               for p in np.linspace(0.0, 2 * np.pi, 361, endpoint=False))
    return best


@pytest.mark.parametrize("tilts,floor", [
    ([30.0, 20.0, 0.0, -10.0, -25.0], 27.5),     # two active joints, opposite sides
    ([35.0, 20.0, 0.0, -20.0, -35.0], 35.0),     # infeasible: spans 70 > 2*LIM
    ([40.0, 30.0, 20.0, 10.0, 0.0], 25.0),       # one-sided, easily feasible
])
def test_torso_cap_reaches_the_min_max(tilts, floor):
    """The correction must land ON the min-max, not beside it.

    The loop is a subgradient method on a min-max, and both simpler loops fail
    it in a way that is invisible on inspection:

      * a FULL step (rotate by the whole deficit) ping-pongs between the two
        active joints where they sit in opposite azimuths, and stalls at
        30.51 deg on seg02 -- over the fatal line;
      * a CONSTANT damped step cures the flip but settles at a biased fixed
        point: constant 0.5 lands this first case at 28.333 when the true
        min-max is 27.500.

    Eta therefore shrinks with the pass.  The floor here is brute-forced, so
    this test fails if the schedule is ever flattened back to a constant.

    Tolerance is 0.2: the schedule is finite, so it stops just short -- worst
    measured residual is 0.15 on the infeasible case, and 0.04 on the first.
    """
    Q = np.array([[1.0, 0.0, 0.0, 0.0]], np.float32)
    Jc = cap_torso_lean(_spine_at(tilts).astype(np.float32), Q, LIM)
    got = _worst(Jc)
    assert got <= floor + 0.2, f"scheduled steps did not reach the min-max: {got:.3f}"


def test_torso_cap_is_continuous_across_the_two_joint_fold():
    """A tiny input change must not flip which joint the correction targets.

    This is the property the full step loses, and it is not cosmetic: the flip
    is a jump in the commanded pose, arriving exactly when the operator is deep
    in the lean the guard exists to bound.  Two frames that differ by 0.01 deg
    must get corrections that differ by far less than the oscillation.
    """
    Q = np.array([[1.0, 0.0, 0.0, 0.0]], np.float32)
    base = [30.0, 20.0, 0.0, -10.0, -25.0]

    def applied(delta):
        J = _spine_at([t + delta * (1.0 if i == 0 else -1.0) for i, t in enumerate(base)])
        Jc = cap_torso_lean(J.astype(np.float32), Q, LIM)
        a, b = J[0, 15] - J[0, 0], Jc[0, 15] - Jc[0, 0]
        return sRot.align_vectors([b], [a])[0]

    for delta in (1e-4, 1e-2):
        d = np.degrees(np.linalg.norm((applied(0.0).inv() * applied(delta)).as_rotvec()))
        assert d < 0.05, f"correction jumped {d:.3f} deg for a {delta:g} deg input change"


def test_at_floor_counts_only_the_unreachable_frames():
    """_at_floor is the guard reporting it ran out of authority, so it must fire
    on exactly the frames a rigid rotation cannot bring under the cap.

    Whether a frame is reachable is not a judgement call -- it is whether the
    smallest cone containing the spine directions fits in LIM, which
    _min_cone_deg brute-forces.  The labels below are checked against it rather
    than trusted: the first draft of this test called a 55 deg fold "feasible"
    and the assertion failed on the test data, not on the code.
    """
    Q = np.array([[1.0, 0.0, 0.0, 0.0]], np.float32)
    for tilts, unreachable in (([10.0, 8.0, 4.0, 2.0, 0.0], False),      # under the cap
                               ([40.0, 30.0, 20.0, 10.0, 0.0], False),    # spans 40 < 2*LIM
                               ([30.0, 20.0, 0.0, -10.0, -25.0], True),   # spans 55 > 2*LIM
                               ([35.0, 20.0, 0.0, -20.0, -35.0], True)):  # spans 70 > 2*LIM
        assert (min(_min_cone_deg(tilts), 90.0) > LIM) is unreachable, \
            f"{tilts}: this case is labelled wrong, not the code"
        Jc = cap_torso_lean(_spine_at(tilts).astype(np.float32), Q, LIM)
        assert _at_floor(Jc, Q) == int(unreachable), \
            f"{tilts}: floor reported {'on a reachable frame' if not unreachable else 'nowhere'}"


def test_guard_window_quats_contract():
    """The live wrapper: off = untouched, under-cap = bit-identical, cap = on."""
    rng = np.random.default_rng(7)
    upright = np.array([1.0, 0.0, 0.0, 0.0], np.float32)
    leaning = sRot.from_euler("x", 60.0, degrees=True).as_quat()[[3, 0, 1, 2]].astype(np.float32)

    prev = clamp_stats()[-1]          # NOT the return of set_root_tilt_cap: new value
    set_root_tilt_cap(LIM)
    try:
        seen0, capped0, _, _, _ = clamp_stats()

        w = np.stack([upright] * 4 + [leaning])
        out = guard_window_quats(w)
        assert np.array_equal(out[:4], w[:4]), "clean frames were disturbed"
        assert abs(float(tilt_deg(_wxyz_to_rot(out[4:]))[0]) - LIM) < 1e-3
        seen1, capped1, _, _, _ = clamp_stats()
        assert (seen1 - seen0, capped1 - capped0) == (5, 1), "stats did not count the window"

        # a window with nothing over the cap comes back bit-identical
        clean = np.stack([upright] * 3)
        assert np.array_equal(guard_window_quats(clean), clean)

        # OFF returns the caller's own object -- no copy, no dtype change
        assert set_root_tilt_cap(0) == 0
        assert guard_window_quats(w) is w
    finally:
        set_root_tilt_cap(prev)


def test_window_insertion():
    """The guard must sit inside window() on BOTH sources (live and file).

    Both channels are checked, not just the root: the root clamp alone leaves
    the operator's fold in the 72-value joint block (see cap_torso_lean), so a
    window() that only guards `quats` is an armed failure and this test fails.
    """
    live = pytest.importorskip("live_pico_smpl_teleop",
                               reason="needs the sim venv (torch/mujoco)")

    upright = np.array([1.0, 0.0, 0.0, 0.0], np.float32)
    leaning = sRot.from_euler("y", 60.0, degrees=True).as_quat()[[3, 0, 1, 2]].astype(np.float32)
    # an UPRIGHT skeleton: the ring frames are clean, so the guard must leave
    # their joints alone and only touch the one leaning frame
    skel = _torso_frames(np.random.default_rng(5), 1, fold_deg=0.0)[0].astype(np.float32)

    # Ramp OFF for this test: the subject here is WHICH window() calls the
    # guard, and a memoryless cap makes the assertion a bit-exact one.  The
    # ramp's own contract is test_root_tilt_ramp_contract below.  (With the
    # ramp on, the first call of a fresh process is deliberately a pass-through
    # and a single call would test that instead of the insertion point.)
    prev_ramp = ramp_seconds()
    set_root_tilt_ramp(0.0)
    try:
        # --- LiveSmplSource.window: 10 frames at 50 Hz, ring (t, joints, quats, ...)
        ring = [(i * live.SMPL_DT, skel, leaning if i == 9 else upright)
                for i in range(10)]

        class _Src:
            _lock = threading.Lock()
            _ring = ring

        joints, quats = live.LiveSmplSource.window(_Src(), now=9 * live.SMPL_DT)
        t = tilt_deg(_wxyz_to_rot(quats))
        assert abs(float(t[9]) - LIM) < 1e-3, "LiveSmplSource.window did not clamp the root"
        assert np.array_equal(quats[:9], np.stack([upright] * 9)), "clean frames were disturbed"
        # the leaning frame's torso is corrected; the upright frames are untouched
        assert np.array_equal(joints[:9], np.stack([skel] * 9)), "clean frames' joints were disturbed"
        R9 = _wxyz_to_rot(quats[9:10])[0]
        worst = max(_lean(R9, joints[9, j] - joints[9, 0]) for j in SPINE_CHAIN)
        assert worst <= LIM + 1e-3, f"LiveSmplSource.window left a {worst:.1f} deg world torso lean"
        still = [0, 1, 2, 4, 5, 7, 8, 10, 11]
        assert np.array_equal(joints[9, still], skel[still]), "the pelvis or a leg moved"

        # --- SmplFileSource.window
        src = object.__new__(live.SmplFileSource)
        src._joints = np.stack([skel] * 10)
        src._quats = np.stack([leaning if i == 0 else upright for i in range(10)])
        src._t0 = None
        src._end_announced = False
        joints, quats = src.window(now=0.0)
        assert tilt_deg(_wxyz_to_rot(quats)).max() <= LIM + 1e-4, \
            "SmplFileSource.window did not clamp"
        R0 = _wxyz_to_rot(quats[0:1])[0]
        worst = max(_lean(R0, joints[0, j] - joints[0, 0]) for j in SPINE_CHAIN)
        assert worst <= LIM + 1e-3, \
            f"SmplFileSource.window left a {worst:.1f} deg world torso lean"
    finally:
        set_root_tilt_ramp(prev_ramp)
        set_root_tilt_cap(LIM)


# --------------------------------------------------------------------------
# the soft-start ramp
# --------------------------------------------------------------------------
def _leaning_window(deg=60.0, n=10):
    """One over-cap pose repeated n times, as (joints (n,24,3), quats (n,4))."""
    J = _torso_frames(np.random.default_rng(0), 1, fold_deg=0.0)
    J = np.repeat(J, n, axis=0).astype(np.float32)
    q = sRot.from_euler("y", deg, degrees=True).as_quat()[[3, 0, 1, 2]].astype(np.float32)
    return J, np.tile(q, (n, 1))


def _upright_window(n=10):
    J = _torso_frames(np.random.default_rng(0), 1, fold_deg=0.0)
    J = np.repeat(J, n, axis=0).astype(np.float32)
    q = np.array([1.0, 0.0, 0.0, 0.0], np.float32)
    return J, np.tile(q, (n, 1))


def test_ramp_zero_is_the_memoryless_clamp_bit_for_bit():
    """--root-tilt-ramp-s 0 must reproduce the old clamp exactly.

    Not approximately: the ramp is the fix for a fall the memoryless clamp
    caused, so the two must be distinguishable by the ramp's off switch alone,
    with nothing else moving.  That is what makes the A/B readable.
    """
    prev = ramp_seconds()
    try:
        set_root_tilt_cap(LIM)
        J, Q = _leaning_window()
        set_root_tilt_ramp(0.0)
        j0, q0 = guard_window_pose(J, Q, now=1.0)
        j0b, q0b = guard_window_pose(J, Q, now=None)
        set_root_tilt_ramp(0.6)
        j1, q1 = guard_window_pose(J, Q, now=None)      # now=None is memoryless
        assert np.array_equal(q0, q1), "now=None is not the memoryless clamp"
        assert np.array_equal(j0, j1), "now=None is not the memoryless clamp (joints)"
        assert np.array_equal(q0, q0b) and np.array_equal(j0, j0b), \
            "ramp 0 at now=1.0 differs from the memoryless path"
    finally:
        set_root_tilt_ramp(prev)
        set_root_tilt_cap(LIM)


def test_ramp_starts_at_zero_and_reaches_full_strength_in_t():
    """Strength is 0 on the first over-cap call, then 1.0 after T seconds.

    The first call contributing 0 is load-bearing, not an off-by-one: a fresh
    process that is ALREADY folded must not be clamped at full strength before
    it has seen a single frame of sustained lean -- that is the exact mistake
    the ramp exists to undo.
    """
    prev = ramp_seconds()
    try:
        set_root_tilt_cap(LIM)
        set_root_tilt_ramp(0.6)
        J, Q = _leaning_window()
        j1, q1 = guard_window_pose(J, Q, now=0.0)
        assert np.array_equal(q1, Q) and np.array_equal(j1, J), \
            "the first over-cap call clamped instead of starting at zero strength"

        now = 0.0
        for _ in range(29):                       # 0.58 s of calls at 50 Hz
            now += 0.02
            guard_window_pose(J, Q, now=now)
        assert ramp_strength() < 1.0, "reached full strength before T"
        now += 0.02
        guard_window_pose(J, Q, now=now)
        assert ramp_strength() == 1.0, "did not reach full strength at T"

        # at full strength the output IS the memoryless clamp, bit for bit
        j2, q2 = guard_window_pose(J, Q, now=now)
        set_root_tilt_ramp(0.0)
        jm, qm = guard_window_pose(J, Q, now=1.0)
        assert np.array_equal(q2, qm) and np.array_equal(j2, jm), \
            "full-strength ramp output differs from the memoryless clamp"
    finally:
        set_root_tilt_ramp(prev)
        set_root_tilt_cap(LIM)


def test_ramp_is_monotone_and_continuous_as_strength_grows():
    """A slowly growing strength must move the output monotonically onto the cap.

    A frame sitting at a fixed 60 deg with the strength rising from 0 to 1 has
    to land exactly on LIM, and every intermediate output must be inside
    [LIM, 60] on the same azimuth -- no overshoot past the cap, no jump.  A
    step here would be the ramp reintroducing the discontinuity the memoryless
    clamp was carefully built without.
    """
    prev = ramp_seconds()
    try:
        set_root_tilt_cap(LIM)
        J, Q = _leaning_window()
        az = None
        outs = []
        for s in np.linspace(0.0, 1.0, 11):
            out = cap_root_tilt(Q[:1], LIM, strength=s)
            outs.append(out)
            R = _wxyz_to_rot(out)[0]
            t = float(tilt_deg(R)[0])
            assert LIM - 1e-3 <= t <= 60.0 + 1e-4, f"strength {s}: {t:.3f} deg out of range"
            az_k = _azimuth(R)
            if az is None:
                az = az_k
            else:
                d = abs((az_k - az + 180.0) % 360.0 - 180.0)
                assert d < 1e-3, f"strength {s}: azimuth moved {d:.4f} deg"
        assert np.allclose([float(tilt_deg(_wxyz_to_rot(o))[0]) for o in outs],
                           np.linspace(60.0, LIM, 11), atol=1e-3), \
            "the output tilt is not linear in strength"
        # and the joints follow the same path for the torso channel
        prev_t = None
        for s in np.linspace(0.0, 1.0, 11):
            j2 = cap_torso_lean(J, Q, LIM, strength=s)
            cur = max(_lean(_wxyz_to_rot(Q[:1])[0], j2[0, j] - j2[0, 0])
                      for j in SPINE_CHAIN)
            assert cur <= 60.0 + 1e-4, f"strength {s}: torso lean {cur:.3f} grew"
            if prev_t is not None:
                assert cur <= prev_t + 1e-6, "torso lean did not fall monotonically"
            prev_t = cur
    finally:
        set_root_tilt_ramp(prev)
        set_root_tilt_cap(LIM)


def test_ramp_forgets_a_lean_once_it_clears():
    """A cleared lean must give the strength back before the next one arrives.

    Without the decay, one over-cap frame would arm every later event at full
    strength and the ramp would be a delay, not a duration test -- the blip at
    seg02 t=78 would then still be clamped full and still drop the robot.
    """
    prev = ramp_seconds()
    try:
        set_root_tilt_cap(LIM)
        set_root_tilt_ramp(0.5)
        over = _leaning_window()
        clear = _upright_window()
        now = 0.6
        for _ in range(40):                       # charge to full
            now += 0.02
            guard_window_pose(*over, now=now)
        assert ramp_strength() == 1.0
        for _ in range(40):                       # 0.8 s clear, > 0.5 s / decay
            now += 0.02
            guard_window_pose(*clear, now=now)
        assert ramp_strength() == 0.0, "the ramp kept its strength through a clear window"
        # and the cleared window was never touched on the way down
        j, q = guard_window_pose(*clear, now=now)
        assert np.array_equal(q, clear[1]) and np.array_equal(j, clear[0]), \
            "a clean window was disturbed while the ramp decayed"
    finally:
        set_root_tilt_ramp(prev)
        set_root_tilt_cap(LIM)
