r"""Operator lean cap for the Pico SMPL path (2026-09-30).

Bounds the lean the operator can command, on BOTH channels that carry it: the
root orientation (``guard_window_quats``) and the torso chain inside the
72-value joint block (``cap_torso_lean``).  See COST and TWO CHANNELS below.

WHY
---
The two bundled Pico tapes make the X2 fall, deterministically: whenever the
recorded operator root leaves upright by more than ~30 deg for longer than
~1.5 s, the robot follows the lean and trips the deploy tilt watchdog
4.22 +/- 0.11 s later (2 tapes / 3 cut points / 4 independent runs, residual
< 0.13 s).  Bursts shorter than 0.4 s never fell -- the discriminator is
DURATION, not magnitude.

Every existing protection is post-hoc.  `--tilt-cos -0.3` trips at 72.5 deg
from upright and safety.cpp:209-239 calls a trip a committed fall; and
`--max-target-dev` clamps only JOINT POSITIONS (safety.cpp:267-275).  Nothing
anywhere bounds the reference ORIENTATION.  This module is that missing
command-side bound: it caps the operator root tilt BEFORE the policy sees it,
with 4.22 s of margin instead of after the fact.

FRAME
-----
The cap runs on the wire quaternion -- the wxyz root orientation that
pico_intent_sender.py puts in the `human_quat` field.  In that frame the
operator's up axis is the BODY +Z.  The tapes were measured in the DEVICE
frame instead (up = +Y), and the two statistics are PROVABLY equal:

    R_wire = Rx(90) * R_device * Ry(180) * conj(120deg about (1,1,1)/sqrt3)
             \_____/   \______/   \_____/   \___________________________/
           smpl_root_  compute_    the y-180   remove_smpl_base_rot
           ytoz_up     from_body_   pre-rot
                       poses:297

`conj(120deg) @ e_z == e_y` (Rodrigues) and `Ry(180) @ e_y == e_y`, so
`R_wire @ e_z == Rx(90) @ (R_device @ e_y)`, and since `Rx(90) @ e_y == e_z`
both vectors are acted on by the same orthogonal map:

    arccos((R_wire @ e_z)[2]) == arccos((R_device @ e_y)[1])

Verified over all 19358 frames of session_20260917_014453Z_x2_seg02.npz:
max deviation 1.9e-05 deg (float32 noise).  A threshold measured on a tape
therefore transfers to this frame unchanged.

WHY SWING, NOT ZYX
------------------
In ZYX Euler the total tilt is `arccos(cos(pitch) * cos(roll))`, so clamping
pitch and roll to +-lim each does NOT bound it: at lim=30 the worst case is
41.4 deg.  The existing `LevelOperatorRoot clamp:<deg>`
(smpl_obs.cpp:82-96, pc2_pico_token_service.py:391-413) has exactly that
semantics, and measured on these tapes it still peaks at 40.7 deg (seg01)
while rotating the lean AZIMUTH by up to 18.4 deg -- at the critical moment
it would change which way the robot falls.  So this module caps the SWING
(total tilt) about the up axis and keeps the twist: it lands exactly on the
measured boundary and leaves heading untouched.

ONLINE IS THE ONE THAT COUNTS.  ``guard_window_quats`` is the live insertion
point: LiveSmplSource.window() and SmplFileSource.window() in
live_pico_smpl_teleop.py, which sit upstream of BOTH consumers of the root
channel -- pico_intent_sender.py's wire quat and build_smpl_obs' 6-D root block
-- so a session tape replayed through --tape-replay exercises the guard exactly
as a live operator would, and pico_token_sender.py gets it for free.
``cap_body_root_tilt`` (pico_tape_root_tilt_cap.py) rewrites a tape offline and
is a PREDICTOR only; it cannot reproduce the live obs bit-for-bit (see its
docstring).

TWO CHANNELS, NOT ONE
---------------------
The root cap is not sufficient on its own, and the measurement is unambiguous:
with the root pinned perfectly upright the root-LOCAL skeleton still produces
28.9 deg of pelvis->head tilt on seg01 -- a structural floor sitting 1.1 deg
under the 30 deg line, with the root channel contributing nothing.  The
operator's fold is carried by the 72-value joint block as much as by the 6-D
root block.  Root-only clamping left seg01 falling at 25 deg and again at
10 deg.  `cap_torso_lean` is the second channel: it bounds the WORLD lean of
the spine chain by rotating the torso subtree rigidly about the pelvis, so the
stance is untouched.  See its docstring for why the step size shrinks.

COST
----
Measured on this machine (Jetson, the sender's own calling pattern: a 10-frame
window sliding one frame per tick, CONTROL_DT = 20 ms):

    clean window (nothing over cap)      151 us     0.8% of the tick
    capped window, median               2670 us    13%
    capped window, worst                6071 us    30%

The worst case is set by the geometrically INFEASIBLE frames -- where the
operator's own spine spans more than 2*cap and no rigid rotation can reach the
cap -- because those never satisfy the exit test and run to _TORSO_ITERS.  They
are 0.300% of seg02 and 0% of seg01, they come back at the min-max of the
reachable cone (27.81 and 25.00 deg, both under the 30 deg line), and _at_floor
counts them so a guard out of authority is visible in the log.

Two things that look like easy wins here are NOT, and both were tried:

  - Opening with full steps (they clear the worst joint in one pass) takes
    seg01's worst frame from 128 iterations to 2 -- and reintroduces a 0.077 deg
    discontinuity for a 1e-4 deg input change, because the joint-selection flip
    comes back.  Speed is not worth a jump in the commanded pose.
  - A stall detector (stop when the max stops improving) leaves a synthetic
    fold at 45.0 deg against a 35.0 deg floor and seg02 at 29.0 -- 1 deg under
    the fatal line.  The k**-4 tail's per-step improvement drops below any
    sensible threshold long before it has converged, so "stalled" and "still
    crawling" are indistinguishable by that test.

What DID pay, exactly and with no change to the output: writing
``cross(u, [0,0,1])`` out as ``[u_y, -u_x, 0]`` (numpy's cross costs 17 us on
arrays this small), applying one rotation to K joints as a (3,3) matmul rather
than a repeated scipy stack, and accumulating the total rotation as a matrix.
Together: 12.0 ms -> 6.1 ms worst and 5.5 ms -> 2.9 ms median, with the cap
quality unchanged (the cross rewrite is bit-identical; the matmul path differs
by 4.4e-16, float64 epsilon and four orders below the float32 the result is
written as).
"""
from __future__ import annotations

import os as _os

import numpy as np
from scipy.spatial.transform import Rotation as Rot

# The three constant rotations composing the device -> wire chain (see FRAME).
# Kept as Rot objects so wire_from_device / device_from_wire are exact
# inverses of each other and of the production pipeline.
_A = Rot.from_euler("x", 90.0, degrees=True)     # smpl_root_ytoz_up
_B = Rot.from_euler("y", 180.0, degrees=True)    # compute_from_body_poses:297
_RB = Rot.from_quat(np.array([-0.5, -0.5, -0.5, 0.5]))   # conj(120deg about (1,1,1))

UP = np.array([0.0, 0.0, 1.0])                   # body-frame up, wire frame

# Operator ROOT TILT CAP (2026-09-30).  Degrees; 0 = off.
# 25 leaves 5 deg of margin under the measured 30 deg fall boundary and clips
# 2.6% of seg02 / 12.2% of seg01 (the tape with the sustained deep bend), none
# of normal teleop (median 4.9 deg, p95 17.4 deg).  Env knob so a live A/B
# needs no CLI edit on every entry point -- the sender, the sim driver,
# pico_token_sender.py and run_x2_pico_wbc.sh all construct LiveSmplSource
# from this module.
ROOT_TILT_CAP_DEG: float = float(_os.environ.get("PICO_ROOT_TILT_CAP_DEG", "25"))

# Seconds of CONTINUOUS over-cap lean needed for the clamp to reach full
# strength.  See the ramp note above guard_window_pose for why a memoryless
# clamp is not enough; 0 disables the ramp and restores it bit-for-bit.
#
# 0.6 is the value the tape A/B was run at (seg02, cap 25, TRIG 6.00); that run
# played the whole 234.5 s record with FATAL=0, removing BOTH the memoryless
# clamp's own fall at tape_t 78.52 and the unguarded one at 219.96/220.45.  Do
# not lower it without re-running that A/B: shortening the ramp moves the clamp
# toward the memoryless end, and THAT end is the one that fell -- the fatal
# 0.30 s blip is over-clamped, not under-clamped, so more clamping is the
# dangerous direction here.  At 0.30 s over cap this value yields strength
# 0.50; the memoryless clamp's 1.00 is what put the robot down.
ROOT_TILT_RAMP_S: float = float(_os.environ.get("PICO_ROOT_TILT_RAMP_S", "0.6"))

# How much faster the ramp falls than it rises, per second under the cap.
# Asymmetric on purpose: an over-cap frame is evidence the operator is leaning
# and should hold its ground, while a single under-cap frame is not evidence
# the lean is over -- a 10-frame window sliding at 20 ms spends most of a real
# event with a few frames still under.  At 2.0 a full rise is forgotten in half
# its rise time, which is longer than the ~0.1 s the blip spends dropping back
# under, so the blip cannot accumulate across calls.
_RAMP_DECAY = 2.0


def wire_from_device(body: np.ndarray) -> Rot:
    """(N,24,7) device body frames -> (N,) wire root rotations."""
    rd = Rot.from_quat(np.asarray(body, np.float64)[:, 0, [6, 3, 4, 5]],
                       scalar_first=True)
    return _A * (rd * _B) * _RB


def device_from_wire(rot: Rot) -> np.ndarray:
    """Inverse of wire_from_device. (N,) rotations -> (N,4) device quats, xyzw.

    Order matters: wire = A @ rd @ B @ RB, so (B @ RB)^-1 = RB^-1 @ B^-1.
    """
    return (_A.inv() * rot * _RB.inv() * _B.inv()).as_quat()


def tilt_deg(rot: Rot) -> np.ndarray:
    """Tilt from upright, in the SAME units the tapes were measured in.

    Accepts a single rotation or a stack; always returns an array.
    """
    z = np.atleast_2d(rot.apply(UP))[:, 2]
    return np.degrees(np.arccos(np.clip(z, -1.0, 1.0)))


def cap_body_root_tilt(body: np.ndarray, cap_deg: float,
                       return_mask: bool = False) -> np.ndarray | tuple:
    """Rigid-body form of the same cap, for REWRITING A TAPE.

    Why this exists and cap_root_tilt is not enough on a tape
    --------------------------------------------------------
    The pipeline derives root-local joints from WORLD-frame body poses:
    ``smpl_joints_local = R_root^-1 @ joints_world``.  Rotating only the root
    quaternion in a tape therefore leaves every child's world orientation where
    it was, and the re-derived ``smpl_joints_local`` absorbs the whole
    difference -- measured on seg01, ``pose_aa`` moved 60.6 deg and
    ``smpl_joints_local`` 0.86 m.  The world-space body lean is NOT reduced:
    the lean just moves from the root channel into the joint channel, which is
    also the channel the obs carries as its 72-value block.

    Rotating ALL 24 frames by the same correction fixes that: the whole raw
    stream turns rigidly, so the pipeline's FK comes out an EXACT rigid
    rotation of the original, about the SMPL rest pelvis J0:

        joints' == J0 + dR @ (joints - J0)        (measured 4e-7 m on seg01)

    THE PIVOT IS J0, NOT THE ORIGIN.  `transforms_mat[0]` carries
    ``rel_joints[0] = J0`` as the root's own translation, so ``joints[0] == J0``
    for every input rotation -- the FK spins the body about the pelvis.  A
    correction applied about the origin therefore misses by ``(I - dR) @ J0``
    (0.067 m at 61 deg, |J0| = 0.351 m), which looks exactly like a residual
    bug until you notice it is uniform across all 24 joints.  Assert the
    pelvis-pivot form, never ``dR @ joints``.

    Positions rotate about the pelvis too -- ``p_i' = p_0 + dR (p_i - p_0)`` --
    so the stream stays one rigid body and the pelvis translation is untouched.
    (It is also inert here: ``adjusted_transl`` has no consumer anywhere, and
    ``body[:, 1:, :3]`` is read by nothing.)

    CAVEAT, and why the live guard is still the one to trust
    -------------------------------------------------------
    This is NOT bit-for-bit the live guard's obs.  The guard clamps ``quats``
    in window() and leaves ``smpl_joints_local`` at its original value, so the
    obs it emits implies a rotation about the FK ORIGIN; this rewrite rotates
    about the PELVIS.  The two differ by exactly

        RB^-1 R_root^-1 (dR^-1 - I) @ J0

    -- a rigid translation of the whole local skeleton of norm
    ``|(dR^-1 - I) @ J0|`` (measured max 0.084 m on seg01 cut@10, 0.162 m on
    seg02 cut@220; bounded by ``2 |J0|`` = 0.70 m), with ZERO deformation.
    It is the pivot offset, not a shape change, and it cannot be removed from
    the tape side: the FK has no knob that adds ``(I - dR) @ J0`` to ``joints``
    (``transl`` never reaches it).  Use this tool to PREDICT -- tilt statistics,
    fall-time estimates, which events get clipped; run the actual
    effectiveness experiment ONLINE, with the guard live in window().

    body: (N,24,7) device body frames [pos(3) | quat xyzw(4)], modified copy.
    """
    out = np.array(body, np.float64, copy=True)
    Rw = wire_from_device(out)                                  # (N,) wire roots
    capped, hit = cap_root_tilt(Rw.as_quat()[:, [3, 0, 1, 2]], cap_deg, return_mask=True)
    if hit.any():
        # body[..., 3:7] is xyzw throughout this module.
        # dR in the wire frame, conjugated into the device frame by A, since
        # R_wire = A @ R_dev @ B @ RB  =>  dR_dev = A^-1 @ dR_wire @ A.
        dR = Rot.from_quat(capped[hit][:, [1, 2, 3, 0]]) * Rw[hit].inv()
        dRd = _A.inv() * dR * _A
        sub = out[hit]                                    # (M,24,7)
        m, j = sub.shape[0], sub.shape[1]
        # flatten the (M,24) grid explicitly: scipy is ambiguous on 3-D input
        # for both from_quat and apply.
        dR_flat = Rot.from_quat(np.repeat(dRd.as_quat(), j, axis=0))
        q_sub = Rot.from_quat(sub[:, :, 3:7].reshape(-1, 4))
        sub[:, :, 3:7] = (dR_flat * q_sub).as_quat().reshape(m, j, 4)
        # Rigid about the pelvis: p_i' = p_0 + dRd (p_i - p_0).  The pelvis
        # itself does not move, matching a rotation about J0 (and the live
        # guard, which edits quats and no positions at all).
        dp = (sub[:, :, :3] - sub[:, :1, :3]).reshape(-1, 3)
        # dR_flat is already the (M,24) grid of corrections -- the same stack
        # rotates the positions, one per joint.
        sub[:, :, :3] = sub[:, :1, :3] + dR_flat.apply(dp).reshape(m, j, 3)
        out[hit] = sub
    return (out, hit) if return_mask else out


def cap_root_tilt(quat_wxyz, cap_deg, return_mask: bool = False,
                  strength: float = 1.0):
    """Swing-cap the operator root tilt, preserving heading.

    quat_wxyz: (4,) or (N,4) wxyz.  Returns float32 of the same shape.

    cap_deg <= 0 is the OFF switch and is bit-identical to the input.  Frames
    already at or under the cap come back bit-identical too, so a clean session
    passes through with zero disturbance; only over-cap frames are rewritten.
    Non-finite frames are left alone rather than turned into something worse.

    ``strength`` in [0, 1] scales how far each over-cap frame is pulled toward
    the cap, and exists for the ramp in ``guard_window_pose``: 1.0 is the
    memoryless clamp (every over-cap frame lands exactly on cap_deg), 0.0
    leaves every frame at its own tilt.  It is continuous in between, so a
    frame's rewritten tilt moves smoothly as strength grows -- the ramp cannot
    itself introduce a step.

    With return_mask, also returns the (N,) bool of frames actually rewritten.
    """
    # copy=True is load-bearing: asarray would alias a float64 input, and the
    # clamped rows below are written in place.
    q = np.array(quat_wxyz, np.float64, copy=True)
    single = q.ndim == 1
    q = q.reshape(1, 4) if single else q.reshape(-1, 4)
    hit = np.zeros(q.shape[0], bool)

    if (np.isfinite(cap_deg) and float(cap_deg) > 0.0 and q.shape[0]
            and float(strength) > 0.0):
        tmax = np.radians(min(float(cap_deg), 90.0))
        with np.errstate(invalid="ignore", divide="ignore"):
            finite = np.isfinite(q).all(axis=1)
            if finite.any():
                fi = np.flatnonzero(finite)
                rot = Rot.from_quat(q[fi][:, [1, 2, 3, 0]])
                v = rot.apply(UP)
                th = np.arccos(np.clip(v[:, 2], -1.0, 1.0))
                # Each frame's target swing: cap_deg at strength 1, its own
                # tilt at strength 0.  Written as a lerp so strength >= 1 is
                # bit-identical to the memoryless clamp (tgt == tmax).
                tgt = tmax + (th - tmax) * (1.0 - float(strength))
                over = th > tgt + 1e-12
                if over.any():
                    idx = fi[over]
                    ax = np.cross(UP, v[over])
                    n = np.linalg.norm(ax, axis=1, keepdims=True)
                    # u == -UP (fully inverted): swing axis is degenerate; any
                    # horizontal axis is equivalent there, so take +X.
                    degen = n[:, 0] < 1e-9
                    ax = np.where(degen[:, None], np.array([1.0, 0.0, 0.0]), ax)
                    n = np.where(degen[:, None], 1.0, n)
                    ax = ax / n
                    # R = swing * twist,  R_capped = swing(tgt) * twist.
                    # Writing it this way keeps the twist (heading) exact
                    # rather than re-deriving it from Euler angles.
                    twist = Rot.from_rotvec(th[over, None] * ax).inv() * rot[over]
                    fixed = (Rot.from_rotvec(tgt[over, None] * ax)
                             * twist).as_quat()
                    q[idx] = fixed[:, [3, 0, 1, 2]]        # xyzw -> wxyz
                    hit[idx] = True

    out = q.astype(np.float32)
    if single:
        out, hit = out[0], hit[:1]
    return (out, hit) if return_mask else out


# SMPL-24 (SMPL_PARENTS, pico_tape_to_smpl_obs.py:65).  The torso subtree is
# rooted at spine1 (3) -- every joint except the pelvis and the two legs.  A
# correction applied to this subtree about the pelvis leaves the stance alone.
SPINE_CHAIN = (3, 6, 9, 12, 15)                     # spine1/2/3, neck, head
TORSO_SUBTREE = (3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23)

# The correction is a min-max over the chain, so it takes MANY small steps, not
# one big one -- see the step-size note on cap_torso_lean.  eta_k is
# _TORSO_STEP / (1 + k/_TORSO_DECAY): a subgradient schedule, so the steps
# shrink and the iteration settles ON the min-max instead of beside it.
_TORSO_STEP = 0.5
_TORSO_DECAY = 8.0
_TORSO_ITERS = 128
# Stop when the worst joint is within this many radians of the cap.  The exit
# test is on the max, so every joint is then within tol of it.  Without a
# tolerance the test is `tw > tau` exactly, which in float64 is never quite
# false -- the loop ran its full 128 iterations on EVERY capped frame, and the
# bound rather than the algorithm was doing the work.  The deficit falls like
# k**-4 (the damping product telescopes), so the last ~65 iterations of the 128
# were chasing 1e-16.  1e-5 rad = 5.7e-04 deg is four orders of magnitude under
# the 5 deg of margin the cap keeps under the 30 deg line, and a 0.15 m bone
# rotated by that much moves 1.5e-06 m -- below the float32 resolution of the
# coordinates being written.  Measured: 63 iterations instead of 128 for a
# 15 deg deficit, synthetic worst 25.0006 deg against a 25.0 cap.
_TORSO_TOL = 1e-5


def _chain_dirs(R: Rot, d: np.ndarray) -> np.ndarray:
    """World directions of the (T,K,3) root-local offsets under (T,) roots."""
    T, K = d.shape[0], d.shape[1]
    Rf = Rot.from_quat(np.repeat(R.as_quat(), K, axis=0))
    u = Rf.apply(d.reshape(-1, 3)).reshape(T, K, 3)
    return u / np.maximum(np.linalg.norm(u, axis=2, keepdims=True), 1e-12)


def cap_torso_lean(joints: np.ndarray, quats: np.ndarray, cap_deg: float,
                   return_mask: bool = False, strength: float = 1.0):
    """Bound the WORLD torso lean, rotating the torso subtree rigidly.

    WHY THIS EXISTS
    ---------------
    The root clamp removes the part of the operator's lean that lives in the
    root channel.  Measured on the bundled tapes, that is NOT all of it: with
    the root pinned perfectly upright the pelvis->head direction still reaches
    28.9 deg on seg01 -- the operator's own spine bend -- against a 30 deg
    fatal line, i.e. no margin at all.  At the shipped 25 deg default the
    residual reaches 44.7 deg on 37.9% of clamped frames (65.8 deg on seg02).
    That is why seg01 still falls at 25 deg, and again at 10 deg (13.0 s ->
    26.0 s -> 87.0 s: monotone, never clean).  A root-only clamp cannot fix a
    fold that is not in the root.

    WHAT IT DOES
    ------------
    ``smpl_joints_local`` is root-local, so the world direction of joint j is
    ``R_root @ (loc_j - loc_0)``.  Rotating the whole torso subtree by ONE
    rigid dR leaves every internal angle of the skeleton exactly where it was
    and changes only the torso's orientation -- the same operation
    ``cap_body_root_tilt`` performs on a whole tape, applied to a subtree:

        p_i' = loc_0 + dR_local @ (p_i - loc_0)
        dR_local = R_root^-1 @ dR_world @ R_root        (conjugated into local)

    The pelvis (0) and both legs are not in ``TORSO_SUBTREE``, so they do not
    move: the pose the policy is asked to hold in the stance is unchanged, only
    how far the operator is folded over.

    A rigid rotation moves the whole chain, so a single-vector clamp does not
    bound the rest: the correction is chosen from the worst spine joint and
    re-applied until every joint in ``SPINE_CHAIN`` is inside the cap.

    WHY THE STEPS SHRINK, AND WHY A FULL STEP DOES NOT WORK
    -------------------------------------------------------
    This is a min-max: rotate the chain so that the WORST of its joints is at
    tau.  The obvious loop takes a full step every pass -- rotate by exactly
    ``worst - tau``, which lands that joint precisely on the cap.  It does not
    converge.  When two joints are active in nearly opposite azimuths, fixing
    one lifts the other by the same amount and the max ping-pongs: seg02 frame
    18600 holds spine joints 0 and 4 at 30.3 / 25.0 against [11.8, 51.8, 63.3,
    62.1, 65.8], and the full step stalls at 30.51 deg, flipping which joint it
    corrects from pass to pass.  That is a discontinuity in the commanded pose
    at exactly the moment the guard is meant to be smooth, and it leaves 0.062%
    of seg02's frames over the 30 deg line.

    A constant damped step cures the flip but not the bias: the iteration
    settles where the two active joints' deficits balance, which is NOT the
    optimum.  On a synthetic two-sided fold (spine tilts [30,20,0,-10,-25],
    true min-max 27.50 deg) a constant 0.5 settles at 28.33, and on seg02 at
    28.68 against a floor of 27.81.

    Shrinking steps are what a subgradient method needs.  eta_k = 0.5/(1+k/8)
    reaches the min-max: the synthetic fold lands at 27.537 (0.04 deg off the
    exact optimum), seg01 lands exactly on tau, and seg02's worst is 27.81 --
    under the 30 deg line with 2.2 deg to spare.

    The loop stops on _TORSO_TOL rather than on ``tw > tau``: the schedule's
    product telescopes to k**-4, so an exact test is never quite satisfied in
    float64 and every capped frame used to burn all 128 iterations.  The
    tolerance costs 5.7e-04 deg of accuracy and halves the work -- see the
    constant.

    tau itself is unreachable when the operator's own spine spans more than
    2*tau: a rigid rotation moves the whole chain and cannot narrow it.  58
    frames of seg02 (0.300%) are like that, and seg01 has none.  Those frames
    come back at the min-max of the reachable cone -- the best any rigid
    rotation can do -- and ``_at_floor`` counts them, so a guard running out of
    authority is visible in the log rather than silent.

    joints: (T,24,3) root-local.  quats: (T,4) wxyz, ALREADY root-clamped --
    the residual has to be measured against the root the policy actually gets.

    ``strength`` in [0, 1] is the same ramp as ``cap_root_tilt``: 1.0 converges
    the worst joint onto cap_deg, 0.0 leaves the chain alone, and 0 < s < 1
    leaves the worst joint at ``cap + (worst - cap) * (1 - s)``.  The target is
    computed once from the INPUT chain, not re-read from ``u`` as the loop
    rotates it, so the iteration still converges to a fixed point.
    """
    out = np.array(joints, np.float64, copy=True)
    q = np.asarray(quats, np.float64).reshape(-1, 4)
    T = out.shape[0]
    hit = np.zeros(T, bool)
    if (not (np.isfinite(cap_deg) and float(cap_deg) > 0.0) or T == 0
            or float(strength) <= 0.0):
        return (out.astype(np.float32), hit) if return_mask else out.astype(np.float32)

    tau = np.radians(min(float(cap_deg), 90.0))
    R = Rot.from_quat(q[:, [1, 2, 3, 0]])
    p0 = out[:, :1, :]                                        # (T,1,3) pelvis
    d = out[:, list(SPINE_CHAIN), :] - p0                     # (T,K,3)
    u = _chain_dirs(R, d)                                     # world unit dirs
    tot_R = np.tile(np.eye(3), (T, 1, 1))                     # accumulated dR_world

    strength = float(strength)
    if strength < 1.0:
        tw0 = np.arccos(np.clip(u @ UP, -1.0, 1.0)).max(axis=1)    # (T,) input
        tau_f = tau + (tw0 - tau) * (1.0 - strength)              # (T,) targets
    else:
        tau_f = np.full(T, tau)

    for k in range(_TORSO_ITERS):
        th = np.arccos(np.clip(u @ UP, -1.0, 1.0))            # (T,K)
        tw = th.max(axis=1)
        sel = np.flatnonzero(tw > tau_f + _TORSO_TOL)
        if not len(sel):
            break
        hit[sel] = True
        uu = u[sel]                                           # (M,K,3)
        uw = uu[np.arange(len(sel)), th[sel].argmax(axis=1)]  # worst joint, (M,3)
        # uw x UP, not UP x uw: rotating v about (UP x v) by +a takes its tilt
        # to tilt(v) + a -- the wrong way. About (v x UP) it takes it to
        # tilt(v) - a, which is the correction we want.  Written out because
        # cross(., [0,0,1]) is just [y, -x, 0] and np.cross costs 17 us on
        # arrays this small -- measured, not assumed.
        ax = np.stack([uw[:, 1], -uw[:, 0], np.zeros(len(sel))], axis=1)
        n = np.linalg.norm(ax, axis=1, keepdims=True)
        degen = n[:, 0] < 1e-9                                # chain fully inverted
        ax = np.where(degen[:, None], np.array([1.0, 0.0, 0.0]), ax)
        n = np.where(degen[:, None], 1.0, n)
        ax = ax / n
        # A shrinking fraction of the deficit: see the step-size note in the
        # docstring.  Full steps ping-pong; constant steps settle off-centre.
        # A full-step warmup was tried and rejected: four opening full steps
        # take seg01's worst frame to 2 iterations instead of 128, but the
        # continuity test caught a 0.077 deg jump for a 1e-4 deg input change
        # -- the selection flip back, at exactly the moment the guard is meant
        # to be smooth.  Speed is not worth a discontinuity here.
        eta = _TORSO_STEP / (1.0 + k / _TORSO_DECAY)
        dR = Rot.from_rotvec((eta * (tw[sel] - tau_f[sel]))[:, None] * ax)  # (M,)
        # The same rotation goes to all K joints of a frame, so apply it as a
        # (3,3) matmul rather than building a (M*K,) scipy stack: exact to
        # 4.4e-16, and repeat+from_quat+apply costs 20 us against matmul's 8.
        dRm = dR.as_matrix()                                  # (M,3,3)
        u[sel] = (dRm[:, None] @ uu[..., None])[..., 0]
        # Accumulate in matrix form too: dRm @ tot_R[sel] is 4 us against the
        # 19 of from_quat + multiply + as_quat, and it keeps the whole loop in
        # one representation.
        tot_R[sel] = dRm @ tot_R[sel]

    sel = np.flatnonzero(hit)
    if len(sel):
        # dR is built in the world frame; conjure it into the root-local frame
        # the joints live in:  R @ (dR_l @ v) == dR_w @ (R @ v).
        Rwm = Rot.from_quat(q[sel][:, [1, 2, 3, 0]]).as_matrix()      # (M,3,3)
        dRl = np.transpose(Rwm, (0, 2, 1)) @ tot_R[sel] @ Rwm
        sub = out[sel][:, list(TORSO_SUBTREE), :] - p0[sel]   # (M,S,3)
        out[np.ix_(sel, list(TORSO_SUBTREE))] = (
            p0[sel] + (dRl[:, None] @ sub[..., None])[..., 0])
    return (out.astype(np.float32), hit) if return_mask else out.astype(np.float32)


# --------------------------------------------------------------------------
# Live guard state.  One process runs one source, so the cap is a module
# global rather than a constructor argument: LiveSmplSource is built in three
# places (live_pico_smpl_teleop.py, pico_intent_sender.py,
# pico_token_sender.py) and threading a kwarg through all of them makes the
# one thing that must never be forgotten into the one thing that can be.
# Entry points override the env default with set_root_tilt_cap() right after
# parsing args, and print the value in force.
# --------------------------------------------------------------------------
_CAP_DEG: float = ROOT_TILT_CAP_DEG
_seen = 0
_capped = 0
_torso = 0
_floor = 0
_reported = 0
_REPORT_EVERY = 5000


def set_root_tilt_cap(deg: float) -> float:
    """Set the live cap in degrees (<= 0 disables).

    Returns the NEW value in force, not the previous one -- the entry points
    print the return to report the cap they are running with.  Read the old
    value from ``clamp_stats()`` if you need to restore it.
    """
    global _CAP_DEG
    _CAP_DEG = float(deg)
    return _CAP_DEG


def clamp_stats() -> tuple[int, int, int, int, float]:
    """(frames seen, root-rewritten, torso-rewritten, at-floor, cap in force).

    The torso count includes the floor frames; those are the subset the clamp
    could not finish, so they are the ones to look at when tuning.
    """
    return _seen, _capped, _torso, _floor, _CAP_DEG


def _report() -> None:
    """One periodic line, from whichever counter moved."""
    global _reported
    if _seen - _reported >= _REPORT_EVERY:
        _reported = _seen
        line = (f"[tiltcap] > {_CAP_DEG:.0f} deg: root capped on {_capped}/{_seen} "
                f"frames ({100.0 * _capped / _seen:.2f}%), torso on {_torso} "
                f"({100.0 * _torso / _seen:.2f}%)")
        if _RAMP_T > 0.0:
            line += f", ramp {_RAMP_T:.1f}s (now {ramp_strength():.2f})"
        if _floor:
            line += f", {_floor} at the geometric floor"
        print(line, flush=True)


# Ramp state.  One process runs one source, like _CAP_DEG.
_RAMP_T: float = ROOT_TILT_RAMP_S
_ramp_run = 0.0                    # seconds of continuous over-cap lean so far
_ramp_last: "float | None" = None  # stamp of the previous advance


def set_root_tilt_ramp(sec: float) -> float:
    """Set the ramp rise time in seconds; <= 0 restores the memoryless clamp.

    Returns the NEW value, matching set_root_tilt_cap.  The counters reset so a
    fresh run does not start part-way up the ramp.
    """
    global _RAMP_T, _ramp_run, _ramp_last
    _RAMP_T = max(0.0, float(sec))
    _ramp_run, _ramp_last = 0.0, None
    return _RAMP_T


def ramp_strength() -> float:
    """Current ramp strength in [0, 1], for tests and the report line."""
    return 1.0 if _RAMP_T <= 0.0 else min(1.0, _ramp_run / _RAMP_T)


def ramp_seconds() -> float:
    """The rise time in force; read it to restore the previous value."""
    return _RAMP_T


def _over_cap(joints, quats) -> bool:
    """Is any frame of this window over the cap, root or torso?

    Measured on the RAW window.  The guard's job is to judge the operator's
    own pose, not the pose it is about to hand out.
    """
    q = np.asarray(quats, np.float64).reshape(-1, 4)
    if not len(q):
        return False
    R = Rot.from_quat(q[:, [1, 2, 3, 0]])
    th = np.arccos(np.clip(R.apply(UP)[:, 2], -1.0, 1.0))
    if (np.degrees(th) > _CAP_DEG).any():
        return True
    j = np.asarray(joints, np.float64)
    if j.ndim != 3 or j.shape[0] != len(q):
        return False
    d = j[:, list(SPINE_CHAIN), :] - j[:, :1, :]
    tw = np.arccos(np.clip(_chain_dirs(R, d) @ UP, -1.0, 1.0)).max(axis=1)
    return bool((np.degrees(tw) > _CAP_DEG).any())


def _ramp_advance(now, joints, quats) -> float:
    """Advance the ramp once per ``window()`` call; return the strength.

    ONCE PER CALL, not once per frame.  ``window()`` returns 10 frames that
    overlap the previous call's by 9, so advancing per frame would count the
    same instant about ten times over.  ``now`` is the caller's stamp, which
    ticks at CONTROL_DT -- the rate the window itself slides at.

    Over-cap grows the accumulator up to _RAMP_T; under-cap decays it
    _RAMP_DECAY times faster, so one stray frame does not carry strength into
    the next event.  ``now is None`` means no clock (offline per-frame use and
    the isolation tests) and returns 1.0 -- the memoryless clamp exactly.

    The per-call dt is capped at _RAMP_T: a stall cannot charge the ramp more
    than one rise time however long it lasted, so a scheduler hiccup degrades
    to "one full charge", never to "charged by an unmeasured interval".  A
    window whose ``now`` is older than the last call contributes nothing
    rather than rewinding.
    """
    global _ramp_run, _ramp_last
    if now is None or _RAMP_T <= 0.0:
        return 1.0
    dt = 0.0 if _ramp_last is None else min(max(now - _ramp_last, 0.0), 0.5)
    _ramp_last = now
    if _over_cap(joints, quats):
        _ramp_run = min(_RAMP_T, _ramp_run + dt)
    else:
        _ramp_run = max(0.0, _ramp_run - dt * _RAMP_DECAY)
    return _ramp_run / _RAMP_T


def guard_window_quats(quats: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Cap the root quats of one SMPL window (the 6-D obs block).

    Root-only.  Production goes through ``guard_window_pose`` below, which
    calls this and then caps the torso; this one stays for callers that have
    no joints at hand, and for the isolation tests.  Note that capping here
    alone does NOT bound the operator's lean -- see cap_torso_lean.

    Bit-exact pass-through when no frame is over the cap, so a clean session
    sees zero disturbance.  Counters are process-global and reported every
    _REPORT_EVERY frames rather than per frame.
    """
    global _seen, _capped
    if not (_CAP_DEG > 0.0) or len(quats) == 0:
        return quats
    out, hit = cap_root_tilt(quats, _CAP_DEG, return_mask=True,
                             strength=strength)
    _seen += len(out)
    _capped += int(hit.sum())
    _report()
    return out


def guard_window_pose(joints: np.ndarray, quats: np.ndarray, now=None):
    """THE LIVE INSERTION POINT: root clamp, then torso clamp, both ramped.

    ``window()`` in live_pico_smpl_teleop.py returns this pair to both
    consumers -- pico_intent_sender.py's wire quat and build_smpl_obs' 6-D
    root block plus 72-value joint block -- so guarding both channels here
    covers live teleop, --tape-replay and pico_token_sender.py at once, and
    the counter line in the log is the evidence the guard ran.

    Both channels are needed.  The root clamp bounds the root channel only;
    the part of the operator's fold that lives in the skeleton survives it and
    reaches 44.7 deg on seg01 at the 25 deg default (28.9 deg even with the
    root pinned upright), so the root alone leaves an armed failure.  Order
    matters: the torso residual is measured against the root the policy will
    actually receive, so the root is clamped first.

    WHY THE CLAMP IS RAMPED (2026-09-30)
    ------------------------------------
    A memoryless clamp bounds every FRAME's tilt, which is not the same thing
    as bounding the ROBOT, and the difference is a fall the guard caused
    itself.  On seg02 at tape_t 78.00..78.30 there is a 0.30 s blip peaking at
    31.49 deg (torso) / 34.25 deg (root).  Raw, it is harmless -- nothing
    under ~0.4 s over the 30 deg line has ever dropped this robot.  The guard
    engaged on all 26 frames of it and rotated the whole upper body 6.49 deg
    about ROLL, the axis a biped has the least margin on, and the robot was
    down 0.27 s later.

    Strict A/B, same policy_t=6.00 trigger, offsets 8.08 vs 7.92:

        guard OFF -> fell at tape_t 219.96 and 220.45 (two runs, 0.49 s apart)
        guard ON  -> fell at tape_t  78.52

    No cap value fixes this, and the reason is exact rather than empirical:
    missing the blip needs a cap above 34.25 deg, bounding the robot needs one
    below 30, and the interval is empty.  What separates a 0.30 s blip from
    the 2.3 s / 52.9 deg event that is the real target is DURATION -- which is
    what the field data said in the first place.  Hence the ramp: strength
    grows with time over the cap, so a blip is clamped only in part while a
    sustained fold reaches full strength within _RAMP_T.

    Passing ``now`` engages the ramp; ``now=None`` is the old behaviour.
    Off (cap <= 0) returns both inputs untouched, by identity.
    """
    global _seen, _torso, _floor
    if not (_CAP_DEG > 0.0) or len(quats) == 0:
        return joints, quats
    s = _ramp_advance(now, joints, quats)
    if s <= 0.0:
        # Start of a fold -- over the cap, but not yet for long enough to pull
        # anything toward it.  Return BY IDENTITY, not through the clamps: a
        # strength-0 clamp still copies, and the pass-through contract is what
        # lets a clean session see zero disturbance.  Count the frames anyway
        # so the report line's denominator stays the number handed out.
        _seen += len(quats)
        _report()
        return joints, quats
    q2 = guard_window_quats(quats, strength=s)
    j2, hit = cap_torso_lean(joints, q2, _CAP_DEG, return_mask=True, strength=s)
    _torso += int(hit.sum())
    _floor += _at_floor(j2, q2, s)
    _report()
    return j2, q2


def _at_floor(joints: np.ndarray, quats: np.ndarray, strength: float = 1.0) -> int:
    """Frames still over the cap after the torso clamp, i.e. at the floor.

    A rigid rotation cannot fit a chain whose own joints span more than
    2*cap, so on the frames where the operator folds that hard cap_torso_lean
    returns the min-max of the reachable cone instead.  Measured worst on the
    bundled tapes is 27.97 deg -- under the 30 deg line, but it IS the guard
    running out of authority, so it gets counted and printed rather than
    passing silently.

    Only counts at FULL strength.  Under the ramp a sub-unit strength leaves
    its frames over the cap on purpose, so counting those would report the
    chosen trade as a loss of authority and bury the frames that are one.
    """
    if len(joints) == 0 or float(strength) < 1.0 - 1e-9:
        return 0
    q = np.asarray(quats, np.float64).reshape(-1, 4)
    j = np.asarray(joints, np.float64)
    u = _chain_dirs(Rot.from_quat(q[:, [1, 2, 3, 0]]),
                    j[:, list(SPINE_CHAIN), :] - j[:, :1, :])
    th = np.arccos(np.clip(u @ UP, -1.0, 1.0)).max(axis=1)
    return int((th > np.radians(min(_CAP_DEG, 90.0)) + 1e-3).sum())
