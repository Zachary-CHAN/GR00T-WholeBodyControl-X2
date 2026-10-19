"""Headless layer tests for the incremental (clutch-IK) upper teleop.

Mirrors the `--self-test` checks of
``gear_sonic/scripts/incremental_upper_teleop_mujoco.py`` one-to-one.
Run: ``.venv/bin/python -m pytest gear_sonic/tests/test_incremental_upper_teleop.py -q``
"""

from __future__ import annotations

from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gear_sonic" / "scripts"))
sys.path.insert(0, str(REPO))

import incremental_upper_teleop_mujoco as iutm  # noqa: E402


def test_wire_roundtrip():
    iutm.check_wire_roundtrip()


def test_ik_convergence():
    iutm.check_ik_convergence()


def test_clutch_freeze_reanchor():
    iutm.check_clutch_freeze_reanchor()


def test_waist_rest_and_twist():
    iutm.check_waist_rest_and_twist()


def test_waist_chain_gating():
    iutm.check_waist_chain_gating()


def test_grip_mapping():
    iutm.check_grip_mapping()


def test_trigger_hand_grip_clutch():
    iutm.check_trigger_hand_grip_clutch()


def test_input_lpf():
    iutm.check_input_lpf()


def test_torso_frame():
    iutm.check_torso_frame()


def test_overhead_clamp():
    iutm.check_overhead_clamp()


def test_extended_stability():
    iutm.check_extended_stability()


def test_ik_freeze_guard():
    iutm.check_ik_freeze_guard()


def test_ctrl_canonicalization_matches_wire_chain():
    """The sender's controller canonicalisation must be the EXACT rotation
    the wire smpl_joints chain applies: canonicalising the chain's own FK
    output reproduces smpl_joints_local bit-for-bit (the live "hand in
    front, robot hand behind" inversion was a frame bug here)."""
    from live_pico_smpl_teleop import SMPL_PARENTS
    import numpy as np
    from pico_intent_sender import _canonical_root_inv
    from pico_manager_thread_server import compute_from_body_poses
    from scipy.spatial.transform import Rotation as sRot

    rng = np.random.RandomState(3)
    body = np.zeros((24, 7))
    body[:, :3] = rng.randn(24, 3) * 0.3
    for i in range(24):
        body[i, 3:7] = sRot.random(random_state=rng).as_quat()  # xyzw at [3:7]
    r = compute_from_body_poses(SMPL_PARENTS, "cpu", body.astype(np.float32))
    jw = r["joints"][0].numpy().astype(np.float64)  # FK output, raw y-up world
    jl = r["smpl_joints_local"][0].numpy().astype(np.float64)  # wire frame
    err = np.abs(_canonical_root_inv(body).apply(jw) - jl).max()
    assert err < 1e-5, f"canonical frame mismatch: {err * 1000:.3f} mm"


def test_ctrl_torso_frame_from_raw_joints():
    """The sender frames controller poses with M built from the RAW SDK
    joints (up=neck-pelvis, lat=shoulder line, fwd=lat x up). A synthetic
    person in an arbitrary SLAM pose: a controller offset that is +fwd in
    the person's anatomical axes must come out +x regardless of the world
    pose (live round 3: this is what finally killed the jitter + the
    unreachable-target thrash -- the canonical chain cannot do this)."""
    import numpy as np
    from pico_intent_sender import _torso_M
    from scipy.spatial.transform import Rotation as sRot

    body = np.zeros((24, 7))
    body[0, :3] = [1.2, -0.4, 0.7]  # pelvis anywhere in the SLAM world
    body[12, :3] = body[0, :3] + [0.1, 0.05, 0.6]  # neck up-ish
    body[16, :3] = body[0, :3] + [-0.1, 0.2, 0.45]  # L shoulder
    body[17, :3] = body[0, :3] + [-0.1, -0.2, 0.45]  # R shoulder
    M = _torso_M(body, None, 0.2, 0.02)
    # the person's anatomical axes expressed in world coords (rows of M are
    # fwd/lat/up in world; columns map world -> torso)
    fwd_w, up_w = M[0], M[2]
    ctrl = body[0, :3] + 0.3 * fwd_w + 0.15 * up_w  # 0.3 m forward, 0.15 m up
    np.testing.assert_allclose(M @ (ctrl - body[0, :3]), [0.3, 0.0, 0.15], atol=1e-9)
    assert abs(np.linalg.det(M) - 1.0) < 1e-9, "M must be a proper rotation"
    # rotation path: the sender transforms a controller ORIENTATION as a
    # pose (q_torso = R(M) * q_c, not a conjugation) -- a controller whose
    # world orientation equals the torso frame itself ((R_M)^-1) must
    # arrive as the identity
    rM = sRot.from_matrix(M)
    q_ctrl = rM.inv()
    q_torso = rM * q_ctrl
    assert q_torso.magnitude() < 1e-9, "quat pose path broke the torso frame"
    # degenerate joints: keep the previous frame (None stays None -> dead)
    assert _torso_M(np.zeros((24, 7)), None, 0.2, 0.02) is None
    assert _torso_M(np.zeros((24, 7)), M, 0.2, 0.02) is M


def test_model_and_head_hold():
    iutm.check_model_and_head_hold()


def test_self_test_bundle():
    assert iutm.run_self_test()
