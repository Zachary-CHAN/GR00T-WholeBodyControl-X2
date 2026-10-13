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


def test_input_lpf():
    iutm.check_input_lpf()


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


def test_model_and_head_hold():
    iutm.check_model_and_head_hold()


def test_self_test_bundle():
    assert iutm.run_self_test()
