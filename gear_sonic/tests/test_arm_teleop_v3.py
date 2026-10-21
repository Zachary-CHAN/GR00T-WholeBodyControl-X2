"""Headless layer tests for the v3 arm teleop (wrist decoupled from the IK).

Mirrors the `--self-test` checks of
``gear_sonic/scripts/arm_teleop_v3_mujoco.py`` one-to-one.
Run: ``.venv/bin/python -m pytest gear_sonic/tests/test_arm_teleop_v3.py -q``
"""

from __future__ import annotations

from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gear_sonic" / "scripts"))
sys.path.insert(0, str(REPO))

import arm_teleop_v3_mujoco as a3  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy.spatial.transform import Rotation as sRot  # noqa: E402

from gear_sonic.utils.teleop.pico_3pt import (  # noqa: E402
    controller_pose_to_operator_frame,
    fixed_operator_frame_from_controllers,
)


def test_wrist_zero_delta_noop():
    a3.CHAIN_V3_CHECKS[0][1]()


def test_wrist_single_axis_direction():
    a3.CHAIN_V3_CHECKS[1][1]()


def test_wrist_limit_clamp():
    a3.CHAIN_V3_CHECKS[2][1]()


def test_wrist_slew_and_pi_guard():
    a3.CHAIN_V3_CHECKS[3][1]()


def test_ik4_convergence_wrist_frozen():
    a3.CHAIN_V3_CHECKS[4][1]()


def test_ik4_tracks_wrist_coupling():
    a3.CHAIN_V3_CHECKS[5][1]()


def test_clutch_freeze_reanchor_v3():
    a3.CHAIN_V3_CHECKS[6][1]()


def test_ik_freeze_guard_v3():
    a3.CHAIN_V3_CHECKS[7][1]()


def test_torso_keepout_v3():
    a3.CHAIN_V3_CHECKS[8][1]()


def test_waist_fixed_and_smpl_ignored():
    a3.CHAIN_V3_CHECKS[9][1]()


def test_grip_stale_open():
    a3.CHAIN_V3_CHECKS[10][1]()


def test_trigger_hand_grip_clutch():
    a3.CHAIN_V3_CHECKS[12][1]()


def test_default_4dof_and_equal_full_pose_weights():
    a3.CHAIN_V3_CHECKS[13][1]()


def test_ctrl_lever_pure_wrist_rotation():
    a3.CHAIN_V3_CHECKS[14][1]()


def test_position_deadband():
    a3.CHAIN_V3_CHECKS[15][1]()


def test_posture_gain_null_space():
    a3.CHAIN_V3_CHECKS[16][1]()


def test_wrist_nonzero_anchor_tracks_tool_rotation():
    a3.CHAIN_V3_CHECKS[17][1]()


def test_3pt_controller_frame():
    a3.CHAIN_V3_CHECKS[18][1]()


def test_wrist_default_signs_side_parallel():
    a3.CHAIN_V3_CHECKS[19][1]()


def test_fixed_operator_frame_position_only():
    """The operator frame reads positions only: arbitrary (even zero)
    quaternions on the same pair of positions must give the same M/origin."""
    left = np.array([0.30, 0.05, 1.20, 12.0, -4.0, 9.0, 0.0])
    right = np.array([-0.30, -0.02, 1.15, 0.0, 0.0, 0.0, 0.0])
    M1, o1 = fixed_operator_frame_from_controllers(left, right)
    left2, right2 = left.copy(), right.copy()
    left2[3:] = [0.0, 0.0, 0.0, 1.0]
    right2[3:] = [1.0, 2.0, 3.0, 4.0]
    M2, o2 = fixed_operator_frame_from_controllers(left2, right2)
    assert np.allclose(M1, M2)
    assert np.allclose(o1, o2)


def test_fixed_operator_frame_lateral_sign():
    """p_left - p_right along raw +X -> the frame's lateral row is +X, i.e.
    the left-axis component of the same delta is POSITIVE (left is +y)."""
    left = np.array([0.30, 0.0, 1.20])
    right = np.array([-0.30, 0.0, 1.20])
    M, origin = fixed_operator_frame_from_controllers(left, right)
    d = left - right
    assert (M @ d)[1] > 0.5
    assert np.allclose(origin, 0.5 * (left + right))


def test_controller_wire_wrist_delta_frame_free():
    """A constant frame left-multiply cancels in the wrist delta
    R_anchor^-1 R_now, so the existing --ctrl-lever stays valid."""
    M, o = fixed_operator_frame_from_controllers([0.30, 0.0, 1.20], [-0.30, 0.0, 1.20])
    rng = np.random.default_rng(0)
    R0, R1 = sRot.random(random_state=rng), sRot.random(random_state=rng)
    p0 = np.array([0.10, 0.20, 1.00])
    p1 = np.array([0.10, 0.20, 1.00])
    w0 = controller_pose_to_operator_frame(np.concatenate([p0, R0.as_quat()]), M, o)
    w1 = controller_pose_to_operator_frame(np.concatenate([p1, R1.as_quat()]), M, o)
    Rw0 = sRot.from_quat(w0[3:], scalar_first=True)
    Rw1 = sRot.from_quat(w1[3:], scalar_first=True)
    assert np.allclose((Rw0.inv() * Rw1).as_matrix(), (R0.inv() * R1).as_matrix(), atol=1e-6)


def test_controller_wire_position_delta_origin_free():
    """The frame origin cancels in position deltas."""
    M, o = fixed_operator_frame_from_controllers([0.30, 0.0, 1.20], [-0.30, 0.0, 1.20])
    p0 = np.array([0.10, 0.20, 1.00, 0.0, 0.0, 0.0, 1.0])
    p1 = np.array([0.25, 0.10, 0.90, 0.0, 0.0, 0.0, 1.0])
    w0 = controller_pose_to_operator_frame(p0, M, o)
    w1 = controller_pose_to_operator_frame(p1, M, o)
    assert np.allclose(w1[:3] - w0[:3], M @ (p1[:3] - p0[:3]), atol=1e-6)


def test_fixed_operator_frame_rejects_stacked_controllers():
    """Controllers closer than 15 cm horizontally cannot seed a frame."""
    with pytest.raises(ValueError):
        fixed_operator_frame_from_controllers([0.05, 0.0, 1.20], [-0.02, 0.0, 1.21])


def test_wire_roundtrip():
    a3.check_wire_roundtrip()


def test_grip_mapping_v3():
    a3.check_grip_mapping_v3()


def test_keepout_matches_model():
    a3.check_keepout_matches_model()


def test_limits_match_sdk():
    a3.check_limits_match_sdk()


def test_neutral_matches_vr_defaults():
    a3.check_neutral_matches_vr_defaults()


def test_model_and_head_hold():
    a3.check_model_and_head_hold()


def test_self_test_bundle():
    assert a3.run_self_test()
