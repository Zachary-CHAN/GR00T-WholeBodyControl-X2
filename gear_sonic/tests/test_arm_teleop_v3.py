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


def test_wrist_nonzero_anchor_decoupled():
    a3.CHAIN_V3_CHECKS[17][1]()


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
