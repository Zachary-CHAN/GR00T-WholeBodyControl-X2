"""Headless layer tests for the incremental (clutch-IK) upper teleop.

Mirrors the `--self-test` checks of
``gear_sonic/scripts/incremental_upper_teleop_mujoco.py`` one-to-one.
Run: ``.venv/bin/python -m pytest gear_sonic/tests/test_incremental_upper_teleop.py -q``
"""
from __future__ import annotations

import sys
from pathlib import Path

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


def test_model_and_head_hold():
    iutm.check_model_and_head_hold()


def test_self_test_bundle():
    assert iutm.run_self_test()
