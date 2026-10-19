#!/usr/bin/env python3
"""Fit the Pico controller-origin -> operator-wrist lever from a v3 record.

The Pico controller pose is the controller's tracking origin ``C``, not the
operator's wrist centre ``W``. They are related by a fixed lever ``r`` in the
controller body frame::

    C_i = W + R_i @ r

so with a STATIONARY wrist, a wrist-only rotation sweeps ``C`` on the lever
arc and the v3 position target chases it -- the "shoulder/elbow moves while
only my wrist turns" symptom. ``arm_teleop_v3_mujoco.py --ctrl-lever X Y Z``
removes exactly that term (``W = C - R @ r``).

This tool estimates ``r`` from the per-tick controller telemetry the v3
harness records (``--record``):

    column 21:24 / 24:28   left  filtered controller pos / quat (w,x,y,z)
    column 28:31 / 31:35   right filtered controller pos / quat

It slides a window over the record, solves the 6-unknown linear system
``[I | R_i] [W; r] = C_i`` per window, and reports the lowest-residual window.
Need enough rotation inside the window (``--min-sweep-deg``) or the fit is
under-determined; need a still wrist (low residual) or the translation
confounds it. Use the residual as the honesty check, not the lever alone.

    .venv/bin/python tools/fit_pico_ctrl_lever.py --record /tmp/v3_live.npz --side left

Then re-run the same wrist-only sweep with the printed ``--ctrl-lever`` and
confirm the robot's shoulder/elbow stop moving before trusting it live.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as sRot

# v3 harness --record layout (85 fixed columns; see the header there)
LEFT_POS = 21
LEFT_QUAT = 24
RIGHT_POS = 28
RIGHT_QUAT = 31


@dataclass
class LeverFit:
    lever: np.ndarray
    wrist: np.ndarray
    rms_m: float
    sweep_rad: float
    start: int


def _quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, np.float64)
    return q[..., [1, 2, 3, 0]]


def fit_lever(
    positions: np.ndarray,
    quats_wxyz: np.ndarray,
    window: int = 50,
    step: int = 5,
    min_sweep_deg: float = 20.0,
) -> LeverFit | None:
    """Best sliding-window fit of ``C_i = W + R_i @ r``.

    Returns ``None`` when no window rotates enough. ``rms_m`` is the
    per-coordinate RMS of the 3-D residuals (metres); pure wrist rotation
    and millimetre tracking noise give ~1e-3, a translating hand much more.
    """
    positions = np.asarray(positions, np.float64)
    quats = np.asarray(quats_wxyz, np.float64)
    n = len(positions)
    if n < window or len(quats) != n:
        return None
    valid = np.linalg.norm(quats, axis=1) > 1e-6
    if not valid.any():
        return None
    rots = sRot.from_quat(_quat_wxyz_to_xyzw(quats))
    mats = rots.as_matrix()
    eye3 = np.eye(3)

    best: LeverFit | None = None
    for start in range(0, n - window + 1, step):
        sl = slice(start, start + window)
        if not valid[sl].all():
            continue
        R = mats[sl]
        # orientation sweep inside the window, measured as the max deviation
        # from the first frame (not just the endpoint)
        devs = np.array([sRot.from_matrix(R[0].T @ Ri).magnitude() for Ri in R])
        sweep = float(devs.max())
        if sweep < np.radians(min_sweep_deg):
            continue
        A = np.zeros((3 * window, 6), np.float64)
        A[:, :3] = np.tile(eye3, (window, 1))
        A[:, 3:] = R.reshape(-1, 3)  # rows: R_0; R_1; ... (3x3 each)
        b = positions[sl].reshape(-1)
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        rms = float(np.linalg.norm(A @ x - b) / np.sqrt(len(b)))
        if best is None or rms < best.rms_m:
            best = LeverFit(
                lever=x[3:].copy(),
                wrist=x[:3].copy(),
                rms_m=rms,
                sweep_rad=sweep,
                start=start,
            )
    return best


def _self_test() -> None:
    """Synthetic stationary wrist: recover a known lever within 5 mm."""
    rng = np.random.RandomState(0)
    r_true = np.array([0.012, -0.021, 0.083])
    # random rotation walk: varied axes are required to identify all three
    # lever components (a single-axis sweep leaves the component along that
    # axis unobservable)
    steps = sRot.from_rotvec(rng.normal(scale=0.04, size=(240, 3)))
    rots = sRot.identity()
    quats, pos = [], []
    for s in steps:
        rots = rots * s
        R = rots.as_matrix()
        quats.append(rots.as_quat(scalar_first=True))
        pos.append(R @ r_true + rng.normal(scale=5e-4, size=3))
    fit = fit_lever(np.asarray(pos), np.asarray(quats))
    assert fit is not None, "synthetic sweep did not qualify"
    err = float(np.linalg.norm(fit.lever - r_true))
    assert err < 0.005, f"lever error {err * 1e3:.2f} mm (> 5 mm)"
    assert fit.rms_m < 2e-3, f"residual {fit.rms_m * 1e3:.2f} mm (> 2 mm)"
    print(
        "[self-test] recovered "
        f"{np.round(fit.lever, 5)} vs true {r_true} (err {err * 1e3:.2f} mm), "
        f"rms {fit.rms_m * 1e3:.2f} mm, sweep {np.degrees(fit.sweep_rad):.0f} deg"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--record", default="", help="v3 harness --record .npz (empty = --self-test)")
    ap.add_argument("--side", choices=("left", "right"), default="left")
    ap.add_argument("--window", type=int, default=50, help="samples per fit window (50 Hz)")
    ap.add_argument("--step", type=int, default=5, help="window stride in samples")
    ap.add_argument(
        "--min-sweep-deg",
        type=float,
        default=20.0,
        help="minimum in-window controller rotation; below this the lever is " "under-determined",
    )
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test or not args.record:
        _self_test()
        return 0

    rec = np.load(args.record)
    pos_col = LEFT_POS if args.side == "left" else RIGHT_POS
    quat_col = LEFT_QUAT if args.side == "left" else RIGHT_QUAT
    fit = fit_lever(
        rec[:, pos_col : pos_col + 3],
        rec[:, quat_col : quat_col + 4],
        window=args.window,
        step=args.step,
        min_sweep_deg=args.min_sweep_deg,
    )
    if fit is None:
        print(
            f"no {args.window}-sample window with >= {args.min_sweep_deg:g} deg of "
            "controller rotation; record a slower/longer wrist-only sweep"
        )
        return 1
    print(f"record {args.record}  side {args.side}")
    print(
        f"best window start {fit.start} ({fit.start / 50.0:.2f} s), "
        f"sweep {np.degrees(fit.sweep_rad):.1f} deg, rms {fit.rms_m * 1e3:.2f} mm"
    )
    print(f"lever  {np.round(fit.lever, 5)} m  (|r| {np.linalg.norm(fit.lever) * 1e3:.1f} mm)")
    print(f"wrist  {np.round(fit.wrist, 5)} m  (rough stationary-wrist position)")
    print()
    print("re-run with:")
    print("  --ctrl-lever " + " ".join(f"{v:.4f}" for v in fit.lever))
    if fit.rms_m > 0.015:
        print(
            "\nWARNING: rms > 15 mm -- the hand was probably translating, not just "
            "rotating. The lever is not trustworthy; re-record a still-wrist sweep."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
