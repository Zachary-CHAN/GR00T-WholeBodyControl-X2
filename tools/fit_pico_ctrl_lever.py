#!/usr/bin/env python3
"""Fit the Pico controller-origin -> operator-wrist lever from a v3 record.

The Pico controller pose is the controller's tracking origin ``C``, not the
operator's wrist centre ``W``. They are related by a fixed lever ``r`` in the
controller body frame::

    C_i = W + R_i @ r

so with a STATIONARY wrist, a wrist-only rotation sweeps ``C`` on the lever
arc and the v3 position target chases it -- the "shoulder/elbow moves while
only my wrist turns" symptom. ``arm_teleop_v3_mujoco.py --ctrl-lever-left X Y Z``
(or ``--ctrl-lever-right``) removes exactly that term (``W = C - R @ r``).

This tool estimates ``r`` from the per-tick controller telemetry the v3
harness records (``--record``):

    column 21:24 / 24:28   left  filtered controller pos / quat (w,x,y,z)
    column 28:31 / 31:35   right filtered controller pos / quat

The harness writes ``np.savez_compressed(..., rows=arr, joints=...)``; this
tool reads the ``rows`` member (a bare ``.npy`` matrix is accepted too).

It slides a window over the record, solves the 6-unknown linear system
``[I | R_i] [W; r] = C_i`` per window, and reports the lowest-residual window.
Need enough rotation inside the window (``--min-sweep-deg``) or the fit is
under-determined; need a still wrist or the translation confounds it. A low
residual alone is NOT proof: a slowly translating wrist still fits well with
a wrong ``r``. The report refits at +/-20% window widths around the same time
center; use that spread (and the physical size of ``|r|``) as the honesty
check.

    .venv/bin/python tools/fit_pico_ctrl_lever.py --record /tmp/v3_live.npz --side left

Then re-run the same wrist-only sweep with the printed per-hand
``--ctrl-lever-left`` / ``--ctrl-lever-right`` and confirm the robot's
shoulder/elbow stop moving before trusting it live.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation as sRot

# v3 harness --record layout (85 fixed columns; see the header there)
LEFT_POS = 21
LEFT_QUAT = 24
RIGHT_POS = 28
RIGHT_QUAT = 31
N_COLS = 35  # last column used here is RIGHT_QUAT + 3 = 34


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


def _rotation_mats(quats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row rotation matrices, plus the mask of usable quaternions.

    The first frames of a live record are zero placeholders until the
    sender's first intent arrives. Build rotations only for the valid rows
    (scipy rejects a zero-norm quaternion in the batch) and leave identity
    in the placeholders: every window containing one is skipped.
    """
    n = len(quats)
    valid = np.linalg.norm(quats, axis=1) > 1e-6
    mats = np.tile(np.eye(3), (n, 1, 1))
    if valid.any():
        mats[valid] = sRot.from_quat(_quat_wxyz_to_xyzw(quats[valid])).as_matrix()
    return mats, valid


def _fit_window(
    positions: np.ndarray,
    mats: np.ndarray,
    start: int,
    window: int,
    min_sweep_deg: float,
) -> LeverFit | None:
    """One window; the caller guarantees the slice is valid and in bounds."""
    R = mats[start : start + window]
    # orientation sweep inside the window, measured as the max deviation
    # from the first frame (not just the endpoint)
    devs = np.array([sRot.from_matrix(R[0].T @ Ri).magnitude() for Ri in R])
    sweep = float(devs.max())
    if sweep < np.radians(min_sweep_deg):
        return None
    A = np.zeros((3 * window, 6), np.float64)
    A[:, :3] = np.tile(np.eye(3), (window, 1))
    A[:, 3:] = R.reshape(-1, 3)  # rows: R_0; R_1; ... (3x3 each)
    b = positions[start : start + window].reshape(-1)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    rms = float(np.linalg.norm(A @ x - b) / np.sqrt(len(b)))
    return LeverFit(
        lever=x[3:].copy(),
        wrist=x[:3].copy(),
        rms_m=rms,
        sweep_rad=sweep,
        start=start,
    )


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
    mats, valid = _rotation_mats(quats)
    if not valid.any():
        return None

    best: LeverFit | None = None
    for start in range(0, n - window + 1, step):
        if not valid[start : start + window].all():
            continue
        f = _fit_window(positions, mats, start, window, min_sweep_deg)
        if f is not None and (best is None or f.rms_m < best.rms_m):
            best = f
    return best


def load_rows(path: str | Path) -> np.ndarray:
    """Load the flat per-tick matrix from a v3 harness ``--record`` file.

    The harness writes ``np.savez_compressed(..., rows=arr, joints=...)``, so
    ``np.load`` returns an ``NpzFile``, not an array. A bare ``.npy`` matrix
    is accepted too.
    """
    data = np.load(path, allow_pickle=False)
    if isinstance(data, np.ndarray):
        return data
    if "rows" not in data.files:
        raise SystemExit(
            f"{path}: no 'rows' array (keys: {data.files}); expected a v3 "
            "harness --record .npz or a .npy matrix"
        )
    return data["rows"]


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
    # The harness record is an npz with a 'rows' member; indexing the NpzFile
    # directly (the original bug) raises KeyError. Round-trip one here.
    rows = np.zeros((len(pos), 35), np.float64)
    rows[:, LEFT_POS : LEFT_POS + 3] = np.asarray(pos)
    rows[:, LEFT_QUAT : LEFT_QUAT + 4] = np.asarray(quats)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "record.npz"
        np.savez_compressed(path, rows=rows, joints=np.array(["x"]))
        loaded = load_rows(path)
    assert loaded.shape == rows.shape and np.array_equal(loaded, rows), "record loader round-trip"
    print("[self-test] record loader: rows member round-trip OK")
    # A live record starts with zero placeholders until the sender's first
    # intent frame. scipy rejects a zero-norm quaternion in the batch, so the
    # fit must build rotations from the valid rows only (the second bug).
    mixed = fit_lever(
        np.vstack([np.zeros((30, 3)), np.asarray(pos)]),
        np.vstack([np.zeros((30, 4)), np.asarray(quats)]),
    )
    assert mixed is not None, "mixed zero/valid record did not qualify"
    err_mixed = float(np.linalg.norm(mixed.lever - r_true))
    assert err_mixed < 0.005, f"mixed-record lever error {err_mixed * 1e3:.2f} mm (> 5 mm)"
    print("[self-test] mixed zero/valid frames: fit skips placeholders OK")


def _window_stability(
    positions: np.ndarray,
    quats_wxyz: np.ndarray,
    best: LeverFit,
    window: int,
    min_sweep_deg: float,
) -> list[tuple[int, LeverFit]]:
    """Refit at -20% / requested / +20% widths around the SAME segment.

    A rigid lever on a still wrist gives the same ``r`` at every width; a
    translating wrist lets each width absorb a different amount of the
    translation, which is the failure mode a low RMS alone does not catch.

    Keep the window centered on ``best.start``: letting each width search its
    own best start compares different time segments (and different hand
    positions), which reports false instability. Same-segment fits isolate
    the width effect.
    """
    positions = np.asarray(positions, np.float64)
    quats = np.asarray(quats_wxyz, np.float64)
    mats, valid = _rotation_mats(quats)
    n = len(positions)
    center = best.start + 0.5 * window
    widths = sorted({max(10, int(round(window * s))) for s in (0.8, 1.0, 1.2)})
    out: list[tuple[int, LeverFit]] = []
    for w in widths:
        start = int(round(center - 0.5 * w))
        start = max(0, min(start, n - w))
        if start < 0 or not valid[start : start + w].all():
            continue
        f = _fit_window(positions, mats, start, w, min_sweep_deg)
        if f is not None:
            out.append((w, f))
    return out


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

    rec = load_rows(args.record)
    if rec.ndim != 2 or rec.shape[1] < N_COLS:
        raise SystemExit(f"{args.record}: expected (N, >= {N_COLS}) rows; got {rec.shape}")
    pos_col = LEFT_POS if args.side == "left" else RIGHT_POS
    quat_col = LEFT_QUAT if args.side == "left" else RIGHT_QUAT
    if not np.any(np.linalg.norm(rec[:, quat_col : quat_col + 4], axis=1) > 1e-6):
        raise SystemExit(
            f"{args.record}: {args.side} controller quaternions are all zero. "
            "A harness-only --record stores no intent frames; start "
            "pico_intent_sender.py (headset connected) during the recording."
        )
    positions = rec[:, pos_col : pos_col + 3]
    quats = rec[:, quat_col : quat_col + 4]
    fit = fit_lever(
        positions,
        quats,
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
    alts = _window_stability(positions, quats, fit, args.window, args.min_sweep_deg)
    if len(alts) >= 2:
        spread = max(float(np.linalg.norm(f.lever - fit.lever)) for _, f in alts)
        print(f"window-size stability (same segment; spread vs best {spread * 1e3:.1f} mm):")
        for w, f in alts:
            d = float(np.linalg.norm(f.lever - fit.lever))
            print(
                f"  w={w:3d} start {f.start:4d}  lever {np.round(f.lever, 4)}  "
                f"(d {d * 1e3:.1f} mm, rms {f.rms_m * 1e3:.2f} mm)"
            )
        # 20 mm absolute or 20% of |r|, whichever is larger: a still wrist at
        # ~90 mm lever reproduces r within a few mm, while a translating one
        # shifts it by tens of mm as the window width changes.
        limit = max(0.02, 0.2 * float(np.linalg.norm(fit.lever)))
        if spread > limit:
            print(
                f"WARNING: lever spread > {limit * 1e3:.0f} mm across window widths -- "
                "the wrist was probably moving, and the low RMS is absorbing that "
                "translation. Do not trust this vector; re-record with the wrist "
                "truly still."
            )
    if np.linalg.norm(fit.lever) > 0.15:
        print(
            "WARNING: |r| > 150 mm is larger than a typical controller-to-wrist "
            "lever; treat it as translation contamination, not a handle offset."
        )
    print()
    print("re-run with:")
    print(f"  --ctrl-lever-{args.side} " + " ".join(f"{v:.4f}" for v in fit.lever))
    print("(fit the other hand separately and pass both flags: the vectors differ)")
    if fit.rms_m > 0.015:
        print(
            "\nWARNING: rms > 15 mm -- the hand was probably translating, not just "
            "rotating. The lever is not trustworthy; re-record a still-wrist sweep."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
