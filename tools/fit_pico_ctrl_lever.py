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
``[I | R_i] [W; r] = C_i`` per window, and reports the lowest-residual
window. Need enough rotation inside the window (``--min-sweep-deg``) or the
fit is under-determined; need a still wrist or the translation confounds it.

A low residual alone is NOT proof, and neither is a large rotation sweep:
the sweep can be about a SINGLE axis, in which case the lever component
along that axis never moves the controller origin and is unobservable --
the fit silently fills it with wrist translation. The window is therefore
also scored by the observability of its rotation cloud (``sigma_min /
sigma_max`` of the stacked ``R_i - mean(R)``); a single-axis sweep scores
near zero and is rejected (``--min-observability``). Windows that do pass
are then combined into one robust shared-``r`` fit (each window keeps its
own constant ``W``).

Even then, a constant-W window assumes the wrist stays put for a whole
second. A live calibration drifts by centimetres per second, and that drift
is absorbed into ``r`` -- which is why a low RMS and stable window widths
can still hand you a wrong vector. The recommended estimate therefore
differences the record over short lags (``--lags``, 40-400 ms):
``dC = dR @ r``, where the drift over one lag is millimetres and averages
out. Estimates at several lags are combined by median; their spread is the
honesty check. Move the controller in a CONE / circle so it rotates about
at least two, ideally three, axes; the report also refits at +/-20% window
widths around the same time center as a second honesty check.

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
    observability: float = 1.0


@dataclass
class LeverConsensus:
    lever: np.ndarray
    n_windows: int
    spread_m: float
    rms_median_m: float
    rms_max_m: float


@dataclass
class LeverIncrement:
    lever: np.ndarray
    lags: tuple[int, ...]
    n_pairs: int
    resid_median_m: float
    spread_m: float
    observability: float


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
    # Observability of the lever itself: with W eliminated, r is identified
    # by the variation of R about its window mean. A rotation cloud that is
    # nearly one-dimensional has a near-zero smallest singular value, and
    # the component of r along that axis is then free to absorb wrist
    # translation. This is independent of the residual, so a low RMS on a
    # single-axis sweep is NOT evidence that r is right.
    sv = np.linalg.svd((R - R.mean(axis=0)).reshape(-1, 3), compute_uv=False)
    observability = float(sv[2] / sv[0]) if sv[0] > 0.0 else 0.0
    return LeverFit(
        lever=x[3:].copy(),
        wrist=x[:3].copy(),
        rms_m=rms,
        sweep_rad=sweep,
        start=start,
        observability=observability,
    )


def fit_lever(
    positions: np.ndarray,
    quats_wxyz: np.ndarray,
    window: int = 50,
    step: int = 5,
    min_sweep_deg: float = 20.0,
    min_observability: float = 0.3,
) -> LeverFit | None:
    """Best sliding-window fit of ``C_i = W + R_i @ r``.

    Returns ``None`` when no window rotates enough AND about enough axes.
    ``rms_m`` is the per-coordinate RMS of the 3-D residuals (metres); pure
    wrist rotation and millimetre tracking noise give ~1e-3, a translating
    hand much more. ``min_observability`` (0..1) rejects windows whose
    rotation cloud is nearly single-axis; pass 0 to disable the guard.
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
        if f is None or f.observability < min_observability:
            continue
        if best is None or f.rms_m < best.rms_m:
            best = f
    return best


def consensus_lever(
    positions: np.ndarray,
    quats_wxyz: np.ndarray,
    window: int = 50,
    step: int = 5,
    min_sweep_deg: float = 20.0,
    min_observability: float = 0.3,
    huber_m: float = 0.004,
) -> LeverConsensus | None:
    """One ``r`` shared by every well-conditioned window, robustly.

    Each window is allowed its own constant wrist position ``W_j``; that is
    eliminated by subtracting the window mean, leaving ``P_i - mean(P) =
    (R_i - mean(R)) r``. Stacking all qualifying windows and solving for the
    single ``r`` uses the whole recording instead of one lucky segment.
    Windows whose wrist was translating are inconsistent with any shared
    ``r``; an IRLS Huber weight (scale ``huber_m``) down-weights them.

    Returns ``None`` when no window is both long-sweeping and 3-axis.
    """
    positions = np.asarray(positions, np.float64)
    quats = np.asarray(quats_wxyz, np.float64)
    n = len(positions)
    if n < window or len(quats) != n:
        return None
    mats, valid = _rotation_mats(quats)
    blocks: list[tuple[LeverFit, np.ndarray, np.ndarray]] = []
    for start in range(0, n - window + 1, step):
        if not valid[start : start + window].all():
            continue
        f = _fit_window(positions, mats, start, window, min_sweep_deg)
        if f is None or f.observability < min_observability:
            continue
        R = mats[start : start + window]
        P = positions[start : start + window]
        A = (R - R.mean(axis=0)).reshape(-1, 3)
        b = (P - P.mean(axis=0)).reshape(-1)
        blocks.append((f, A, b))
    if not blocks:
        return None

    r = np.linalg.lstsq(
        np.concatenate([A for _, A, _ in blocks]),
        np.concatenate([b for _, _, b in blocks]),
        rcond=None,
    )[0]
    for _ in range(20):
        weights = []
        for _, A, b in blocks:
            rms = float(np.sqrt(np.mean((A @ r - b) ** 2)))
            weights.append(1.0 if rms <= huber_m else huber_m / rms)
        sw = np.concatenate([np.full(len(b), np.sqrt(w)) for w, (_, _, b) in zip(weights, blocks)])
        r = np.linalg.lstsq(
            np.concatenate([np.sqrt(w) * A for w, (_, A, _) in zip(weights, blocks)]),
            sw * np.concatenate([b for _, _, b in blocks]),
            rcond=None,
        )[0]

    per_window = np.array([f.lever for f, _, _ in blocks])
    resid = np.array([float(np.sqrt(np.mean((A @ r - b) ** 2))) for _, A, b in blocks])
    return LeverConsensus(
        lever=r,
        n_windows=len(blocks),
        spread_m=float(np.linalg.norm(per_window - r, axis=1).max()),
        rms_median_m=float(np.median(resid)),
        rms_max_m=float(np.max(resid)),
    )


def _lag_fit(
    positions: np.ndarray,
    mats: np.ndarray,
    valid: np.ndarray,
    lag: int,
    min_rot_deg: float,
    huber_m: float,
) -> tuple[np.ndarray, int, float, float] | None:
    """One short-lag estimate of ``r`` from ``dC = dR @ r``.

    Differencing consecutive samples removes any slowly varying wrist
    position ``W``; over ``lag`` samples (default 100 ms) a still-ish wrist
    contributes millimetres, while the rotation term carries the lever.
    Returns ``(lever, n_pairs, median_residual_m, observability)``.
    """
    n = len(positions)
    if lag < 1 or lag >= n:
        return None
    idx = np.arange(n - lag)
    keep = valid[idx] & valid[idx + lag]
    dR = mats[idx + lag] - mats[idx]
    # Frobenius norm of a relative rotation: ||R2 - R1||_F = 2*sqrt(2)*sin(theta/2)
    thr = 2.0 * np.sqrt(2.0) * np.sin(0.5 * np.radians(min_rot_deg))
    keep &= np.linalg.norm(dR, axis=(1, 2)) >= thr
    m = int(keep.sum())
    if m < 30:
        return None
    A = dR[keep].reshape(-1, 3)
    b = (positions[idx + lag] - positions[idx])[keep].reshape(-1)
    sv = np.linalg.svd(A, compute_uv=False)
    observability = float(sv[2] / sv[0]) if sv[0] > 0.0 else 0.0
    r = np.linalg.lstsq(A, b, rcond=None)[0]
    # Huber IRLS: pairs whose wrist moved more than ~huber_m are down-weighted.
    for _ in range(20):
        res = np.linalg.norm((A @ r - b).reshape(-1, 3), axis=1)
        w = np.where(res <= huber_m, 1.0, huber_m / np.maximum(res, 1e-12))
        sw = np.repeat(np.sqrt(w), 3)
        r = np.linalg.lstsq(sw[:, None] * A, sw * b, rcond=None)[0]
    res = np.linalg.norm((A @ r - b).reshape(-1, 3), axis=1)
    return r, m, float(np.median(res)), observability


def increment_lever(
    positions: np.ndarray,
    quats_wxyz: np.ndarray,
    lags: tuple[int, ...] = (2, 5, 10, 20),
    min_rot_deg: float = 1.0,
    huber_m: float = 0.002,
    min_observability: float = 0.3,
) -> LeverIncrement | None:
    """Estimate ``r`` from short-time increments, robust to wrist drift.

    The constant-W window fit assumes the wrist does not move during the
    window. A live calibration almost always drifts a few cm/s; that drift
    is then absorbed into ``r``, which is why low RMS and stable window
    widths can still produce a wrong vector. Differencing over a short lag
    (40-400 ms) makes the drift contribution small and, because it is not
    correlated with the rotation increment, unbiased after averaging.

    Estimates at several lags are combined by component-wise median; their
    spread is the honesty check. Lags whose stacked rotation increments are
    still ~single-axis are rejected by ``min_observability``.
    """
    positions = np.asarray(positions, np.float64)
    quats = np.asarray(quats_wxyz, np.float64)
    if len(quats) != len(positions):
        return None
    mats, valid = _rotation_mats(quats)
    per: list[tuple[int, np.ndarray, int, float, float]] = []
    for lag in sorted(set(int(v) for v in lags)):
        est = _lag_fit(positions, mats, valid, lag, min_rot_deg, huber_m)
        if est is None:
            continue
        r, m, resid, obs = est
        if obs < min_observability:
            continue
        per.append((lag, r, m, resid, obs))
    if not per:
        return None
    rs = np.array([p[1] for p in per])
    med = np.median(rs, axis=0)
    return LeverIncrement(
        lever=med,
        lags=tuple(p[0] for p in per),
        n_pairs=sum(p[2] for p in per),
        resid_median_m=float(np.median([p[3] for p in per])),
        spread_m=float(np.linalg.norm(rs - med, axis=1).max()),
        observability=float(min(p[4] for p in per)),
    )


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
    assert (
        fit.observability >= 0.3
    ), f"3-axis sweep scored observability {fit.observability:.2f} (< 0.3)"
    print(
        "[self-test] recovered "
        f"{np.round(fit.lever, 5)} vs true {r_true} (err {err * 1e3:.2f} mm), "
        f"rms {fit.rms_m * 1e3:.2f} mm, sweep {np.degrees(fit.sweep_rad):.0f} deg, "
        f"obs {fit.observability:.2f}"
    )
    cons = consensus_lever(np.asarray(pos), np.asarray(quats))
    assert cons is not None, "synthetic sweep gave no consensus"
    err_cons = float(np.linalg.norm(cons.lever - r_true))
    assert err_cons < 0.005, f"consensus lever error {err_cons * 1e3:.2f} mm (> 5 mm)"
    print(
        f"[self-test] consensus over {cons.n_windows} windows: "
        f"{np.round(cons.lever, 5)} (err {err_cons * 1e3:.2f} mm, "
        f"spread {cons.spread_m * 1e3:.2f} mm)"
    )
    inc = increment_lever(np.asarray(pos), np.asarray(quats))
    assert inc is not None, "synthetic sweep gave no increment estimate"
    err_inc = float(np.linalg.norm(inc.lever - r_true))
    assert err_inc < 0.005, f"increment lever error {err_inc * 1e3:.2f} mm (> 5 mm)"
    print(
        f"[self-test] increment estimate (lags {list(inc.lags)}): "
        f"{np.round(inc.lever, 5)} (err {err_inc * 1e3:.2f} mm, "
        f"lag spread {inc.spread_m * 1e3:.2f} mm)"
    )
    # A single-axis (yaw-only) sweep has an unobservable lever component: it
    # must be rejected by the observability guard even though the residual is
    # tiny and the sweep is wide. Without the guard the fit invents a value
    # for the missing component from tracking noise / wrist translation.
    yaw = sRot.from_rotvec(rng.normal(scale=0.09, size=(240, 1)) @ np.array([[0.0, 0.0, 1.0]]))
    yrots = sRot.identity()
    yquats, ypos = [], []
    for s in yaw:
        yrots = yrots * s
        R = yrots.as_matrix()
        yquats.append(yrots.as_quat(scalar_first=True))
        ypos.append(R @ r_true + rng.normal(scale=5e-4, size=3))
    ypos, yquats = np.asarray(ypos), np.asarray(yquats)
    assert fit_lever(ypos, yquats) is None, "single-axis sweep was not rejected"
    assert consensus_lever(ypos, yquats) is None, "single-axis consensus was not rejected"
    assert increment_lever(ypos, yquats) is None, "single-axis increment estimate was not rejected"
    free = fit_lever(ypos, yquats, min_observability=0.0)
    assert free is not None, "unguarded single-axis fit unexpectedly failed"
    assert (
        free.observability < 0.1
    ), f"single-axis sweep scored observability {free.observability:.2f} (>= 0.1)"
    print(
        "[self-test] single-axis sweep rejected by observability guard "
        f"(obs {free.observability:.3f}, rms {free.rms_m * 1e3:.2f} mm)"
    )
    # Constant wrist drift (a few cm/s) is absorbed by a 1 s constant-W fit
    # and biases it by tens of mm even though its RMS stays small. The
    # short-lag increment estimate differences the drift away and must stay
    # close to the truth -- this is the regression that motivated it.
    rng_d = np.random.RandomState(2)
    n_d = 800
    steps_d = sRot.from_rotvec(rng_d.normal(scale=0.04, size=(n_d, 3)))
    rots_d = sRot.identity()
    quats_d, mats_d = [], []
    for s in steps_d:
        rots_d = rots_d * s
        mats_d.append(rots_d.as_matrix())
        quats_d.append(rots_d.as_quat(scalar_first=True))
    mats_d = np.asarray(mats_d)
    drift = np.outer(np.arange(n_d) / 50.0, np.array([0.02, -0.01, 0.005]))
    pos_d = np.einsum("nij,j->ni", mats_d, r_true) + drift + rng_d.normal(scale=5e-4, size=(n_d, 3))
    win_d = fit_lever(pos_d, np.asarray(quats_d))
    inc_d = increment_lever(pos_d, np.asarray(quats_d))
    assert win_d is not None and inc_d is not None, "drift record did not qualify"
    err_win_d = float(np.linalg.norm(win_d.lever - r_true))
    err_inc_d = float(np.linalg.norm(inc_d.lever - r_true))
    assert err_inc_d < 0.003, f"increment drift error {err_inc_d * 1e3:.2f} mm (> 3 mm)"
    assert err_inc_d < err_win_d - 0.005, (
        f"increment ({err_inc_d * 1e3:.2f} mm) not clearly better than the "
        f"window fit ({err_win_d * 1e3:.2f} mm) under wrist drift"
    )
    print(
        "[self-test] constant wrist drift: increment err "
        f"{err_inc_d * 1e3:.2f} mm vs window err {err_win_d * 1e3:.2f} mm"
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
    min_observability: float,
) -> list[tuple[int, LeverFit]]:
    """Refit at -20% / requested / +20% widths around the SAME segment.

    A rigid lever on a still wrist gives the same ``r`` at every width; a
    translating wrist lets each width absorb a different amount of the
    translation, which is the failure mode a low RMS alone does not catch.

    Keep the window centered on ``best.start``: letting each width search its
    own best start compares different time segments (and different hand
    positions), which reports false instability. Same-segment fits isolate
    the width effect. Widths that lose 3-axis observability are skipped --
    their ``r`` is not comparable.
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
        if f is not None and f.observability >= min_observability:
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
    ap.add_argument(
        "--min-observability",
        type=float,
        default=0.3,
        help="minimum sigma_min/sigma_max of the window's stacked (R_i - mean R); "
        "below it the rotation is ~single-axis and the lever component along "
        "that axis is unobservable. 0 disables the guard",
    )
    ap.add_argument(
        "--consensus-huber-mm",
        type=float,
        default=4.0,
        help="Huber scale (mm) for down-weighting windows whose wrist moved "
        "while the shared lever is fitted",
    )
    ap.add_argument(
        "--lags",
        type=int,
        nargs="+",
        default=[2, 5, 10, 20],
        help="sample lags (50 Hz) for the increment estimator; estimates at "
        "several lags are medians and their spread is the honesty check",
    )
    ap.add_argument(
        "--min-inc-rot-deg",
        type=float,
        default=1.0,
        help="minimum controller rotation between a lag's two samples (degrees); "
        "smaller increments are dominated by tracking noise",
    )
    ap.add_argument(
        "--inc-huber-mm",
        type=float,
        default=2.0,
        help="Huber scale (mm) for down-weighting lag pairs whose wrist moved",
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
    fit_kwargs = dict(
        window=args.window,
        step=args.step,
        min_sweep_deg=args.min_sweep_deg,
    )
    fit = fit_lever(
        positions,
        quats,
        min_observability=args.min_observability,
        **fit_kwargs,
    )
    cons = consensus_lever(
        positions,
        quats,
        min_observability=args.min_observability,
        huber_m=args.consensus_huber_mm * 1e-3,
        **fit_kwargs,
    )
    inc = increment_lever(
        positions,
        quats,
        lags=tuple(args.lags),
        min_rot_deg=args.min_inc_rot_deg,
        huber_m=args.inc_huber_mm * 1e-3,
        min_observability=args.min_observability,
    )
    if fit is None and cons is None and inc is None:
        diag = fit_lever(positions, quats, min_observability=0.0, **fit_kwargs)
        if diag is not None:
            print(
                f"no {args.window}-sample window with >= {args.min_sweep_deg:g} deg "
                f"AND observability >= {args.min_observability:g}"
            )
            print(
                f"best rotation window start {diag.start} ({diag.start / 50.0:.2f} s): "
                f"sweep {np.degrees(diag.sweep_rad):.1f} deg, "
                f"observability {diag.observability:.2f}, rms {diag.rms_m * 1e3:.2f} mm"
            )
            print(
                "The rotation in that window is nearly single-axis: the lever "
                "component along that axis never moves the controller origin and "
                "cannot be measured from this motion. Re-record a still-wrist "
                "CONE/circle sweep that rotates the controller about at least two, "
                "ideally three, axes (yaw+pitch+roll). The increment estimator "
                "over the whole record failed the same guard."
            )
        else:
            print(
                f"no {args.window}-sample window with >= {args.min_sweep_deg:g} deg of "
                "controller rotation; record a slower/longer wrist-only sweep"
            )
        return 1
    print(f"record {args.record}  side {args.side}")
    if fit is not None:
        print(
            f"best window start {fit.start} ({fit.start / 50.0:.2f} s), "
            f"sweep {np.degrees(fit.sweep_rad):.1f} deg, "
            f"observability {fit.observability:.2f}, rms {fit.rms_m * 1e3:.2f} mm"
        )
        print(f"lever  {np.round(fit.lever, 5)} m  (|r| {np.linalg.norm(fit.lever) * 1e3:.1f} mm)")
        print(f"wrist  {np.round(fit.wrist, 5)} m  (rough stationary-wrist position)")
    else:
        print(
            "no single window is long-sweeping AND 3-axis; the record as a whole "
            "still covers enough axes for the increment estimator below"
        )
    if cons is not None:
        print(
            f"consensus over {cons.n_windows} well-conditioned windows: "
            f"{np.round(cons.lever, 5)} m  (|r| {np.linalg.norm(cons.lever) * 1e3:.1f} mm)"
        )
        print(
            f"  per-window spread {cons.spread_m * 1e3:.1f} mm, "
            f"window rms median {cons.rms_median_m * 1e3:.2f} mm "
            f"(max {cons.rms_max_m * 1e3:.2f})"
        )
        limit = max(0.02, 0.2 * float(np.linalg.norm(cons.lever)))
        if cons.spread_m > limit:
            print(
                f"WARNING: per-window lever spread > {limit * 1e3:.0f} mm -- the "
                "windows disagree (wrist translating and/or grip changed). The "
                "consensus is the dominant cluster; re-record a still-wrist "
                "cone sweep before trusting it."
            )
    if inc is not None:
        print(
            f"increment estimate (lags {list(inc.lags)} samples, "
            f"{inc.n_pairs} pair-lags): {np.round(inc.lever, 5)} m  "
            f"(|r| {np.linalg.norm(inc.lever) * 1e3:.1f} mm)"
        )
        print(
            f"  lag spread {inc.spread_m * 1e3:.2f} mm, median pair residual "
            f"{inc.resid_median_m * 1e3:.2f} mm, observability {inc.observability:.2f}"
        )
        if inc.spread_m > max(0.01, 0.1 * float(np.linalg.norm(inc.lever))):
            print(
                "WARNING: the lever changes with the differencing lag -- the wrist "
                "motion is correlated with the rotation (or the grip moved). The "
                "increment estimate averages over it; re-record a cleaner sweep."
            )
        if inc.resid_median_m > 0.005:
            print(
                "WARNING: median short-lag residual > 5 mm -- the wrist moved during "
                "the sweep; the increment estimate is more robust than the window "
                "fit but still has some translation bias. Prefer a cleaner re-record."
            )
        if cons is not None:
            d = float(np.linalg.norm(inc.lever - cons.lever))
            if d > max(0.015, 0.15 * float(np.linalg.norm(inc.lever))):
                print(
                    f"NOTE: increment and window-consensus levers differ by {d * 1e3:.0f} mm. "
                    "That is the expected signature of wrist drift corrupting the "
                    "constant-W window fit; use the increment estimate."
                )
    # The increment estimate is the one that does not assume a still wrist over
    # a whole second, so it is the recommended vector. The consensus is the
    # fallback; the single best window is reported for comparison only.
    lever = inc.lever if inc is not None else (cons.lever if cons is not None else fit.lever)
    if fit is not None:
        alts = _window_stability(
            positions,
            quats,
            fit,
            args.window,
            args.min_sweep_deg,
            args.min_observability,
        )
        if len(alts) >= 2:
            spread = max(float(np.linalg.norm(f.lever - lever)) for _, f in alts)
            print(
                f"window-size stability (same segment; spread vs reported {spread * 1e3:.1f} mm):"
            )
            for w, f in alts:
                d = float(np.linalg.norm(f.lever - lever))
                print(
                    f"  w={w:3d} start {f.start:4d}  lever {np.round(f.lever, 4)}  "
                    f"(d {d * 1e3:.1f} mm, obs {f.observability:.2f}, "
                    f"rms {f.rms_m * 1e3:.2f} mm)"
                )
            # 20 mm absolute or 20% of |r|, whichever is larger: a still wrist at
            # ~90 mm lever reproduces r within a few mm, while a translating one
            # shifts it by tens of mm as the window width changes.
            limit = max(0.02, 0.2 * float(np.linalg.norm(lever)))
            if spread > limit:
                print(
                    f"WARNING: lever spread > {limit * 1e3:.0f} mm across window widths -- "
                    "the wrist was probably moving, and the low RMS is absorbing that "
                    "translation. This is why the increment estimate is preferred."
                )
    if np.linalg.norm(lever) > 0.15:
        print(
            "WARNING: |r| > 150 mm is larger than a typical controller-to-wrist "
            "lever; treat it as translation contamination, not a handle offset."
        )
    print()
    print("re-run with:")
    print(f"  --ctrl-lever-{args.side} " + " ".join(f"{v:.4f}" for v in lever))
    print("(fit the other hand separately and pass both flags: the vectors differ)")
    if inc is None and cons is not None and cons.rms_median_m > 0.015:
        print(
            "\nWARNING: window rms > 15 mm and no increment estimate was available -- "
            "the hand was probably translating, not just rotating. The lever is not "
            "trustworthy; re-record a still-wrist sweep."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
