#!/usr/bin/env python3
"""A/B the two arm IK backends (mink QP vs legacy DLS) on one intent tape.

The migration made mink the default solver of the incremental chain.  The
claim that has to hold is not "mink is better" but "for the motions the
operator actually makes, the two backends agree closely enough that swapping
them is a change of solver, not a change of behaviour".  This tool is that
measurement: ONE tape, TWO tickers, tick-by-tick comparison of the joint
command stream.

    .venv/bin/python tools/gen_v4_intent_tape.py /tmp/v4_tape.npz
    .venv/bin/python tools/ab_arm_ik_backends.py --tape /tmp/v4_tape.npz

Both sides run the SHIPPED path: ``IncrementalTeleopTickerV3`` with
``full_pose_ik=True`` (7-DoF, position + orientation), differing only in
``backend``.  The ticker is otherwise built from its defaults, so the numbers
are the harness's numbers, not a hand-tuned scenario.

GATED (the migration's own budget):

  * max |dq| per joint          <= 0.05 rad
  * RMS  |dq| per joint         <= 0.01 rad
  * final |dq| per joint        <= 0.02 rad
  * ``holding`` / ``engaged``   identical on every tick, both sides
  * pos_err difference between the backends <= 2 mm on every tick
  * every commanded joint inside ``X2_ARM_JOINT_LIMITS_RAD``

REPORTED, NOT GATED -- and this is the honest part.  The absolute tracking
error is NOT a backend property on this tape: the synthetic figure motion
drives both arms through the elbow-near-straight singularity, where the arm
crawls (~3%/step, the documented round-4/5 behaviour) and lags the moving
target by tens of mm, and the right arm eventually latches the freeze guard
and holds a 172 mm error.  Both backends do this at the SAME ticks with the
same joint commands.  ``--- worst ticks`` therefore prints, per side, the
elbow angle at the worst tick and the best distance ANY joint vector in the
box could achieve to that target: when best-achievable is ~0 the shortfall is
tracking lag in a singular configuration, not an unreachable target and not
the solver.  Gating the absolute number would fail the tool for a property
the migration does not change.

The tool also measures the risk ACCEPTED when the per-tick step cap moved to
mink's ``VelocityLimit`` (round 5 on the robot concluded that per-joint step
clipping twists the step direction near the elbow-extended singularity):
per-tick step-direction cosine against DLS in the elbow-near-straight
windows, plus a bang-bang counter (a joint whose step reverses sign on
consecutive ticks).  Reported for the same reason -- it is the price of the
design, and a number is how you decide whether the price is acceptable.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "gear_sonic"))

from gear_sonic.utils.teleop.incremental_chain_v3 import (  # noqa: E402
    IncrementalTeleopTickerV3,
)
from gear_sonic.utils.teleop.solver.arm.x2_arm_ik import (  # noqa: E402
    X2_ARM_JOINT_LIMITS_RAD,
)

AXES = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_yaw",
    "wrist_pitch",
    "wrist_roll",
)
ELBOW = 3
# elbow range is [-2.3556, 0]: |elbow| below this counts as "arm nearly
# straight", the singular configuration round 5 warned about
STRAIGHT_ELBOW_RAD = 0.15
# a step this small is noise, not a direction
MIN_STEP_RAD = 1e-3
# a reversal counts as bang-bang only if both ticks moved meaningfully
BANGBANG_STEP_RAD = 0.01

BUDGET = {"dq_max": 0.05, "dq_rms": 0.01, "dq_final": 0.02, "ee_diff_m": 0.002}
SIDES = ("left", "right")


def _frames(tape: dict, dt: float):
    for k, t in enumerate(tape["t"]):
        yield k, {
            "t": 1000.0 + k * dt,
            "trigger": tape["trigger"][k],
            "grip": tape["grip"][k],
            "lctrl_pose": np.asarray(tape["lctrl_pose"][k], np.float64),
            "rctrl_pose": np.asarray(tape["rctrl_pose"][k], np.float64),
            "smpl_joints": np.asarray(tape["smpl_joints"][k], np.float64),
        }


def run_backend(tape: dict, backend: str) -> dict:
    """One full pass over the tape. Pure: no plant, no clock, no wire."""
    ticker = IncrementalTeleopTickerV3(backend=backend)
    n = len(tape["t"])
    out = {
        "backend": backend,
        "resolved": ticker.arms["right"].backend,
        "q": {s: np.zeros((n, 7)) for s in SIDES},
        # where the COMMANDED q puts the hand: the quantity the operator
        # sees. Gate on this, not on the solver's self-reported pos_err --
        # that one measures the candidate step the freeze guard may have
        # discarded, so it is not the command.
        "ee": {s: np.zeros((n, 3)) for s in SIDES},
        "holding": {s: np.zeros(n, bool) for s in SIDES},
        "engaged": {s: np.zeros(n, bool) for s in SIDES},
        "pos_err": {s: np.zeros(n) for s in SIDES},
        "rot_err": {s: np.zeros(n) for s in SIDES},
        # the target the chain actually solved against (post keep-out, post
        # reach sphere, post slew) -- used to explain the tracking numbers
        "target": {s: np.full((n, 3), np.nan) for s in SIDES},
        "clamped": {s: np.zeros(n, bool) for s in SIDES},
        "reach": {s: 0.0 for s in SIDES},
        "shoulder": {s: np.zeros(3) for s in SIDES},
    }
    for k, frame in _frames(tape, ticker.dt):
        st = ticker.step(frame, now=frame["t"])
        for side, arm in ticker.arms.items():
            out["q"][side][k] = st["q"][side]
            out["ee"][side][k] = arm.solver.fk(st["q"][side])[0]
            out["holding"][side][k] = bool(arm.holding)
            out["engaged"][side][k] = bool(arm.engaged)
            out["pos_err"][side][k] = float(arm.last_ik.pos_err_m)
            out["rot_err"][side][k] = float(arm.last_ik.rot_err_rad)
            out["reach"][side] = float(arm.max_reach)
            out["shoulder"][side] = np.asarray(arm.shoulder, np.float64)
            tgt = arm._tgt_slewed
            if tgt is not None:
                out["target"][side][k] = tgt
                out["clamped"][side][k] = bool(
                    np.linalg.norm(tgt - arm.shoulder) >= arm.max_reach - 1e-9
                )
    return out


def _report_agreement(a: dict, b: dict) -> bool:
    print(f"\n== joint command stream: {a['resolved']} vs {b['resolved']} ==")
    ok = True
    for side in SIDES:
        live = a["engaged"][side] & b["engaged"][side]
        if not live.any():
            print(f"  {side}: never engaged on both sides -- nothing compared")
            ok = False
            continue
        d = a["q"][side][live] - b["q"][side][live]
        mx = np.abs(d).max(axis=0)
        rms = np.sqrt((d**2).mean(axis=0))
        fin = np.abs(d[-1])
        print(f"  {side}: {int(live.sum())} of {len(live)} ticks engaged on both sides")
        for i, name in enumerate(AXES):
            bad = (
                mx[i] > BUDGET["dq_max"] or rms[i] > BUDGET["dq_rms"] or fin[i] > BUDGET["dq_final"]
            )
            ok &= not bad
            print(
                f"    {name:15s} max {mx[i]:9.5f}  RMS {rms[i]:9.5f}  final {fin[i]:9.5f}"
                f"   {'FAIL' if bad else 'ok'}   (budget {BUDGET['dq_max']}/{BUDGET['dq_rms']}/"
                f"{BUDGET['dq_final']})"
            )
    return ok


def _report_flags(a: dict, b: dict) -> bool:
    print("\n== safety flags (must be identical every tick) ==")
    ok = True
    for run in (a, b):
        latch = {s: _runs(run["holding"][s]) for s in SIDES}
        print(
            f"  {run['resolved']:5s} latch windows: " + ", ".join(f"{s} {latch[s]}" for s in SIDES)
        )
    for side in SIDES:
        for flag in ("holding", "engaged"):
            diff = np.flatnonzero(a[flag][side] != b[flag][side])
            ok &= diff.size == 0
            where = ""
            if diff.size:
                k = int(diff[0])
                where = f"  first at tick {k} ({a['resolved']}={bool(a[flag][side][k])})"
            print(f"  {side:5s} {flag:8s} differing ticks: {diff.size}{where}")
    return ok


def _runs(mask: np.ndarray) -> str:
    out, k, n = [], 0, len(mask)
    while k < n:
        if mask[k]:
            j = k
            while j < n and mask[j]:
                j += 1
            out.append(f"{k}..{j - 1}")
            k = j
        else:
            k += 1
    return "[" + ", ".join(out) + "]" if out else "none"


def _report_tracking(a: dict, b: dict) -> bool:
    print("\n== tracking ==")
    print(
        f"  GATE: the COMMANDED hand pose may not differ by more than "
        f"{BUDGET['ee_diff_m'] * 1e3:.0f} mm -- the operator must not be able to tell "
        f"which solver is running"
    )
    ok = True
    for side in SIDES:
        live = a["engaged"][side] & b["engaged"][side]
        if not live.any():
            continue
        diff = np.linalg.norm(a["ee"][side] - b["ee"][side], axis=1)[live]
        worst = int(np.flatnonzero(live)[int(np.argmax(diff))])
        bad = float(diff.max()) > BUDGET["ee_diff_m"]
        ok &= not bad
        print(
            f"  {side:5s} hand difference max {diff.max() * 1e3:6.3f} mm (tick {worst}), "
            f"mean {diff.mean() * 1e3:6.4f} mm   {'FAIL' if bad else 'ok'}"
        )
    print("\n  solver-reported pos_err (reported, NOT gated -- see the module docstring):")
    for side in SIDES:
        live = a["engaged"][side] & b["engaged"][side]
        if not live.any():
            continue
        for run in (a, b):
            pe = run["pos_err"][side][live]
            held = run["holding"][side][live]
            free = pe[~held]
            print(
                f"    {run['resolved']:5s} {side:5s} all ticks  max {pe.max() * 1e3:8.3f}  "
                f"mean {pe.mean() * 1e3:7.3f} mm | unlatched max "
                f"{(free.max() if free.size else 0.0) * 1e3:8.3f} mm ({int((~held).sum())} ticks)"
            )
    _report_worst_ticks(a)
    return ok


def _report_worst_ticks(a: dict) -> None:
    """Why the absolute error is large: measure it, don't assert it."""
    print("\n  --- worst ticks, explained (each side, on the dls run)")
    for side in SIDES:
        live = np.flatnonzero(a["engaged"][side] & (a["pos_err"][side] > 0.0))
        if not live.size:
            continue
        off = 0 if side == "left" else 7
        lo = np.asarray(X2_ARM_JOINT_LIMITS_RAD, float)[off : off + 7, 0]
        hi = np.asarray(X2_ARM_JOINT_LIMITS_RAD, float)[off : off + 7, 1]
        k = int(live[np.argmax(a["pos_err"][side][live])])
        tgt = a["target"][side][k]
        q = a["q"][side][k]
        radius = float(np.linalg.norm(tgt - a["shoulder"][side]))
        print(
            f"    {side:5s} tick {k:4d} ({k * 0.02:5.2f} s) pos_err {a['pos_err'][side][k] * 1e3:8.3f} mm  "
            f"elbow {q[ELBOW]:+.4f} rad ({'NEAR-STRAIGHT' if abs(q[ELBOW]) < STRAIGHT_ELBOW_RAD else 'bent'})  "
            f"latched {'yes' if a['holding'][side][k] else 'no'}  "
            f"target radius {radius:.3f} of {a['reach'][side]:.3f} m"
        )
        solver = a["_solver"][side]
        best = float("inf")
        for qq in (q, np.zeros(7), 0.5 * (lo + hi)):
            r = least_squares(
                lambda x: solver.fk(np.asarray(x, np.float64))[0] - tgt,
                qq,
                bounds=(lo, hi),
                xtol=1e-12,
                ftol=1e-12,
                max_nfev=20000,
            )
            best = min(best, float(np.linalg.norm(r.fun)))
        print(
            f"          best distance ANY joint vector in the box reaches that target: "
            f"{best * 1e3:.3f} mm -> "
            + (
                "target REACHABLE, the shortfall is tracking lag"
                if best < 1e-3
                else "target OUT OF REACH"
            )
        )


def _report_round5_risk(a: dict, b: dict) -> None:
    """Step-direction cosine + bang-bang, in the elbow-near-straight windows.

    UNGATED: the accepted risk of handing the step cap to mink's
    ``VelocityLimit``.  The direction cosine compares the one-tick joint step
    each backend took -- from its own state, since the runs diverge; read it
    as a smoke test (a cosine near -1 would mean the backends push opposite
    ways at the singularity, which is the failure round 5 saw on the robot).
    """
    print("\n== round-5 risk: elbow-near-straight windows (reported, not gated) ==")
    for side in SIDES:
        live = a["engaged"][side] & b["engaged"][side]
        if not live.any():
            continue
        qa, qb = a["q"][side][live], b["q"][side][live]
        da, db = np.diff(qa, axis=0), np.diff(qb, axis=0)
        window = (
            (np.abs(qa[:-1, ELBOW]) < STRAIGHT_ELBOW_RAD)
            & (np.abs(da) > MIN_STEP_RAD).any(axis=1)
            & (np.abs(db) > MIN_STEP_RAD).any(axis=1)
        )
        if window.any():
            na = np.linalg.norm(da[window], axis=1)
            nb = np.linalg.norm(db[window], axis=1)
            cos = np.sum(da[window] * db[window], axis=1) / (na * nb)
            print(
                f"  {side:5s} |elbow| < {STRAIGHT_ELBOW_RAD} rad: {int(window.sum()):4d} ticks  "
                f"direction cos min {cos.min():+.3f} mean {cos.mean():+.3f}  "
                f"peak step {a['resolved']} {na.max():.4f} vs {b['resolved']} {nb.max():.4f} rad/tick"
                f" (cap {0.30 / 0.02:.0f} rad/s = 0.30 rad/tick)"
            )
        else:
            print(f"  {side:5s} |elbow| < {STRAIGHT_ELBOW_RAD} rad: no moving tick in the window")
        for run, d in ((a, da), (b, db)):
            rev = (np.abs(d[:-1]) > BANGBANG_STEP_RAD) & (np.abs(d[1:]) > BANGBANG_STEP_RAD)
            flips = (np.sign(d[:-1]) != np.sign(d[1:])) & rev
            per_joint = flips.sum(axis=0)
            worst = int(per_joint.max())
            print(
                f"  {run['resolved']:5s} bang-bang: {int(flips.any(axis=1).sum()):4d} ticks, "
                f"{int(flips.sum()):4d} joint-reversals, worst joint "
                f"{AXES[int(np.argmax(per_joint))] if worst else '-'} x{worst}"
            )


def _report_ranges(runs) -> bool:
    print("\n== joint catalog (every commanded tick, both backends) ==")
    L = np.asarray(X2_ARM_JOINT_LIMITS_RAD, float)
    ok = True
    for run in runs:
        worst = 0.0
        for side in SIDES:
            off = 0 if side == "left" else 7
            q = run["q"][side][run["engaged"][side]]
            if not q.size:
                continue
            worst = max(
                worst,
                float(np.max(np.maximum(L[off : off + 7, 0] - q, q - L[off : off + 7, 1]))),
            )
        bad = worst > 1e-9
        ok &= not bad
        print(f"  {run['resolved']:5s} worst excursion {worst:.2e} rad   {'FAIL' if bad else 'ok'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tape", default="/tmp/v4_tape.npz", help="pico_intent_sender-shaped .npz")
    ap.add_argument("--json", default=None, help="write both command streams here (.npz)")
    args = ap.parse_args()

    path = Path(args.tape)
    if not path.exists():
        print(
            f"no tape at {path}\nmake one with: .venv/bin/python tools/gen_v4_intent_tape.py {path}",
            file=sys.stderr,
        )
        return 2
    raw = np.load(path)
    tape = {k: raw[k] for k in raw.files}

    runs = [run_backend(tape, backend) for backend in ("dls", "mink")]
    for run, backend in zip(runs, ("dls", "mink")):
        # a private solver per side, only for the joint-box reachability search
        ticker = IncrementalTeleopTickerV3(backend=backend)
        run["_solver"] = {s: ticker.arms[s].solver for s in SIDES}
    a, b = runs
    n = len(tape["t"])
    print(
        f"tape {path}: {n} frames ({n * IncrementalTeleopTickerV3().dt:.1f} s) -> "
        f"{a['resolved']} vs {b['resolved']}"
    )

    ok = True
    ok &= _report_agreement(a, b)
    ok &= _report_flags(a, b)
    ok &= _report_tracking(a, b)
    ok &= _report_ranges(runs)
    _report_round5_risk(a, b)

    if args.json:
        np.savez_compressed(
            args.json,
            **{
                f"{run['resolved']}_{side}_{name}": run[name][side]
                for run in runs
                for side in SIDES
                for name in ("q", "holding", "engaged", "pos_err", "rot_err")
            },
        )
        print(f"\nwrote {args.json}")

    print(f"\nA/B: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
