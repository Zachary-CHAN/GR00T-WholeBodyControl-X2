#!/usr/bin/env python
"""Cap the operator ROOT tilt in a recorded Pico tape (offline causal intervention).

The two bundled X2 Pico tapes make the robot fall on every sustained root tilt
past ~30 deg: the policy tracks the recorded lean and trips the deploy tilt
watchdog 4.22 +/- 0.11 s after the event onset (2 tapes / 3 cut points / 4 runs,
residual < 0.13 s).  This tool rewrites the tape's root quaternion so that tilt
never exceeds --limit-deg, and nothing else changes, which isolates ONE
question: does bounding the root orientation alone prevent the fall?

It calls the SAME ``cap_root_tilt`` the live guard uses
(gear_sonic/utils/teleop/root_tilt_guard.py).  The clamp is applied in the DEVICE
frame (body[:,0,3:7], up = +Y), which is equivalent to clamping the wire
quaternion by the frame lemma proved in that module -- verified here as gate G2.

The rewrite is RIGID: the correction goes onto all 24 frames and the positions
turn about the pelvis.  Editing the root quaternion alone would leave every
child's world orientation in place and push the whole lean into the joint
channel instead of removing it (measured: 60.6 deg of pose_aa, 0.86 m of
smpl_joints_local).

This tool PREDICTS; it does not replace the online experiment.  Its obs differs
from the live guard's by the pivot term documented on ``cap_body_root_tilt``:
a rigid TRANSLATION of the whole local skeleton, <= 0.081 m on seg01, with zero
deformation.  Only ``body`` is written; ``stamp_ns`` keeps its ORIGINAL values,
since replay takes stamp_ns[0] as its base and only the span matters.

In-process gates (non-zero exit on failure, as in pico_tape_to_smpl_obs.py):
  G1 round-trip    device -> wire -> device recovers the recorded root rotation.
                   This is not ceremony: an inverted composition order here is
                   invisible on inspection and costs exactly 180 deg per frame.
  G2 cap holds     tilt of the written tape, recomputed through the PRODUCTION
                   chain (compute_from_body_poses), is <= limit + 1e-4.
  G3 no collateral under-cap frames stay bit-identical; the lean azimuth and the
                   heading (twist) do not move.
  G4 rigid         joints' == J0 + dR @ (joints - J0) through the production
                   chain: one rigid rotation about the SMPL rest pelvis.  NOT
                   dR @ joints -- joints[0] == J0 for every input, so an
                   origin-pivot check reports a phantom (I - dR) @ J0 residual
                   (0.067 m at 61 deg) that is uniform across all 24 joints.
  G5 undeformed    the local skeleton (the obs's 72 values) only TRANSLATES, by
                   exactly that pivot offset, and never changes shape.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as sRot

from gear_sonic.utils.teleop.root_tilt_guard import (
    UP,
    cap_body_root_tilt,
    cap_root_tilt,
    device_from_wire,
    tilt_deg,
    wire_from_device,
)

FALL_DELAY_S = 4.22     # measured: event onset -> tilt watchdog trip
SUSTAIN_S = 1.5         # measured: a >30 deg excursion shorter than this never fell
BOUNDARY_DEG = 30.0     # measured: the line sustained excursions cross to be fatal


def report_events(tilt: np.ndarray, fps: float, limit: float) -> None:
    """Print tilt stats and the first SUSTAINED over-boundary excursion.

    First-in-time, not largest: seg01 has three events of near-identical
    duration and the robot fell at the earliest one.
    """
    over = tilt > BOUNDARY_DEG
    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(over):
        if over[i]:
            j = i
            while j + 1 < len(over) and over[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    sustained = [(a, b) for a, b in runs if (b - a + 1) / fps >= SUSTAIN_S]
    print(f"[tiltcap] tilt: median {np.median(tilt):5.2f}  p95 {np.percentile(tilt, 95):6.2f}  "
          f"max {tilt.max():6.2f} deg")
    print(f"[tiltcap] excursions >{BOUNDARY_DEG:.0f} deg: {len(runs)} total, "
          f"{len(sustained)} sustained >={SUSTAIN_S}s")
    if sustained:
        a, b = sustained[0]
        onset = a / fps
        print(f"[tiltcap] FIRST sustained event at t={onset:.3f}s "
              f"({b - a + 1} frames, peak {tilt[a:b + 1].max():.1f} deg)")
        print(f"[tiltcap]   -> unguarded prediction: tilt watchdog at "
              f"t={onset + FALL_DELAY_S:.3f}s")
    else:
        print("[tiltcap] no sustained event: this tape is not expected to fall")
    if limit > 0:
        print(f"[tiltcap] frames the {limit:.0f} deg cap rewrites: "
              f"{int((tilt > limit).sum())}/{len(tilt)} ({100 * (tilt > limit).mean():.2f}%)")
    else:
        print("[tiltcap] cap is OFF: every frame passes through unchanged")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tape", required=True, type=Path)
    ap.add_argument("--out", type=Path, help="write the capped tape here")
    ap.add_argument("--limit-deg", type=float, default=25.0,
                    help="root tilt cap in degrees; 0 = off (plain copy)")
    ap.add_argument("--cut-from-s", type=float, default=None,
                    help="drop frames before this time (stamp_ns is preserved)")
    ap.add_argument("--fps", type=float, default=None,
                    help="frame rate for reporting; default is MEASURED from the "
                         "stamp_ns span. Do not assume 50: replay follows the device "
                         "timestamps, and these tapes run ~82.5 fps, so frames/fps is "
                         "off by 65%%")
    ap.add_argument("--report-only", action="store_true",
                    help="analyse the input and exit without writing")
    args = ap.parse_args()

    z = np.load(args.tape, allow_pickle=True)
    fields = {k: z[k] for k in z.files}
    body = fields["body"].astype(np.float64)
    stamp = fields["stamp_ns"]

    if args.cut_from_s is not None:
        base = stamp[0]
        keep = np.flatnonzero(stamp >= base + int(args.cut_from_s * 1e9))
        if not len(keep):
            print(f"[tiltcap] --cut-from-s {args.cut_from_s} drops every frame")
            return 1
        body = body[keep[0]:]
        for k in fields:
            fields[k] = fields[k][keep[0]:]
        print(f"[tiltcap] cut from t={args.cut_from_s}s: kept {len(body)} frames, "
              f"stamp_ns preserved")

    fps = args.fps
    if fps is None:
        span_s = (stamp[-1] - stamp[0]) / 1e9
        fps = (len(stamp) - 1) / span_s if span_s > 0 else 50.0
        print(f"[tiltcap] measured {fps:.2f} fps from the stamp span "
              f"({span_s:.1f}s over {len(stamp)} frames)")

    Rw = wire_from_device(body)
    report_events(tilt_deg(Rw), fps, args.limit_deg)

    # ---- G1: device -> wire -> device must recover the recorded rotation -----
    rt = device_from_wire(Rw)
    err = np.degrees(np.linalg.norm(
        (sRot.from_quat(rt) * sRot.from_quat(body[:, 0, 3:7]).inv()).as_rotvec(), axis=1))
    g1 = err.max() < 1e-4
    print(f"[tiltcap] G1 round-trip: max {err.max():.2e} deg -> {'OK' if g1 else 'FAIL'}")
    if not g1:
        return 1

    if args.report_only:
        return 0
    if args.out is None:
        print("[tiltcap] --out is required unless --report-only")
        return 1

    # ---- clamp ---------------------------------------------------------------
    # RIGID: the correction goes onto all 24 frames, not just the root, so the
    # pipeline's FK sees one rigid rotation about the pelvis (gate G4).  Editing
    # the root alone leaves every child's world orientation in place, so the
    # pipeline re-derives a smpl_joints_local that absorbs the difference (60.6
    # deg of pose_aa, measured) -- the lean would move into the joint channel
    # instead of going away.
    capped_wire, hit = cap_root_tilt(Rw.as_quat()[:, [3, 0, 1, 2]], args.limit_deg,
                                     return_mask=True)
    out_body = cap_body_root_tilt(body, args.limit_deg)

    # ---- G2: the cap holds through the real chain ----------------------------
    # Skipped when the cap is OFF (--limit-deg 0 is the plain-copy mode used to
    # build an uncapped control); asserting "the cap holds" against a disabled
    # cap is meaningless, not a failure.
    if args.limit_deg > 0:
        from gear_sonic.scripts.pico_manager_thread_server import compute_from_body_poses
        from gear_sonic.scripts.pico_tape_to_smpl_obs import SMPL_PARENTS

        import torch

        J0 = np.asarray(torch.load("gear_sonic/data/human/human_joints_info.pkl")["J"])[0]
        J0 = J0.astype(np.float64)

        idx = np.linspace(0, len(out_body) - 1, min(600, len(out_body))).astype(int)
        got = np.empty(len(idx))
        d_rigid = np.empty(len(idx))    # G4: distance from a rigid rotation about J0
        d_shape = np.empty(len(idx))    # G5: spread of the local-skeleton drift
        d_pivot = np.empty(len(idx))    # G5: |(dR^-1 - I) J0|, the live-guard offset
        d_pivot_err = np.empty(len(idx))  # G5: measured drift norm minus that bound
        for n, i in enumerate(idx):
            rc = compute_from_body_poses(SMPL_PARENTS, "cpu", out_body[i])
            ro = compute_from_body_poses(SMPL_PARENTS, "cpu", body[i])
            # the chain returns w-first (remove_smpl_base_rot(..., w_last=False))
            q = np.asarray(rc["global_orient_quat"]).reshape(-1)[:4]
            qi = np.asarray(ro["global_orient_quat"]).reshape(-1)[:4]
            got[n] = tilt_deg(sRot.from_quat(q[[1, 2, 3, 0]]))[0]
            E = sRot.from_quat(q[[1, 2, 3, 0]]) * sRot.from_quat(qi[[1, 2, 3, 0]]).inv()

            # G4: one rigid rotation about the SMPL rest pelvis.  An origin-pivot
            # check would report |(I - E) J0| here, uniform across all 24 joints.
            jo = np.asarray(ro["joints"]).reshape(-1, 3)
            jc = np.asarray(rc["joints"]).reshape(-1, 3)
            d_rigid[n] = np.abs(jc - (J0 + E.apply(jo - J0))).max()

            # G5: what the obs's 72 values actually do.  The live guard leaves
            # smpl_joints_local alone; ours shifts it by the pivot offset only --
            # a rigid translation of the whole skeleton, never a deformation.
            dl = (np.asarray(rc["smpl_joints_local"]).reshape(-1, 3)
                  - np.asarray(ro["smpl_joints_local"]).reshape(-1, 3))
            d_shape[n] = dl.std(axis=0).max()
            d_pivot[n] = np.linalg.norm(E.inv().apply(J0) - J0)
            d_pivot_err[n] = abs(np.linalg.norm(dl, axis=1).max() - d_pivot[n])

        g2 = got.max() <= args.limit_deg + 1e-4
        print(f"[tiltcap] G2 cap holds through the production chain: "
              f"max {got.max():.4f} deg (limit {args.limit_deg}) -> {'OK' if g2 else 'FAIL'}")
        if not g2:
            return 1
        g4 = d_rigid.max() < 1e-3
        print(f"[tiltcap] G4 rigid about the pelvis J0 (NOT dR @ joints): "
              f"max residual {d_rigid.max():.2e} m -> {'OK' if g4 else 'FAIL'}")
        if not g4:
            return 1
        g5 = d_shape.max() < 1e-6 and d_pivot_err.max() < 1e-3
        print(f"[tiltcap] G5 local skeleton undeformed: shape spread {d_shape.max():.2e} m, "
              f"drift == |(dR^-1 - I) J0| to {d_pivot_err.max():.2e} m, "
              f"pivot offset max {d_pivot.max():.4f} m "
              f"(live guard emits 0; the tape side cannot) -> {'OK' if g5 else 'FAIL'}")
        if not g5:
            return 1
    else:
        print("[tiltcap] G2/G4/G5 skipped: cap is OFF (--limit-deg 0), writing a plain copy")

    # ---- G3: under-cap frames untouched, lean azimuth and heading preserved ---
    Ro = sRot.from_quat(capped_wire[:, [1, 2, 3, 0]])
    under = tilt_deg(Rw) <= args.limit_deg
    bit = all(np.array_equal(out_body[i, 0, 3:7], body[i, 0, 3:7])
              for i in np.flatnonzero(under))

    def lean_azimuth(R: sRot) -> np.ndarray:
        """Direction the up-axis leans toward, as an angle in the horizontal plane."""
        h = np.atleast_2d(R.apply(UP))
        h = h - (h @ UP)[:, None] * UP
        return np.arctan2(h[:, 1], h[:, 0])

    def swing(R: sRot) -> sRot:
        """The tilt part of R = swing * twist (twist fixes UP)."""
        u = np.atleast_2d(R.apply(UP))
        ax = np.cross(UP, u)
        n = np.linalg.norm(ax, axis=1, keepdims=True)
        th = np.arccos(np.clip(u @ UP, -1.0, 1.0))
        safe = np.where(n < 1e-12, 1.0, n)
        return sRot.from_rotvec(th[:, None] * ax / safe)

    mv = np.flatnonzero(~under)
    if len(mv):
        dz = lean_azimuth(Ro[mv]) - lean_azimuth(Rw[mv])
        d_az = np.degrees(np.abs(np.angle(np.exp(1j * dz))))
        # R_out must equal swing_out * swing_in^-1 * R_in exactly (heading kept)
        want = swing(Ro[mv]) * swing(Rw[mv]).inv() * Rw[mv]
        d_hd = np.degrees(np.linalg.norm((Ro[mv].inv() * want).as_rotvec(), axis=1))
    else:
        d_az = d_hd = np.zeros(1)
    g3 = bit and d_az.max() < 1e-3 and d_hd.max() < 1e-3
    print(f"[tiltcap] G3 fidelity: under-cap bit-identical {bit} | "
          f"azimuth drift {d_az.max():.1e} deg | heading drift {d_hd.max():.1e} deg "
          f"-> {'OK' if g3 else 'FAIL'}")
    if not g3:
        return 1

    fields["body"] = out_body.astype(fields["body"].dtype)
    np.savez(args.out, **fields)
    print(f"[tiltcap] wrote {args.out} ({len(out_body)} frames, "
          f"limit {args.limit_deg:.0f} deg)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
