#!/usr/bin/env python3
"""Level-0 alignment gate: the sim harness and the robot node, one command stream.

The X2 incremental upper teleop exists twice as a *front end*:

  * sim:   gear_sonic/scripts/incremental_upper_teleop_mujoco.py (plant = MJCF)
  * robot: czq_examples/incremental_upper_teleop.py on PC2 (plant = the robot,
           front end vendored flat into x2_sdk_v100/src/py_examples/py_examples/)

Both drive the SAME :class:`IncrementalTeleopTicker` from
``gear_sonic/utils/teleop/incremental_chain.py``.  The robot tree ships a
flat copy of that module; this tool imports what Python resolves from the
node's directory and PRINTS the resolved chain path (a vendored copy may
carry older semantics).  For one state dict ``st`` the only place the two
can diverge is the mapping ``st -> (joint names, positions, gains)`` each
side applies before handing values to its plant.

This tool drives ONE ticker over a v4 tape and, every tick, compares:

  * sim:  the harness's assembly (mirrored from
          incremental_upper_teleop_mujoco.py -- see _check_mirror_sync,
          which fails loudly if the source lines move or change)
  * node: ``TeleopCore.commands(st)`` -- exactly the payload of the hal
          publishers (``/aima/hal/joint/{arm,waist}/command``)

Position AND gain equality is required BIT-EXACT.  It also cross-checks the
two argparse default sets: "run it with no flags" must mean the same chain on
both sides.  Deterministic by construction -- no wire, no clock, no plant --
so a PASS is a judgement-free gate.  Runbook: czq_test_v2_incremental.md §1.

    python tools/gen_v4_intent_tape.py /tmp/v4_tape.npz
    .venv/bin/python tools/compare_sim_node_command_stream.py --tape /tmp/v4_tape.npz
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np

DEFAULT_SIM = "gear_sonic/scripts/incremental_upper_teleop_mujoco.py"
DEFAULT_ROBOT = (
    "/home/nvidia/Documents/x2_sdk_v100/src/py_examples/py_examples/incremental_upper_teleop.py"
)

# The two harness lines this tool mirrors.  If either disappears from the
# source, the mirror below is stale and the comparison would be vacuous.
MIRROR_LINES = (
    "for i, n in enumerate(ubt.ARM_NAMES[base : base + 7]):",
    'tgt["waist_yaw_joint"] = (st["waist"], ubt.WAIST[0][4], ubt.WAIST[0][5])',
)

FLAGS = [
    "--stale-s",
    "--position-scale",
    "--rotation-weight",
    "--rot-err-clip",
    "--input-tau",
    "--target-slew",
    "--trigger-thresh",
    "--waist-clamp",
    "--waist-sign",
]


def argparse_defaults(path: Path) -> dict[str, str]:
    """{flag: default expression} via AST -- handles single- and multi-line
    add_argument declarations and ignores parentheses inside help strings."""
    out: dict[str, str] = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            for kw in node.keywords:
                if kw.arg == "default":
                    out[node.args[0].value] = ast.unparse(kw.value)
    return out


def check_mirror_sync(sim_path: Path) -> list[str]:
    txt = sim_path.read_text()
    return [line for line in MIRROR_LINES if line not in txt]


def main() -> int:
    here = Path(__file__).resolve()
    repo = here.parent.parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--tape",
        default="/tmp/v4_tape.npz",
        help="v4 intent tape (tools/gen_v4_intent_tape.py); default %(default)s",
    )
    ap.add_argument("--sim-script", default=str(repo / DEFAULT_SIM))
    ap.add_argument(
        "--robot-node",
        default=DEFAULT_ROBOT,
        help="the node on the robot (or a PC2-flat staging copy of it)",
    )
    ap.add_argument("--repo", default=str(repo))
    ap.add_argument(
        "--intent-port",
        type=int,
        default=19999,
        help="scratch port -- TeleopCore() binds a receiver even though "
        "this tool never feeds it frames through the wire",
    )
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    sys.path.insert(0, str(Path(args.robot_node).resolve().parent))
    import incremental_upper_teleop as node_mod  # noqa: E402

    chain, ubt = node_mod.chain, node_mod.ubt
    print(f"repo  {args.repo}")
    print(f"sim   {args.sim_script}")
    print(f"node  {Path(node_mod.__file__).resolve()}")
    repo_chain = (Path(args.repo) / "gear_sonic/utils/teleop/incremental_chain.py").resolve()
    resolved_chain = Path(chain.__file__).resolve()
    print(f"chain {resolved_chain}   (must be the file both trees share)")
    if resolved_chain != repo_chain:
        print(f"WARNING: chain resolved OUTSIDE the repo ({repo_chain} expected);")
        print("         a vendored copy may still carry the OLD trigger/grip roles --")
        print("         this gate only says the two ASSEMBLIES agree on whatever chain won.")

    stale = check_mirror_sync(Path(args.sim_script))
    if stale:
        print("\nFAIL: sim harness no longer contains the assembly lines this tool mirrors:")
        for s in stale:
            print(f"      {s!r}")
        print("      Update the mirror in this file before trusting the comparison.")
        return 1

    # ------------------------------------------------------------ configs
    defaults: dict[str, dict[str, str]] = {
        "sim": argparse_defaults(Path(args.sim_script)),
        "node": argparse_defaults(Path(node_mod.__file__).resolve()),
    }
    ev_env = {"TRIGGER_THRESH": chain.TRIGGER_THRESH, "chain": chain, "np": np}
    print("\nargparse defaults (the 'no flags' run must mean the same chain):")
    bad_cfg = []
    for flag in FLAGS:
        vals, ok = [], True
        for side in ("sim", "node"):
            expr = defaults[side].get(flag)
            if expr is None:
                ok = False
                vals.append("<missing>")
                continue
            try:
                vals.append(repr(eval(expr, dict(ev_env))))
            except Exception as exc:  # noqa: BLE001 - report, do not crash
                ok = False
                vals.append(f"<uneval {exc}>")
        ok = ok and vals[0] == vals[1]
        if not ok:
            bad_cfg.append(flag)
        print(f"  {flag:18s} sim={vals[0]:>16s}  node={vals[1]:>16s}  {'OK' if ok else 'DIFFERS'}")

    # ------------------------------------------------------------ one ticker
    ticker = chain.IncrementalTeleopTicker(
        position_scale=1.0,
        rotation_weight=0.5,
        target_slew_m_s=1.0,
        rot_err_clip_rad=0.08,
        input_tau=0.12,
        trigger_thresh=chain.TRIGGER_THRESH,
        stale_s=0.5,
        waist_sign=1.0,
        waist_clamp=0.6,
        waist_compensate=False,
        torso_keepout=chain.TORSO_KEEPOUT,
    )
    # The arm IK backend does NOT enter this comparison (both assemblies read
    # the same st), but which solver produced st decides what the numbers
    # above are evidence ABOUT -- so name it, and refuse to let the default
    # drift back to dls unnoticed.
    resolved = ticker.arms["right"].backend
    env_backend = os.environ.get("X2_IK_BACKEND")
    print(f"\narm IK backend: {resolved}  (X2_IK_BACKEND={env_backend or '<unset>'})")
    if env_backend is None and importlib.util.find_spec("mink") is not None:
        if resolved != "mink":
            print(
                "FAIL: mink is importable but the chain resolved to dls -- the "
                "default moved back, so every published A/B number describes the "
                "other solver"
            )
            return 1
    elif resolved == "dls":
        print("     (dls explicitly requested: this run says nothing about the mink default)")

    core = node_mod.TeleopCore(
        argparse.Namespace(
            intent_port=args.intent_port,
            stale_s=0.5,
            position_scale=1.0,
            rotation_weight=0.5,
            rot_err_clip=0.08,
            input_tau=0.12,
            target_slew=1.0,
            trigger_thresh=chain.TRIGGER_THRESH,
            waist_clamp=0.6,
            waist_sign=1.0,
            waist_compensate=False,
            record="",
        )
    )
    if hasattr(core.rx, "start"):
        core.rx.start()

    # ------------------------------------------------------------ sim side
    def tgt_init() -> dict:
        tgt = {}
        for i, n in enumerate(ubt.ARM_NAMES):
            tgt[n] = (0.0, ubt.ARM_KP[i], ubt.ARM_KD[i])
        for w in ubt.WAIST:
            tgt[w[0]] = (w[3], w[4], w[5])
        return tgt

    def sim_assembly(st, tgt):
        """incremental_upper_teleop_mujoco.py's per-tick assembly, verbatim."""
        for side in ("left", "right"):
            q = st["q"][side]
            base = 0 if side == "left" else 7
            for i, n in enumerate(ubt.ARM_NAMES[base : base + 7]):
                tgt[n] = (q[i], *tgt[n][1:])
        tgt["waist_yaw_joint"] = (st["waist"], ubt.WAIST[0][4], ubt.WAIST[0][5])
        return tgt

    def node_assembly(st):
        """Unpack TeleopCore.commands() into {joint: (pos, kp, kd)}."""
        cmds = core.commands(st)
        out = {}
        for name, pos, (kp, kd) in zip(*cmds["arm"]):
            out[name] = (float(pos), kp, kd)
        names, poss, gains = cmds["waist"]
        for n, p, (kp, kd) in zip(names, poss, gains):
            out[n] = (float(p), kp, kd)
        return out

    # ------------------------------------------------------------ drive tape
    tape = np.load(args.tape)
    n = len(tape["t"])
    tgt = tgt_init()
    worst, first_bad, n_joints = 0.0, None, 0
    held_ticks = {"left": 0, "right": 0}
    t0 = 1000.0
    for k in range(n):
        frame = {
            "t": t0 + k * chain.DT,
            "trigger": tape["trigger"][k],
            "grip": tape["grip"][k],
            "lctrl_pose": tape["lctrl_pose"][k].astype(np.float64),
            "rctrl_pose": tape["rctrl_pose"][k].astype(np.float64),
            "smpl_joints": tape["smpl_joints"][k].astype(np.float64),
        }
        st = ticker.step(frame, now=frame["t"])
        for side in held_ticks:
            held_ticks[side] += int(bool(st["held"][side]))
        a = sim_assembly(st, dict(tgt))
        b = node_assembly(st)
        if set(a) != set(b):
            print(
                f"\nKEY MISMATCH at tick {k}: sim-only {set(a) - set(b)} "
                f"node-only {set(b) - set(a)}"
            )
            core.stop()
            return 1
        n_joints = len(a)
        for name in a:
            pa, kpa, kda = a[name]
            pb, kpb, kdb = b[name]
            dmax = max(
                abs(float(pa) - float(pb)),
                abs(float(kpa) - float(kpb)),
                abs(float(kda) - float(kdb)),
            )
            worst = max(worst, dmax)
            if dmax != 0.0 and first_bad is None:
                first_bad = (k, name, pa, pb, kpa, kpb, kda, kdb)
        tgt = a

    print(f"\ndrove {n} tape frames x {n_joints} joints (14 arm + waist yaw):")
    print(f"  max |sim - node| over positions AND gains: {worst:.3e}")
    if first_bad:
        k, name, pa, pb, kpa, kpb, kda, kdb = first_bad
        print(
            f"  first mismatch tick {k} joint {name}: pos {pa!r} vs {pb!r} "
            f"gains ({kpa},{kda}) vs ({kpb},{kdb})"
        )
    print(f"  argparse defaults differ on: {bad_cfg if bad_cfg else 'nothing'}")
    print(f"  held ticks (clutch closed): L={held_ticks['left']} R={held_ticks['right']}")
    vacuous = held_ticks["left"] == 0 or held_ticks["right"] == 0
    if vacuous:
        print(
            "  FAIL: a side never engages, so its command stream is a frozen "
            "constant -- regenerate the tape for the current trigger/grip roles."
        )
    ok = (worst == 0.0) and not bad_cfg and not vacuous
    print("\nLEVEL-0:", "PASS (bit-exact)" if ok else "FAIL")
    core.stop()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
