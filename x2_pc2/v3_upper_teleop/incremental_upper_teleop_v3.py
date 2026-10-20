#!/usr/bin/env python3
"""Incremental (clutch-IK) upper-body Pico teleop -- v3 robot node.

Consumes the SAME wire as the v2 node (:5573 "pico_intent", packed binary,
50 Hz) but drives the SHARED v3 tick body from
``incremental_chain_v3.py`` -- the exact object the MuJoCo harness
(``gear_sonic/scripts/arm_teleop_v3_mujoco.py``) runs:

    * arm IK = 4-DoF position solve over shoulder-3 + elbow, with the three
      wrist joints LOCKED at the WristClutch command and re-fed into the FK
      every tick; ``--full-pose-ik`` opts into the 7-DoF pose solve instead.
    * the wrist is filled directly from the anchored controller-quaternion
      differential by solving the X2 wrist chain exactly:
      ``W(q_tgt) = W(q_wa) @ C delta C^T`` with ``W = Rz(yaw)Ry(pitch)Rx(roll)``
      (slew + near-pi guard + joint limits), NOT solved by the arm IK.
      The wrist is yaw-proximal with pitch/roll crossing at one point, so a
      tool pitch/yaw rotation legitimately coordinates pitch+roll (or all
      three) off neutral; tool roll stays a pure roll-joint motion.
      ``--wrist-axis-order-l/-r`` and the per-axis signs define C (controller
      -> wrist axis correspondence).  This is the v3 fix for "rotate only
      the wrist, the robot drags its elbow/shoulder".
    * SMPL is CLOSED: the waist holds a fixed ``--waist-stand``.
    * 2026-10 role swap: the GRIP is the per-arm clutch; the TRIGGER drives
      the (internal) hand channel.  This node does not publish a hand -- the
      trigger therefore has no robot-side effect here.  A stale wire HOLDS
      the arms (clutch semantics), like v2.

Robot-side differences from the sim harness: DLS backend (PC2 has no
mink/mujoco), ROS 2 Humble hal publishers, and an 85-column record with the
plant qpos columns (70:85) left NaN.  Everything else -- receiver, ticker,
guards, record layout, self-test checks -- is the shared code, so a sim run
and a robot run read the same way offline.

Deploy layout (flat, next to this file -- see x2_pc2/v3_upper_teleop/README.md):

    incremental_upper_teleop_v3.py   # this node
    incremental_chain_v3base.py      # shared primitives (repo incremental_chain.py,
                                     #  renamed so v2's flat copy stays intact)
    incremental_chain_v3.py          # v3 ticker / wrist clutch
    x2_arm_ik.py  x2_arm_fk.py       # DLS solver + X2 FK
    upper_body_teleop.py             # v1 hal tables (names/limits/gains)

Run on PC2 (paths per czq_test_v2_incremental.md §3):

    source /opt/ros/humble/setup.bash
    source /home/agi/aimdk/install/setup.bash
    source ~/Workspaces/czq/ros2_ws/install/setup.bash
    ros2 run czq_examples incremental_upper_teleop_v3 --self-test --backend dls
    ros2 run czq_examples incremental_upper_teleop_v3 --dry-run --record /tmp/v3_dry.npz --duration 20
    ros2 run czq_examples incremental_upper_teleop_v3 \
        --ctrl-lever-left  0.0410 0.0090 -0.0912 \
        --ctrl-lever-right -0.0393 0.0115 -0.0791 \
        --record /tmp/v3_live.npz

Without a colcon entry point, run the flat file directly:

    cd ~/Workspaces/czq/ros2_ws/src/czq_examples/czq_examples
    python3 incremental_upper_teleop_v3.py --self-test --backend dls
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np

# Own directory first: on PC2 the shared chain / solver / hal tables are flat
# copies next to this file (czq_examples/), exactly like the sim's sibling
# layout. The chain itself then picks its own flat vs repo imports.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import upper_body_teleop as ubt  # noqa: E402  # hal tables (names/limits/gains)

try:  # v3 bundle: base chain under a v3-only name, v2's copy left untouched
    import incremental_chain_v3base as chain  # type: ignore[no-redef]
except ImportError:  # pragma: no cover - repo tree (laptop), or older bundle
    try:
        import incremental_chain as chain  # type: ignore[no-redef]
    except ImportError:
        from gear_sonic.utils.teleop import incremental_chain as chain  # type: ignore[no-redef]

try:  # flat vendored copy (PC2 czq_examples/)
    import incremental_chain_v3 as chain_v3  # type: ignore[no-redef]
except ImportError:  # pragma: no cover - repo tree (laptop)
    from gear_sonic.utils.teleop import incremental_chain_v3 as chain_v3  # type: ignore[no-redef]

DT = chain.DT
# The sim harness's --record layout is FIXED at 85 columns; offline autopsy
# tools index it positionally, so the robot writes the identical row with the
# plant qpos columns NaN (there is no plant here).
REC_COLS = 85
REC_JOINTS = list(ubt.ARM_NAMES) + ["waist_yaw_joint"]


# ---------------------------------------------------------------------------
# Record row -- ONE pure function so --self-test can pin its width/offsets
# without a receiver, a plant, or ROS.
# ---------------------------------------------------------------------------
def _record_row(st, arms, frame, t_wall: float) -> np.ndarray:
    """One 85-column telemetry row, exactly the sim harness's layout.

    0 t_wall | 1 fresh | 2 lt | 3 rt (raw trigger = hand input)
    4 grip_l | 5 grip_r (LPF'd grip = clutch) | 6 frame_t
    7:14 lcp_raw | 14:21 rcp_raw | 21:28 lcp_filt | 28:35 rcp_filt
    35:42 q_left_cmd | 42:49 q_right_cmd
    49:52 wrist_tgt_L | 52:55 wrist_tgt_R
    55:58 wrist_delta_ypr_L | 58:61 wrist_delta_ypr_R
    61 ik_pos_err_L | 62 ik_pos_err_R
    63 hold_L | 64 hold_R | 65 eng_L | 66 eng_R
    67 wrist_dropped_L | 68 wrist_dropped_R | 69 waist (const stand)
    70:85 qpos arms(14)+waist(1) -- NaN on the robot
    """
    return np.concatenate(
        [
            [
                t_wall,
                float(st["fresh"]),
                st["lt"],
                st["rt"],
                st["grip"]["l"],
                st["grip"]["r"],
            ],
            [float(frame["t"]) if frame is not None else np.nan],
            np.asarray(st["ctrl_pre"]["left"], np.float64).reshape(7),
            np.asarray(st["ctrl_pre"]["right"], np.float64).reshape(7),
            np.asarray(st["ctrl"]["left"], np.float64).reshape(7),
            np.asarray(st["ctrl"]["right"], np.float64).reshape(7),
            np.asarray(st["q"]["left"], np.float64).reshape(7),
            np.asarray(st["q"]["right"], np.float64).reshape(7),
            np.asarray(st["wrist_tgt"]["left"], np.float64).reshape(3),
            np.asarray(st["wrist_tgt"]["right"], np.float64).reshape(3),
            np.asarray(st["wrist_delta_ypr"]["left"], np.float64).reshape(3),
            np.asarray(st["wrist_delta_ypr"]["right"], np.float64).reshape(3),
            [
                arms["left"].last_ik.pos_err_m,
                arms["right"].last_ik.pos_err_m,
            ],
            [
                float(arms["left"].holding),
                float(arms["right"].holding),
                float(arms["left"].engaged),
                float(arms["right"].engaged),
            ],
            [
                float(st["wrist_dropped"]["left"]),
                float(st["wrist_dropped"]["right"]),
                float(st["waist"]),
            ],
            [np.nan] * 15,
        ]
    )


def _parse_axis_order(text: str) -> tuple[int, int, int]:
    """Parse '0,1,2' / '0 1 2' into the wrist-axis correspondence.

    Wrist joints are [yaw, pitch, roll]; the controller differential's
    Euler ZYX components are [about z, about y, about x]. The default
    (0,1,2) means controller z -> yaw, y -> pitch, x -> roll. A
    permutation such as 2,1,0 swaps the controller x/z assignment. It is
    applied to the controller differential BEFORE the exact wrist-chain
    solve, together with the per-axis signs.
    """
    try:
        return chain_v3.parse_wrist_axis_order(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _axis_order_default(env_name: str) -> tuple[int, int, int]:
    try:
        return _parse_axis_order(os.environ.get(env_name, "0,1,2"))
    except argparse.ArgumentTypeError as exc:
        raise SystemExit(f"{env_name}: {exc}") from exc


# ---------------------------------------------------------------------------
# Plant-free per-tick core -- the ONE thing the live node and --dry-run share
# ---------------------------------------------------------------------------
class TeleopCore:
    """Receiver + shared v3 ticker + telemetry. No ROS, no plant.

    tick() is called at 50 Hz by the ROS timer (live) or by a plain pacing
    loop (--dry-run); both get the identical command stream out of the same
    IncrementalTeleopTickerV3 the MuJoCo harness runs.
    """

    def __init__(self, args) -> None:
        wrist_signs = {
            "left": (
                args.wrist_yaw_sign_l,
                args.wrist_pitch_sign_l,
                args.wrist_roll_sign_l,
            ),
            "right": (
                args.wrist_yaw_sign_r,
                args.wrist_pitch_sign_r,
                args.wrist_roll_sign_r,
            ),
        }
        wrist_axis_order = {
            "left": tuple(args.wrist_axis_order_l),
            "right": tuple(args.wrist_axis_order_r),
        }
        ctrl_levers = {
            side: (
                getattr(args, f"ctrl_lever_{side}")
                if getattr(args, f"ctrl_lever_{side}") is not None
                else args.ctrl_lever
            )
            for side in ("left", "right")
        }
        self.ticker = chain_v3.IncrementalTeleopTickerV3(
            position_scale=args.position_scale,
            target_slew_m_s=args.target_slew,
            input_tau=args.input_tau,
            trigger_thresh=args.trigger_thresh,
            stale_s=args.stale_s,
            stale_grip_close=args.stale_grip_close,
            wrist_scale=args.wrist_scale,
            wrist_signs=wrist_signs,
            wrist_axis_order=wrist_axis_order,
            wrist_slew_rad_s=args.wrist_slew,
            wrist_pi_guard_rad=args.wrist_pi_guard,
            waist_stand=args.waist_stand,
            full_pose_ik=args.full_pose_ik,
            backend=args.backend,
            ctrl_lever=ctrl_levers,
            position_deadband_m=args.position_deadband_mm * 1e-3,
            posture_gain=args.posture_gain,
        )
        self.rx = chain.IncrementalIntentReceiver(args.intent_port)
        self.rec_rows = None if not args.record else []
        self.n_ticks = 0
        self.last_flags = None

    def tick(self, now: float | None = None):
        """One 50 Hz tick. -> (state dict, frame, nerr)."""
        frame, nerr = self.rx.snapshot()
        st = self.ticker.step(frame, now=now)
        self.n_ticks += 1
        if self.rec_rows is not None:
            self.rec_rows.append(_record_row(st, self.ticker.arms, frame, time.monotonic()))
        return st, frame, nerr

    def commands(self, st) -> dict:
        """state -> {hal group: (names, positions, gains)} for the groups
        this level publishes. Arms carry the chain's q (already clipped to
        the joint limits by the chain itself); waist yaw is the fixed v3
        stand value clipped to the hal table; waist pitch/roll + head + legs
        are shape holds (stand capture values, same as v1/v2)."""
        yaw_lo, yaw_hi = ubt.WAIST[0][1], ubt.WAIST[0][2]
        waist = [
            float(np.clip(st["waist"], yaw_lo, yaw_hi)),
            ubt.WAIST[1][3],
            ubt.WAIST[2][3],
        ]
        out = {
            "arm": (
                list(ubt.ARM_NAMES),
                np.concatenate([st["q"]["left"], st["q"]["right"]]),
                list(zip(ubt.ARM_KP, ubt.ARM_KD)),
            ),
            "waist": ([w[0] for w in ubt.WAIST], waist, [(w[4], w[5]) for w in ubt.WAIST]),
            "head": (
                [h[0] for h in ubt.HEAD],
                [h[3] for h in ubt.HEAD],
                [(h[4], h[5]) for h in ubt.HEAD],
            ),
            "leg": (
                [leg[0] for leg in ubt.LEG],
                [leg[1] for leg in ubt.LEG],
                [(leg[2], leg[3]) for leg in ubt.LEG],
            ),
        }
        return out

    def status(self, st, frame, nerr) -> str:
        """One-line human status; ``arrived`` flips on wire/engage changes."""
        larm, rarm = self.ticker.arms["left"], self.ticker.arms["right"]
        flags = (
            bool(st["fresh"]),
            bool(st["held"]["left"]),
            bool(st["held"]["right"]),
            larm.holding,
            rarm.holding,
        )
        arrived = "*" if flags != self.last_flags else " "
        self.last_flags = flags
        age = np.nan if frame is None else time.monotonic() - frame["t"]
        hold = f"{'H' if larm.holding else '-'}{'H' if rarm.holding else '-'}"
        eng = f"{'E' if larm.engaged else '-'}{'E' if rarm.engaged else '-'}"
        wl = np.degrees(st["wrist_delta_ypr"]["left"])
        wr = np.degrees(st["wrist_delta_ypr"]["right"])
        dropped = int(st["wrist_dropped"]["left"] + st["wrist_dropped"]["right"])
        return (
            f"{arrived} fresh={int(st['fresh'])} t={time.strftime('%H:%M:%S')} "
            f"age={age:5.3f} trig={st['lt']:.2f}/{st['rt']:.2f} "
            f"grip={st['grip']['l']:.2f}/{st['grip']['r']:.2f} "
            f"eng={eng} hold={hold} "
            f"wristL={wl[0]:+5.1f},{wl[1]:+4.1f},{wl[2]:+5.1f}deg "
            f"wristR={wr[0]:+5.1f},{wr[1]:+4.1f},{wr[2]:+5.1f}deg "
            f"ikerr={larm.last_ik.pos_err_m:.3f}/{rarm.last_ik.pos_err_m:.3f} "
            f"wdrop={dropped} err={nerr}"
        )

    def stop(self) -> None:
        self.rx.stop()

    def dump_record(self, path: str) -> None:
        if self.rec_rows is None:
            return
        arr = np.asarray(self.rec_rows, np.float64) if self.rec_rows else np.zeros((0, REC_COLS))
        np.savez_compressed(path, rows=arr, joints=np.array(REC_JOINTS))
        print(f"[node] recorded {len(self.rec_rows)} ticks -> {path}")


# ---------------------------------------------------------------------------
# Live ROS2 node (50 Hz hal publisher) -- rclpy imported only here
# ---------------------------------------------------------------------------
def run_live(args) -> None:
    from aimdk_msgs.msg import JointCommand, JointCommandArray
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

    pub_qos = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
    )

    class IncrementalUpperTeleopV3Node(Node):
        def __init__(self) -> None:
            super().__init__("incremental_upper_teleop_v3")
            self.core = TeleopCore(args)
            self.core.rx.start()
            self.pubs = {
                g: self.create_publisher(JointCommandArray, f"/aima/hal/joint/{g}/command", pub_qos)
                for g in ("arm", "waist", "head", "leg")
            }
            self.create_timer(DT, self.tick)  # 50 Hz
            self.n_stat = 0
            self.t_stat = time.monotonic()
            arms = self.core.ticker.arms
            w = self.core.ticker
            self.get_logger().info(
                f"incremental_upper_teleop_v3 up: SUB :{args.intent_port}; "
                f"GRIP (LPF'd) > {args.trigger_thresh} clutches per arm "
                f"(TRIGGER drives the internal hand; not published); "
                f"arm IK {arms['right'].backend}/"
                f"{'7-DoF' if arms['right'].full_pose_ik else '4-DoF'}; "
                f"scale {args.position_scale}, lpf tau {args.input_tau}s, "
                f"wrist scale {args.wrist_scale}, "
                f"ctrl lever L{tuple(np.round(w.ctrl_levers['left'], 4))} "
                f"R{tuple(np.round(w.ctrl_levers['right'], 4))}m, "
                f"deadband {w.position_deadband_m * 1e3:.1f}mm, "
                f"posture {w.posture_gain:g}, waist fixed {args.waist_stand}rad, "
                f"stale->arms HOLD, stale hand->"
                f"{'HOLD' if args.stale_grip_close else 'OPEN'}"
                + ("" if args.legs else "; NOT publishing leg/head hold")
            )

        def _publish(self, group, names, positions, gains) -> None:
            cmd = JointCommandArray()
            for name, pos, (kp, kd) in zip(names, positions, gains):
                j = JointCommand()
                j.name = name
                j.position = float(pos)
                j.velocity = 0.0
                j.effort = 0.0
                j.stiffness = float(kp)
                j.damping = float(kd)
                cmd.joints.append(j)
            self.pubs[group].publish(cmd)

        def tick(self) -> None:
            st, frame, nerr = self.core.tick()
            if not args.dry_run:
                cmds = self.core.commands(st)
                self._publish("arm", *cmds["arm"])
                self._publish("waist", *cmds["waist"])
                if args.legs:
                    self._publish("head", *cmds["head"])
                    self._publish("leg", *cmds["leg"])
            # status every 2 s, plus every flag change (engage/release/stale)
            line = self.core.status(st, frame, nerr)
            self.n_stat += 1
            if line.startswith("*") or self.n_stat >= 100:
                self.n_stat = 0
                self.get_logger().info(line.strip())

    rclpy.init()
    node = IncrementalUpperTeleopV3Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.core.stop()
        node.core.dump_record(args.record)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


# ---------------------------------------------------------------------------
# Dry-run: the same 50 Hz tick chain, printed, no rclpy at all
# ---------------------------------------------------------------------------
def run_dry(args) -> None:
    core = TeleopCore(args)
    core.rx.start()
    arms = core.ticker.arms
    print(
        f"[dry-run] SUB :{args.intent_port}; GRIP (LPF'd) > {args.trigger_thresh} "
        f"clutches per arm (TRIGGER drives the internal hand; not published); "
        f"arm IK {arms['right'].backend}/"
        f"{'7-DoF' if arms['right'].full_pose_ik else '4-DoF'}; "
        f"scale {args.position_scale}, lpf tau {args.input_tau}s, "
        f"wrist scale {args.wrist_scale}, "
        f"ctrl lever L{tuple(np.round(core.ticker.ctrl_levers['left'], 4))} "
        f"R{tuple(np.round(core.ticker.ctrl_levers['right'], 4))}m, "
        f"deadband {core.ticker.position_deadband_m * 1e3:.1f}mm, "
        f"posture {core.ticker.posture_gain:g}, waist fixed {args.waist_stand}rad; "
        f"Ctrl-C to stop"
    )
    t0 = time.monotonic()
    next_tick = time.monotonic()
    n_stat = 0
    try:
        while True:
            if args.duration > 0 and time.monotonic() - t0 > args.duration:
                break
            st, frame, nerr = core.tick()
            n_stat += 1
            if n_stat >= 25:  # 0.5 s
                n_stat = 0
                print(core.status(st, frame, nerr))
            # same wall-clock pacing as the sim/live loop: no drift debt
            next_tick += DT
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        core.stop()
        core.dump_record(args.record)


# ---------------------------------------------------------------------------
# Self-test: the shared v3 chain + THIS node's hal wiring (no zmq, no ROS)
# ---------------------------------------------------------------------------
def _record_layout_ok() -> bool:
    """A real (stale, disengaged) tick -> an 85-column row with the plant
    qpos columns NaN. Pins the layout the sim autopsy tools assume."""
    try:
        tick = chain_v3.IncrementalTeleopTickerV3(backend=os.environ.get("X2_IK_BACKEND", "dls"))
        frame = {
            "trigger": np.zeros(2),
            "grip": np.zeros(2),
            "lctrl_pose": np.zeros(7),
            "rctrl_pose": np.zeros(7),
            "t": time.monotonic(),
        }
        st = tick.step(frame, now=frame["t"])
        row = _record_row(st, tick.arms, frame, time.monotonic())
        return (
            row.shape == (REC_COLS,)
            and bool(np.isfinite(row[:70]).all())
            and bool(np.isnan(row[70:]).all())
        )
    except Exception as exc:  # noqa: BLE001 - report, do not abort the suite
        print(f"[self-test] record layout probe failed: {type(exc).__name__}: {exc}")
        return False


def run_self_test(args) -> bool:
    # The v3 chain's own checks construct tickers WITHOUT a backend argument
    # (they are shared with the laptop/mink path), so select the robot's
    # backend through the same env switch the chain documents.
    os.environ["X2_IK_BACKEND"] = args.backend
    print(f"[self-test] v3 chain backend: {args.backend}")
    checks = [("v3 chain self-test", chain_v3.run_self_test())]
    checks.append(
        (
            "hal arm limits == chain solver limits",
            bool(
                np.array_equal(ubt.ARM_LO, chain.ARM_LO_SOLVER)
                and np.array_equal(ubt.ARM_HI, chain.ARM_HI_SOLVER)
            ),
        )
    )
    checks.append(
        (
            "hal arm order = 7 left then 7 right",
            len(ubt.ARM_NAMES) == 14
            and all(n.startswith("left") for n in ubt.ARM_NAMES[:7])
            and all(n.startswith("right") for n in ubt.ARM_NAMES[7:]),
        )
    )
    neutral = np.concatenate([chain.ARM_NEUTRAL["left"], chain.ARM_NEUTRAL["right"]])
    checks.append(
        (
            "neutral arm pose inside hal limits",
            bool(np.all(neutral >= ubt.ARM_LO) and np.all(neutral <= ubt.ARM_HI)),
        )
    )
    checks.append(
        (
            "waist stand inside hal yaw limits",
            bool(ubt.WAIST[0][1] <= args.waist_stand <= ubt.WAIST[0][2]),
        )
    )
    checks.append(
        (
            "gains complete for every published joint",
            len(ubt.ARM_KP) == len(ubt.ARM_KD) == 14
            and all(kp > 0 and kd >= 0 for kp, kd in zip(ubt.ARM_KP, ubt.ARM_KD))
            and all(w[4] > 0 and w[5] >= 0 for w in ubt.WAIST)
            and all(h[4] > 0 and h[5] >= 0 for h in ubt.HEAD)
            and all(leg[2] > 0 and leg[3] >= 0 for leg in ubt.LEG),
        )
    )
    checks.append(("record layout (85 cols, qpos 70:85 NaN)", _record_layout_ok()))

    # live-path wiring: every `self._publish(...)` call must pass the group
    # literal AND unpack the matching 3-tuple from cmds. A wrong arity here
    # only shows at the first live tick -- --dry-run never calls _publish
    # (2026-10-14: exactly this crashed the first robot run).
    def _publish_wiring_ok() -> bool:
        import ast as _ast

        expect = ["self", "group", "names", "positions", "gains"]
        tree = _ast.parse(Path(__file__).read_text())
        pub = next(
            (
                n
                for n in _ast.walk(tree)
                if isinstance(n, _ast.FunctionDef) and n.name == "_publish"
            ),
            None,
        )
        if pub is None or [a.arg for a in pub.args.args] != expect:
            return False
        ncalls = 0
        for n in _ast.walk(tree):
            if not (
                isinstance(n, _ast.Call)
                and isinstance(n.func, _ast.Attribute)
                and n.func.attr == "_publish"
            ):
                continue
            ncalls += 1
            if len(n.args) != 2 or not isinstance(n.args[0], _ast.Constant):
                return False
            star, group = n.args[1], n.args[0].value
            if not isinstance(star, _ast.Starred) or not isinstance(star.value, _ast.Subscript):
                return False
            sub = star.value
            if not (isinstance(sub.value, _ast.Name) and sub.value.id == "cmds"):
                return False
            if getattr(sub.slice, "value", None) != group:
                return False
        return ncalls == 4

    checks.append(("hal publish wiring (group + 3-tuple arity)", _publish_wiring_ok()))

    for name, ok in checks:
        print(f"[self-test]   {'PASS' if ok else 'FAIL'} {name}")
    ok_all = all(ok for _, ok in checks)
    print(
        f"[self-test] {'all' if ok_all else 'FAILED'} checks: "
        f"{sum(ok for _, ok in checks)}/{len(checks)}"
    )
    return ok_all


# ---------------------------------------------------------------------------


def main() -> int:
    signs_env = {
        "left": {
            "yaw": float(
                os.environ.get("WRIST3_YAW_SIGN_L", chain_v3.DEFAULT_WRIST_SIGNS["left"][0])
            ),
            "pitch": float(
                os.environ.get("WRIST3_PITCH_SIGN_L", chain_v3.DEFAULT_WRIST_SIGNS["left"][1])
            ),
            "roll": float(
                os.environ.get("WRIST3_ROLL_SIGN_L", chain_v3.DEFAULT_WRIST_SIGNS["left"][2])
            ),
        },
        "right": {
            "yaw": float(
                os.environ.get("WRIST3_YAW_SIGN_R", chain_v3.DEFAULT_WRIST_SIGNS["right"][0])
            ),
            "pitch": float(
                os.environ.get("WRIST3_PITCH_SIGN_R", chain_v3.DEFAULT_WRIST_SIGNS["right"][1])
            ),
            "roll": float(
                os.environ.get("WRIST3_ROLL_SIGN_R", chain_v3.DEFAULT_WRIST_SIGNS["right"][2])
            ),
        },
    }
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--intent-port", type=int, default=5573)
    ap.add_argument(
        "--stale-s",
        type=float,
        default=0.5,
        help="wire older than this -> clutch drops (arms HOLD, they do not return)",
    )
    ap.add_argument("--position-scale", type=float, default=1.0)
    ap.add_argument(
        "--ctrl-lever",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Pico controller origin -> operator wrist centre, metres in the "
        "controller body frame (W = C - R @ r). Default 0 keeps the raw "
        "controller origin. A pure wrist rotation with a stationary hand "
        "should leave the robot target still; if it sweeps, fit this vector "
        "(docs/x2/PICO_LIVE_TELEOP_PITFALLS.md). Common default for both "
        "hands; --ctrl-lever-left/right override it",
    )
    ap.add_argument(
        "--ctrl-lever-left",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="left-hand override of --ctrl-lever (fit each controller separately)",
    )
    ap.add_argument(
        "--ctrl-lever-right",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="right-hand override of --ctrl-lever (see --ctrl-lever-left)",
    )
    ap.add_argument(
        "--position-deadband-mm",
        type=float,
        default=3.0,
        help="soft radial deadband on the compensated translation (mm); 0 = off",
    )
    ap.add_argument(
        "--posture-gain",
        type=float,
        default=0.01,
        help="null-space posture bias toward the engage configuration; 0 = off",
    )
    ap.add_argument(
        "--input-tau",
        type=float,
        default=0.12,
        help="LPF time constant on the wire controller poses (s); 0 = off",
    )
    ap.add_argument(
        "--target-slew",
        type=float,
        default=1.0,
        help="target slew limit (m/s) after the workspace clamp; 0 = off",
    )
    ap.add_argument(
        "--trigger-thresh",
        type=float,
        default=chain.TRIGGER_THRESH,
        help="historical name: threshold on the LPF'd GRIP (the v3 clutch)",
    )
    ap.add_argument(
        "--full-pose-ik",
        action="store_true",
        help="opt into the 7-DoF pose IK (wrist solved; position and rotation "
        "gains both 1.0) instead of the default 4-DoF position solve + "
        "WristClutch direct wrist fill",
    )
    ap.add_argument(
        "--wrist-scale",
        type=float,
        default=1.0,
        help="gain on the controller differential (scales the rotation angle) "
        "before the exact wrist-chain solve",
    )
    for axis in ("yaw", "pitch", "roll"):
        ap.add_argument(
            f"--wrist-{axis}-sign-l",
            type=float,
            default=signs_env["left"][axis],
            help=f"left wrist {axis} sign (env WRIST3_{axis.upper()}_SIGN_L)",
        )
        ap.add_argument(
            f"--wrist-{axis}-sign-r",
            type=float,
            default=signs_env["right"][axis],
            help=f"right wrist {axis} sign (env WRIST3_{axis.upper()}_SIGN_R)",
        )
    for side, suffix in (("left", "l"), ("right", "r")):
        ap.add_argument(
            f"--wrist-axis-order-{suffix}",
            type=_parse_axis_order,
            default=_axis_order_default(f"WRIST3_AXIS_ORDER_{suffix.upper()}"),
            help=(
                f"{side} controller-axis -> wrist-joint correspondence "
                f"(env WRIST3_AXIS_ORDER_{suffix.upper()}); wrist joints are "
                "[yaw,pitch,roll], default 0,1,2 = controller z->yaw, y->pitch, "
                "x->roll. Use e.g. 2,1,0 if a single anatomical DOF lands on the "
                "wrong wrist joint. Applied to the differential before the exact "
                "wrist solve; not a per-joint scale"
            ),
        )
    ap.add_argument(
        "--wrist-slew",
        type=float,
        default=6.0,
        help="wrist command slew (rad/s); bounds the freeze-guard jump term",
    )
    ap.add_argument(
        "--wrist-pi-guard",
        type=float,
        default=2.94,
        help="drop differentials past this magnitude (rad); engage anchors "
        "must stay well under the ZYX ambiguity near pi",
    )
    ap.add_argument(
        "--waist-stand",
        type=float,
        default=0.0,
        help="fixed waist yaw (rad) -- the SMPL waist channel is closed in v3",
    )
    ap.add_argument(
        "--stale-grip-close",
        action="store_true",
        help="stale frame HOLDS the last filtered trigger (the internal hand) "
        "instead of opening it; the arms always HOLD on stale",
    )
    ap.add_argument(
        "--backend",
        choices=("dls", "mink"),
        default="dls",
        help="arm IK backend; PC2 has no mink/mujoco, so the robot default is dls",
    )
    ap.add_argument(
        "--no-legs",
        dest="legs",
        action="store_false",
        default=True,
        help="do not publish the leg/head shape holds (arm + waist still go)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="run the same 50 Hz chain, print status, publish nothing",
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="v3 chain self-test + hal-wiring checks, no zmq/ROS, then exit",
    )
    ap.add_argument(
        "--record",
        type=str,
        default="",
        help="dump the sim's 85-column per-tick telemetry to this .npz "
        "(qpos columns 70:85 are NaN -- no plant)",
    )
    ap.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="dry-run only: stop after N s (0 = until Ctrl-C)",
    )
    args = ap.parse_args()

    if args.self_test:
        return 0 if run_self_test(args) else 1
    if args.dry_run:
        run_dry(args)
    else:
        run_live(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
