#!/usr/bin/env bash
# v4 SIM preset: trackerless 3-point teleop against the MuJoCo harness.
#
#   ./gear_sonic/scripts/run_pico_teleop_v4_sim.sh
#   ./gear_sonic/scripts/run_pico_teleop_v4_sim.sh --no-viewer --record /tmp/v4_sim.npz
#   ./gear_sonic/scripts/run_pico_teleop_v4_sim.sh --headset-ip ${HEADSET_IP}
#
# This is the sim twin of run_pico_teleop_v4_real.sh: it owns the v4-3point
# preset pieces that the operator must NOT have to remember (frame signs,
# wrist mapping, lever) and starts both halves in one terminal -- the MuJoCo
# harness in the background, the laptop sender in the foreground. Ctrl-C
# stops the sender and the trap kills the harness.
#
# WHY THE FRAME SIGNS ARE BAKED IN (2026-10-30 live-corrected; the
# 2026-10-27 round had the wire x backwards)
# --------------------------------------------------------------------
# The v4 3-point wire freezes an operator frame from the two controllers:
#     M = [fwd, lat, up],  lat = p_L - p_R,  fwd = cross(lat, up)
# The XRoboToolkit PC service delivers a RIGHT-handed world (X right, Y up,
# Z in; its README notes the real data flips Unity's +Z).  On that data
# lat = -X and fwd = cross(-X, +Y) = -Z = the operator's FORWARD, i.e.
# wire = (+d_F, d_L, d_U).  The wire is the operator's own
# (forward, left, up) frame, so the same-facing avatar map is the IDENTITY:
#   * wire +x is operator-forward -> harness keeps --operator-x-sign +1
#   * wire +y is operator-left    -> harness keeps --operator-y-sign +1
# The operator and the robot face the SAME direction (2026-10-30 live
# correction: NOT face-to-face), so the operator's front-left reaches the
# robot's own front-left.  A -1 would mirror an axis, which is only right
# for a face-to-face rig; the 2026-10-27 x=-1 round trusted the
# left-handed Unity figure and made the real robot reach BACKWARD.
# Both flips (if ever needed) happen AFTER the ctrl-lever compensation
# inside the shared chain (W = C - R_c r_c); flipping the raw wire first
# would turn wrist rotation into up to 2|r| ~ 20 cm of fake translation.
#
# The harness DEFAULTS stay +1 on purpose: the full-body `_torso_M` wire
# feeds the same harness and is already in the robot frame.  This wrapper
# passes +1/+1 explicitly so the preset documents the avatar contract, and
# the PC2 node's own defaults are the same +1/+1 (see
# x2_pc2/v3_upper_teleop/README.md section 4.4), so sim and robot agree.
#
# The ctrl-lever values are the 2026-10 fits used by the documented v4 sim
# command and by the real-node command; override them here if the grip or
# the controller mounting changes.
#
# WRIST MAPPING: the harness/chain defaults are still 0,1,2 and
# (yaw,pitch,roll)=(+1,+1,+1) because the full-body wire feeds the same code.
# The PC2 v4 node defaults - measured on the real robot on 2026-10-27 - are
# axis order 0,2,1 (controller x->pitch, y->roll; 0,1,2 crossed pitch/roll
# from the elbow-forward neutral) and pitch sign -1 on both wrists (yaw and
# roll stay +1).  Pass them here so the sim mirrors the robot instead of
# making the operator rediscover the same cross-axis bug in MuJoCo.
#
# Requirements: the laptop .venv (mujoco/mink), a Pico headset + two
# controllers, and the XRoboToolkit PC Service (run_pico_teleop.sh starts
# it). No robot, no PC2, no docker stack.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"

PORT=15573
L_LEVER=(0.0410 0.0090 -0.0912)
R_LEVER=(-0.0393 0.0115 -0.0791)
HARNESS_EXTRA=()
SENDER_ARGS=()

usage() {
    cat <<'EOF'
Usage: run_pico_teleop_v4_sim.sh [options...] [sender options...]

Starts the MuJoCo harness (background, :15573) and the 3-point sender
(foreground). Preset = --tracking 3point --no-locomotion --hand-source off;
the two frame signs and the wrist axis order/pitch signs are fixed here to
match the PC2 v4 node defaults.

Options handled here:
  --intent-port <n>        harness/sender intent port (default 15573)
  --no-viewer              harness without the MuJoCo viewer (headless)
  --record <file.npz>      harness per-tick telemetry dump
  --duration <s>           harness stop after N s (sender keeps running)
  --ctrl-lever-left  X Y Z   override the left  controller->wrist lever
  --ctrl-lever-right X Y Z   override the right controller->wrist lever
  -h, --help               this text

Everything else is passed to run_pico_teleop.sh (--headset-ip, --cam,
--record-dir, --wrist-source, ...). --tracking is owned here; --pc2-host
and --hand-source are refused (this is the sim; real robot =
run_pico_teleop_v4_real.sh).
EOF
}

_need_3_floats() {
    [[ $# -eq 3 ]] || { echo "[v4-sim] ERROR: $1 needs 3 numbers (X Y Z)." >&2; exit 2; }
}

while (( $# )); do
    case "$1" in
        --intent-port)          PORT="${2:-}"; shift 2 ;;
        --intent-port=*)        PORT="${1#*=}"; shift ;;
        --no-viewer)            HARNESS_EXTRA+=(--no-viewer); shift ;;
        --record)               HARNESS_EXTRA+=(--record "${2:-}"); shift 2 ;;
        --record=*)             HARNESS_EXTRA+=(--record "${1#*=}"); shift ;;
        --duration)             HARNESS_EXTRA+=(--duration "${2:-}"); shift 2 ;;
        --duration=*)           HARNESS_EXTRA+=(--duration "${1#*=}"); shift ;;
        --ctrl-lever-left)
            _need_3_floats "$1" "${2:-}" "${3:-}" "${4:-}"
            L_LEVER=("$2" "$3" "$4"); shift 4 ;;
        --ctrl-lever-right)
            _need_3_floats "$1" "${2:-}" "${3:-}" "${4:-}"
            R_LEVER=("$2" "$3" "$4"); shift 4 ;;
        --tracking|--tracking=*)
            echo "[v4-sim] ERROR: this launcher already sets --tracking 3point; drop '$1'." >&2
            exit 2 ;;
        --pc2-host|--pc2-host=*)
            echo "[v4-sim] ERROR: a --pc2-host target means the real robot; use run_pico_teleop_v4_real.sh." >&2
            exit 2 ;;
        --hand-source|--hand-source=*)
            echo "[v4-sim] ERROR: the sim preset owns --hand-source off (no CLAW bridge on 15573)." >&2
            exit 2 ;;
        -h|--help)              usage; exit 0 ;;
        *)                      SENDER_ARGS+=("$1"); shift ;;
    esac
done

[[ "$PORT" =~ ^[0-9]+$ ]] || { echo "[v4-sim] ERROR: bad --intent-port '$PORT'." >&2; exit 2; }
[[ -x "$PYTHON" ]] || { echo "[v4-sim] ERROR: python not executable: $PYTHON (set PYTHON=...)" >&2; exit 2; }

HARNESS_CMD=(
    "$PYTHON" "$REPO_ROOT/gear_sonic/scripts/arm_teleop_v3_mujoco.py"
    --wire --intent-port "$PORT"
    --operator-x-sign +1
    --operator-y-sign +1
    --wrist-axis-order-l 0,2,1
    --wrist-axis-order-r 0,2,1
    --wrist-pitch-sign-l -1
    --wrist-pitch-sign-r -1
    --ctrl-lever-left "${L_LEVER[@]}"
    --ctrl-lever-right "${R_LEVER[@]}"
    "${HARNESS_EXTRA[@]}"
)
SENDER_CMD=(
    "$REPO_ROOT/gear_sonic/scripts/run_pico_teleop.sh"
    --pc2-host 127.0.0.1
    --intent-port "$PORT"
    --tracking 3point
    --no-locomotion
    --hand-source off
)

printf '[v4-sim] harness:'
printf ' %q' "${HARNESS_CMD[@]}"
echo
printf '[v4-sim] sender :'
printf ' %q' "${SENDER_CMD[@]}" "${SENDER_ARGS[@]}"
echo
echo "[v4-sim] frame: operator_x_sign=+1 operator_y_sign=+1 (same-facing avatar map, identity); wrist order 0,2,1 pitch sign -1 (matches the PC2 node)"

if [[ -n "${V4_SIM_DRY_RUN:-}" ]]; then
    exit 0
fi

"${HARNESS_CMD[@]}" &
HARNESS_PID=$!
_harness_down() {
    kill "$HARNESS_PID" 2>/dev/null || true
    wait "$HARNESS_PID" 2>/dev/null || true
}
trap _harness_down EXIT INT TERM

# The harness binds the ZMQ receiver before the sender connects; give it a
# moment so the sender does not race the port.
for _ in $(seq 1 100); do
    if ! kill -0 "$HARNESS_PID" 2>/dev/null; then
        echo "[v4-sim] ERROR: the MuJoCo harness exited during startup." >&2
        exit 1
    fi
    if ss -tln 2>/dev/null | grep -q ":$PORT "; then
        break
    fi
    sleep 0.1
done
if ! ss -tln 2>/dev/null | grep -q ":$PORT "; then
    echo "[v4-sim] ERROR: harness never listened on :$PORT." >&2
    exit 1
fi

"${SENDER_CMD[@]}" "${SENDER_ARGS[@]}"
