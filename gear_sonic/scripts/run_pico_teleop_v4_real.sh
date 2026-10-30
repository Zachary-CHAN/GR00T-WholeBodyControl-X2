#!/usr/bin/env bash
# LAPTOP half of the v4 REAL-ROBOT teleop: trackerless 3-point + the v3 chain.
#
#   ./gear_sonic/scripts/run_pico_teleop_v4_real.sh --pc2-host 10.0.1.41
#   ./gear_sonic/scripts/run_pico_teleop_v4_real.sh --pc2-host 10.0.1.41 --no-hands
#
# Preset (sender options are passed through to run_pico_teleop.sh; PC2 NODE
# options such as --ctrl-lever-*, --wrist-*-sign-*, --operator-x-sign and
# --backend belong on the node command and are refused here):
#   --tracking 3point         headset + two controllers only, no body trackers.
#                             The operator frame is calibrated once from the two
#                             controller positions and then frozen; the headset
#                             stays connected but does not gate the stream.
#   --hand-source trigger     v4 keeps the v3 roles: GRIP = arm clutch,
#   --hand-close-on-grip      TRIGGER = CLAW/OmniHand (press = close).
#                             3point forces --no-locomotion, but the sender now
#                             keeps hand_finger_cmd on :5572 in that mode (the
#                             old "arm-only" gate was removed 2026-10-27).
#
# The PC2 node's own defaults already match this sender: --operator-x-sign +1
# and --operator-y-sign +1. The frozen 3-point wire is the operator's own
# (forward, left, up) frame -- the PC service world is RIGHT-handed (X right,
# Y up, Z in; its README notes the real data flips Unity's +Z), so
# fwd = cross(lat, up) is operator-forward -- and the operator and the robot
# face the SAME direction, so the avatar map is the identity. The PC2 node
# also bakes the measured ctrl levers, so its live command line carries no
# calibration flags.
#
# PC2 prerequisites (this script does NOT start them):
#   1) incremental_upper_teleop_v3 from a bundle rebuilt with this tree (same
#      wire; see x2_pc2/v3_upper_teleop/README.md section 4).
#   2) msgpack CLAW bridge bound on :5572 when hands are on:
#        x2_hand_zmq_to_aimdk_bridge.py --wire-format msgpack \
#          --zmq-host '*' --zmq-port 5572 --zmq-topic hand_finger_cmd \
#          --hand-mode claw --sides auto
#
# --tracking is refused on purpose (this preset owns it), and a loopback host
# is refused because an IP must mean the real robot. Use
# run_pico_teleop.sh --tracking 3point for the local sim stack.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

usage() {
    cat <<'EOF'
Usage: run_pico_teleop_v4_real.sh --pc2-host <PC2_IP> [options...]

v4 real-robot preset: --tracking 3point, hands on trigger (press = close).
Use --no-hands for an arms-only session.

Options handled here:
  --pc2-host <ip>        required; loopback is rejected (sim uses run_pico_teleop.sh)
  --no-hands             pass --hand-source off (arms only)
  --hands                explicit default (hands on)
  -h, --help             this text

Sender/launcher options (--intent-port, --hand-port, --hand-source,
--record-dir, --no-record, --wrist-source, --cam, --headset-ip, ...) are
passed to run_pico_teleop.sh / pico_intent_sender.py unchanged. PC2 NODE
options (--ctrl-lever-*, --wrist-*-sign-*, --wrist-axis-order-*,
--operator-x-sign, --operator-y-sign, --backend, --full-pose-ik, --position-deadband-mm,
--posture-gain, --waist-stand, --no-legs, --record, --duration, --self-test,
--dry-run) are refused here:
put them on the PC2 node command. If you set --hand-source yourself, the
sender's default hand polarity is kept (closed at rest); add
--hand-close-on-grip for press = close.
EOF
}

HOST=""
INTENT_PORT=5573
HAND_PORT=5572
HANDS=1
HAND_SOURCE_SET=0
HAND_SOURCE_VAL=""
PASS=()

while (( $# )); do
    case "$1" in
        --pc2-host)           HOST="${2:-}"; PASS+=("$1" "${2:-}"); shift 2 ;;
        --pc2-host=*)         HOST="${1#*=}"; PASS+=("$1"); shift ;;
        --intent-port)        INTENT_PORT="${2:-}"; PASS+=("$1" "${2:-}"); shift 2 ;;
        --intent-port=*)      INTENT_PORT="${1#*=}"; PASS+=("$1"); shift ;;
        --hand-port)          HAND_PORT="${2:-}"; PASS+=("$1" "${2:-}"); shift 2 ;;
        --hand-port=*)        HAND_PORT="${1#*=}"; PASS+=("$1"); shift ;;
        --hand-source)        HAND_SOURCE_SET=1; HAND_SOURCE_VAL="${2:-}"; PASS+=("$1" "${2:-}"); shift 2 ;;
        --hand-source=*)      HAND_SOURCE_SET=1; HAND_SOURCE_VAL="${1#*=}"; PASS+=("$1"); shift ;;
        --hands)              HANDS=1; shift ;;
        --no-hands)           HANDS=0; shift ;;
        --tracking|--tracking=*)
            echo "[v4-real] ERROR: this launcher already sets --tracking 3point; drop '$1'." >&2
            exit 2 ;;
        --ctrl-lever*|--wrist-*-sign-*|--wrist-axis-order-*|--backend|--backend=*|\
        --operator-x-sign|--operator-x-sign=*|--operator-y-sign|--operator-y-sign=*|\
        --full-pose-ik|--position-deadband-mm|--position-deadband-mm=*|\
        --posture-gain|--posture-gain=*|--waist-stand|--waist-stand=*|\
        --no-legs|--self-test|--dry-run)
            echo "[v4-real] ERROR: '$1' is a PC2 NODE option -- put it on the PC2 node command, not the laptop sender." >&2
            exit 2 ;;
        --record|--duration|--duration=*)
            echo "[v4-real] ERROR: '$1' is the PC2 node's 85-column record option; the sender records intent tapes (see --record-dir / --no-record)." >&2
            exit 2 ;;
        --tape-replay|--tape-once)
            echo "[v4-real] ERROR: --tracking 3point is live-only; tape replay is refused." >&2
            exit 2 ;;
        -h|--help)            usage; exit 0 ;;
        *)                    PASS+=("$1"); shift ;;
    esac
done

if [[ -z "$HOST" ]]; then
    echo "[v4-real] ERROR: --pc2-host <PC2_IP> is required (real robot)." >&2
    echo "[v4-real]        For the local sim stack: run_pico_teleop.sh --tracking 3point." >&2
    exit 2
fi
case "$HOST" in
    127.0.0.1|localhost|::1)
        echo "[v4-real] ERROR: '$HOST' is the sim target; pass the PC2 address." >&2
        exit 2 ;;
esac
[[ "$INTENT_PORT" =~ ^[0-9]+$ ]] || { echo "[v4-real] ERROR: bad --intent-port '$INTENT_PORT'." >&2; exit 2; }
[[ "$HAND_PORT" =~ ^[0-9]+$ ]]   || { echo "[v4-real] ERROR: bad --hand-port '$HAND_PORT'." >&2; exit 2; }

# Working PC Service on this laptop (the GR00T arm64 deb lacks ICU 70). An
# explicit PICO_PC_SERVICE_DIR always wins; otherwise fall back to the local
# czq copy only when /opt/apps is absent.
if [[ -z "${PICO_PC_SERVICE_DIR:-}" ]] \
   && [[ ! -x /opt/apps/roboticsservice/RoboticsServiceProcess ]] \
   && [[ -x "$HOME/Documents/czq/roboticsservice/RoboticsServiceProcess" ]]; then
    export PICO_PC_SERVICE_DIR="$HOME/Documents/czq/roboticsservice"
fi

EFFECTIVE_SOURCE="$HAND_SOURCE_VAL"
EFFECTIVE_CLOSE=0
if (( HANDS )); then
    if (( ! HAND_SOURCE_SET )); then
        # Pure preset path: the v3 roles and the robot-day polarity.
        PASS+=(--hand-source trigger --hand-close-on-grip)
        EFFECTIVE_SOURCE=trigger
        EFFECTIVE_CLOSE=1
    elif [[ "$HAND_SOURCE_VAL" == "grip" ]]; then
        echo "[v4-real] WARNING: --hand-source grip uses the GRIP, which is the ARM CLUTCH in v4." >&2
        echo "[v4-real]          Use trigger to drive the CLAW; grip moves the arm." >&2
    fi
    for a in "${PASS[@]}"; do
        [[ "$a" == "--hand-close-on-grip" ]] && EFFECTIVE_CLOSE=1
    done
else
    if (( HAND_SOURCE_SET )) && [[ "$HAND_SOURCE_VAL" != "off" ]]; then
        echo "[v4-real] ERROR: --no-hands conflicts with --hand-source $HAND_SOURCE_VAL." >&2
        exit 2
    fi
    PASS+=(--hand-source off)
    EFFECTIVE_SOURCE=off
fi

PASS+=(--tracking 3point)

echo "[v4-real] target PC2 $HOST (intent :$INTENT_PORT, hands :$HAND_PORT)"
if (( HANDS )); then
    if (( EFFECTIVE_CLOSE )); then
        echo "[v4-real] preset: --tracking 3point --hand-source $EFFECTIVE_SOURCE --hand-close-on-grip (press = close)"
    else
        echo "[v4-real] preset: --tracking 3point --hand-source $EFFECTIVE_SOURCE (sender default polarity: closed at rest)"
    fi
else
    echo "[v4-real] preset: --tracking 3point --hand-source off (arms only)"
fi

_port_open() { timeout 2 bash -c "</dev/tcp/$1/$2" 2>/dev/null; }
if ! _port_open "$HOST" "$INTENT_PORT"; then
    echo "[v4-real] WARNING: $HOST:$INTENT_PORT not accepting TCP -- start incremental_upper_teleop_v3 on PC2." >&2
fi
if (( HANDS )) && [[ "$EFFECTIVE_SOURCE" != "off" ]] && ! _port_open "$HOST" "$HAND_PORT"; then
    echo "[v4-real] WARNING: $HOST:$HAND_PORT not accepting TCP -- the CLAW hand bridge is not bound." >&2
fi

if [[ -n "${V4_REAL_DRY_RUN:-}" ]]; then
    printf '[v4-real] would exec:'
    printf ' %q' "$REPO_ROOT/gear_sonic/scripts/run_pico_teleop.sh" "${PASS[@]}"
    echo
    exit 0
fi

exec "$REPO_ROOT/gear_sonic/scripts/run_pico_teleop.sh" "${PASS[@]}"
