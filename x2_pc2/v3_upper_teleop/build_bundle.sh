#!/usr/bin/env bash
# Assemble the flat v3 robot-node bundle that is scp'd into the PC2
# czq_examples package (no gear_sonic package exists there).
#
# The node source lives HERE; the chain / solver files are copied VERBATIM
# from the repo so the sim tree and the robot tree cannot drift. The HAL
# table is the v1 vendor table from the local x2_sdk_v100 checkout (PC2
# already has a copy; the bundle makes the directory self-contained).
#
# Usage:
#   ./build_bundle.sh                    # -> ./dist/
#   SELF_TEST=1 ./build_bundle.sh        # also run --self-test --backend dls
#   PYTHON=/path/to/python ./build_bundle.sh
#   HAL_TABLE=/path/to/upper_body_teleop.py ./build_bundle.sh
#
# The output is disposable: every run wipes and recreates it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
OUT="${1:-$HERE/dist}"
PYTHON="${PYTHON:-python3}"
HAL_TABLE="${HAL_TABLE:-$REPO/../../x2_sdk_v100/src/py_examples/py_examples/upper_body_teleop.py}"

if [[ "$OUT" == "/" || -z "$OUT" ]]; then
    echo "refusing to build into '$OUT'" >&2
    exit 2
fi
for f in \
    "$HERE/incremental_upper_teleop_v3.py" \
    "$REPO/gear_sonic/utils/teleop/incremental_chain.py" \
    "$REPO/gear_sonic/utils/teleop/incremental_chain_v3.py" \
    "$REPO/gear_sonic/utils/teleop/solver/arm/x2_arm_ik.py" \
    "$REPO/gear_sonic/utils/teleop/solver/arm/x2_arm_fk.py" \
    "$HAL_TABLE"; do
    [[ -f "$f" ]] || { echo "missing source: $f" >&2; exit 2; }
done

rm -rf "$OUT"
mkdir -p "$OUT"

install -m 0755 "$HERE/incremental_upper_teleop_v3.py" "$OUT/"
# v3's base chain gets a v3-only name on purpose: PC2's `incremental_chain.py`
# belongs to the v2 node (old trigger/grip semantics) and must stay intact.
install -m 0644 "$REPO/gear_sonic/utils/teleop/incremental_chain.py" \
    "$OUT/incremental_chain_v3base.py"
install -m 0644 "$REPO/gear_sonic/utils/teleop/incremental_chain_v3.py" "$OUT/"
install -m 0644 "$REPO/gear_sonic/utils/teleop/solver/arm/x2_arm_ik.py" "$OUT/"
install -m 0644 "$REPO/gear_sonic/utils/teleop/solver/arm/x2_arm_fk.py" "$OUT/"
install -m 0644 "$HAL_TABLE" "$OUT/upper_body_teleop.py"
install -m 0644 "$HERE/README.md" "$OUT/README.md"

{
    echo "built_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "source_repo=$REPO"
    echo "source_commit=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "source_dirty=$(test -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" && echo yes || echo no)"
    echo "hal_table=$HAL_TABLE"
    echo "python=$PYTHON"
    echo "files:"
    for f in "$OUT"/*.py; do
        echo "  $(basename "$f")"
    done
} > "$OUT/bundle_info.txt"

# Syntax gate: a stale .pyc hiding a syntax error must not reach the robot.
"$PYTHON" -m py_compile "$OUT"/*.py

# The robot node only needs the flat siblings; prove the import chain and a
# real tick/record row work before the bundle leaves the laptop.
"$PYTHON" - "$OUT" <<'PY'
import sys
from pathlib import Path

out = Path(sys.argv[1])
sys.path.insert(0, str(out))
import incremental_chain_v3base  # noqa: F401
import incremental_chain_v3  # noqa: F401
import x2_arm_ik  # noqa: F401
import x2_arm_fk  # noqa: F401

print(f"[bundle] flat imports OK: {sorted(p.name for p in out.glob('*.py'))}")
PY

(cd "$OUT" && sha256sum ./*.py README.md bundle_info.txt > SHA256SUMS)

echo "[bundle] wrote $OUT"
echo "[bundle] source commit: $(awk -F= '/^source_commit=/{print $2}' "$OUT/bundle_info.txt")"
echo "[bundle] scp the six .py files FLAT into the PC2 package directory:"
echo "  scp $OUT/incremental_upper_teleop_v3.py \\"
echo "      $OUT/incremental_chain_v3base.py $OUT/incremental_chain_v3.py \\"
echo "      $OUT/x2_arm_ik.py $OUT/x2_arm_fk.py $OUT/upper_body_teleop.py \\"
echo "      Agibot-X2-PC2-Orin-NX:~/Workspaces/czq/ros2_ws/src/czq_examples/czq_examples/"

if [[ "${SELF_TEST:-0}" == "1" ]]; then
    echo "[bundle] running the robot self-test on the flat bundle (backend dls) ..."
    "$PYTHON" "$OUT/incremental_upper_teleop_v3.py" --self-test --backend dls
fi
