#!/usr/bin/env bash
# Drawing against physics, the lab's scenes that matter (the derby at 4 and 8 cars, the
# mountain at 8), each a probe of the page's own real-time loop (physics-first.js: 30 s
# drawn as the page draws, then 8 s with nothing drawn) and the drawing's parts with the
# World paused (draw-cost.js). One GPU job at a time (run.sh takes the lock).
#
#   engine/tools/vehicle-lab/physics-first.sh OUTDIR [scene ...]   (scenes: derby4 derby8 mountain8)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:?out dir}"; shift
mkdir -p "$OUT"
query() {
  case "$1" in
    derby4) echo "mode=derby&car=gen&surface=0&derbyCars=4&timing=step" ;;
    derby8) echo "mode=derby&car=gen&surface=0&derbyCars=8&timing=step" ;;
    mountain8) echo "mode=mountain&car=gen&cars=8&timing=step" ;;
    *) echo "unknown scene $1" >&2; exit 1 ;;
  esac
}
[[ $# -gt 0 ]] || set -- derby4 derby8 mountain8
for name in "$@"; do
  q="$(query "$name")${EXTRA_Q:+&$EXTRA_Q}"
  LIMIT=600 bash "$HERE/run.sh" --probe --secs 1 --limit 10 ${UNCAPPED:+--uncapped} --eval "$HERE/physics-first.js" --out "$OUT" "$name:$q" > "$OUT/$name.live.txt" 2>&1 || echo "$name live: exit $?" >> "$OUT/$name.live.txt"
  [[ -n "${NO_PARTS:-}" ]] || LIMIT=600 bash "$HERE/run.sh" --probe --secs 1 --limit 10 ${UNCAPPED:+--uncapped} --eval "$HERE/draw-cost.js" --out "$OUT" "$name:$q" > "$OUT/$name.parts.txt" 2>&1 || echo "$name parts: exit $?" >> "$OUT/$name.parts.txt"
done
