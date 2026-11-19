#!/usr/bin/env bash
# The page's own real-time loop (physics-first.js) before and after the skin batch,
# ROUNDS rounds interleaved per scene: "before" skins one by one with BASE's kernel
# (skin-batch.sh puts it under generated/), "after" is the page's default. One GPU job
# at a time (run.sh takes the lock); each run is bounded.
#
#   engine/tools/vehicle-lab/skin-batch-live.sh OUTDIR [scene ...]   (derby4 derby8 mountain8)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WEB="$HERE/../../crates/engine-webgpu"
mkdir -p "${1:?out dir}"
OUT="$(cd "$1" && pwd)"; shift
BASE="${BASE:-origin/main}"
ROUNDS="${ROUNDS:-3}"
mkdir -p "$WEB/generated/skin-ref"
git -C "$HERE" show "$BASE:engine/crates/engine-webgpu/showcase-car-render.mjs" \
  | sed "s#from './#from '../../#" > "$WEB/generated/skin-ref/showcase-car-render.mjs"
query() {
  case "$1" in
    derby4) echo "mode=derby&car=gen&surface=0&derbyCars=4&timing=step" ;;
    derby8) echo "mode=derby&car=gen&surface=0&derbyCars=8&timing=step" ;;
    mountain8) echo "mode=mountain&car=gen&cars=8&timing=step" ;;
    *) echo "unknown scene $1" >&2; exit 1 ;;
  esac
}
[[ $# -gt 0 ]] || set -- derby4 derby8 mountain8
before="(async () => { const ref = (await import('./generated/skin-ref/showcase-car-render.mjs')).SHOWCASE_SKIN_WGSL.skin; const c = document.getElementById('skinBatch'); c.value = 'off'; c.dispatchEvent(new Event('change', { bubbles: true })); window.__skinCode(ref); })();"
for name in "$@"; do
  for r in $(seq 1 "$ROUNDS"); do
    for side in before after; do
      { echo "window.__pfWall = ${WALL:-20}; window.__pfDrawOff = ${DRAWOFF:-2};"; [[ $side == before ]] && echo "$before"; cat "$HERE/physics-first.js"; } > "$OUT/$name.$side.eval.js"
      LIMIT=600 bash "$HERE/run.sh" --probe --secs 1 --limit 6 ${UNCAPPED:+--uncapped} --eval "$OUT/$name.$side.eval.js" --out "$OUT" "$name:$(query "$name")" > "$OUT/$name.$side.$r.txt" 2>&1 || echo "exit $?" >> "$OUT/$name.$side.$r.txt"
    done
  done
done
