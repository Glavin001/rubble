#!/usr/bin/env bash
# The page's own real-time loop before and after the skin batch, rounds interleaved in
# one run a scene (skin-batch-live.js): "before" skins one by one with BASE's kernel
# (put under generated/, gitignored), "after" is the page's default. One GPU job at a
# time (run.sh takes the lock), one lock hold a scene; each run is bounded.
# Summary: python3 engine/tools/vehicle-lab/skin-batch-live.py OUTDIR
#
#   engine/tools/vehicle-lab/skin-batch-live.sh OUTDIR [scene ...]   (derby4 derby8 mountain8)
#   PRE="window.__sblRounds = 3; window.__sblWindow = 10;" ...         (the defaults)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WEB="$HERE/../../crates/engine-webgpu"
mkdir -p "${1:?out dir}"
OUT="$(cd "$1" && pwd)"; shift
BASE="${BASE:-origin/main}"
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
for name in "$@"; do
  { echo "window.__skinRefUrl = './generated/skin-ref/showcase-car-render.mjs'; ${PRE:-}"; cat "$HERE/skin-batch-live.js"; } > "$OUT/$name.eval.js"
  LIMIT=600 bash "$HERE/run.sh" --probe --secs 1 --limit 8 ${UNCAPPED:+--uncapped} --eval "$OUT/$name.eval.js" --out "$OUT" "$name:$(query "$name")" > "$OUT/$name.txt" 2>&1 || echo "exit $?" >> "$OUT/$name.txt"
done
