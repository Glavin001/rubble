#!/usr/bin/env bash
# The skin batch's check and timing (skin-batch.js) on the lab's scenes, against the
# skin kernel of BASE (a commit; default origin/main): its showcase-car-render.mjs is
# put under generated/ (gitignored) and the page skins one by one with it too, so the
# skinned vertices are compared with the code before the change on the same state.
# One GPU job at a time (run.sh takes the lock).
#
#   engine/tools/vehicle-lab/skin-batch.sh OUTDIR [scene ...]   (derby4 derby8 mountain8)
#   PRE="window.__skinBatchSamples=[5];" ...                    (options for skin-batch.js)
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
  { echo "window.__skinRefUrl = './generated/skin-ref/showcase-car-render.mjs'; ${PRE:-}"; cat "$HERE/skin-batch.js"; } > "$OUT/$name.eval.js"
  LIMIT=900 bash "$HERE/run.sh" --probe --secs 1 --limit 14 --eval "$OUT/$name.eval.js" --out "$OUT" "$name:$(query "$name")" > "$OUT/$name.txt" 2>&1 || echo "$name: exit $?" >> "$OUT/$name.txt"
done
