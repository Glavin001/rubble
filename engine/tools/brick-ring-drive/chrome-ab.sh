#!/bin/bash
# Interleaved A/B of the brick ring drive in headless Chrome (each run its own Chrome, under
# engine/scripts/with-gpu-lock.py with a 10 min kill): two engine-webgpu directories, each
# with its own generated/ (wgsl_export, world wasm) and __mods. Per round and side: the
# ETK (n = 1, cruising then through the wall), two ETKs, the sedan (each
# brick-ring-drive-record.mjs --probe, ?auto=1&timing=phases), then the ring alone
# (ring-alone-chrome.mjs, demo.html ?scene=ring110k&sleep=on).
#
#   bash engine/tools/brick-ring-drive/chrome-ab.sh <base webgpu dir> <new webgpu dir> [rounds] [secs]
# Logs: $AB_OUT (default ~/Development/rubble-artifacts/brick-ring-drive/ab-<date>).
set -u
here="$(cd "$(dirname "$0")" && pwd)"
lock="$here/../../scripts/with-gpu-lock.py"
A="$1"; B="$2"; ROUNDS=${3:-2}; SECS=${4:-14}
OUT=${AB_OUT:-$HOME/Development/rubble-artifacts/brick-ring-drive/ab-$(date +%Y%m%d-%H%M)}
mkdir -p "$OUT"
RUNS=${AB_RUNS:-"etk:car=etk&timing=phases etk2:car=etk&n=2&timing=phases sedan:car=sedan&timing=phases"}
for r in $(seq 1 "$ROUNDS"); do
  for side in base:"$A" new:"$B"; do
    name=${side%%:*}; dir=${side#*:}
    for run in $RUNS; do
      log="$OUT/$name-${run%%:*}-$r.log"
      echo "== round $r $name ${run%%:*}" | tee "$log"
      (cd "$dir" && python3 "$lock" --limit 600 -- node brick-ring-drive-record.mjs --probe --secs "$SECS" \
        --out "$OUT/scratch" "$run") >> "$log" 2>&1
      cat "$log"
    done
    if [ "${AB_RING:-1}" = 1 ]; then
      log="$OUT/$name-ring-$r.log"
      echo "== round $r $name ring alone" | tee "$log"
      python3 "$lock" --limit 600 -- node "$here/ring-alone-chrome.mjs" --root "$dir" >> "$log" 2>&1
      cat "$log"
    fi
  done
done
echo "logs: $OUT"
