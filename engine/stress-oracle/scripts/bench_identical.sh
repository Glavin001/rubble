#!/usr/bin/env bash
# Wall-time benchmark and bit-identity check of a stress-ref build.
#
#   scripts/bench_identical.sh BIN OUTDIR [BASELINE_DIR]
#
# Runs a fixed set of scenes with BIN, writing OUTDIR/<scene>.json and OUTDIR/times.txt.
# With BASELINE_DIR (an earlier OUTDIR), prints each scene's wall time against the
# baseline and whether the observation is identical (ignoring wall_seconds).
# Run it on a quiet machine: concurrent jobs make the timings meaningless.
set -euo pipefail
BIN=$1; OUT=$2; BASE=${3:-}
cd "$(dirname "$0")/.."
mkdir -p "$OUT"; : > "$OUT/times.txt"
for s in b1_bond_tension b2_cantilever b3_bar_wave b4_support_loss_sudden b5_wall_impact_v10 \
         b8_frame_sudden b9_panel_high showcases/s_arch_keystone_removed showcases/s_car_brick \
         showcases/s_house_car showcases/s_floor_drop; do
  n=$(basename "$s")
  start=$(date +%s.%N)
  "$BIN" run "scenes/$s.json" --out "$OUT/$n.json" > /dev/null 2>&1
  echo "$n $(echo "$(date +%s.%N) - $start" | bc)" >> "$OUT/times.txt"
done
[ -z "$BASE" ] && { cat "$OUT/times.txt"; exit 0; }
python3 - "$OUT" "$BASE" <<'PY'
import json, sys
out, base = sys.argv[1], sys.argv[2]
def strip(o):
    if isinstance(o, dict): return {k: strip(v) for k, v in o.items() if k != 'wall_seconds'}
    if isinstance(o, list): return [strip(x) for x in o]
    return o
times = lambda d: dict(l.split() for l in open(f'{d}/times.txt'))
t, b = times(out), times(base)
for n in t:
    same = strip(json.load(open(f'{out}/{n}.json'))) == strip(json.load(open(f'{base}/{n}.json')))
    print(f"{n:26} {float(b[n]):8.2f} s -> {float(t[n]):8.2f} s  {'IDENTICAL' if same else 'DIFFERENT'}")
PY
