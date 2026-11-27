#!/usr/bin/env bash
# Generates every registered design into OUT/<design>_new with an optional --params file per
# design (PARAMS_DIR/<design>.json when it exists). From engine/:
#   tools/sections/gen_all.sh OUT [PARAMS_DIR]
set -u
out=${1:?out dir}
pdir=${2:-}
bin=target/release/examples/vehicle_gen_rig
for d in $($bin --list </dev/null); do
  args=("$d" --out "$out/${d}_new")
  if [ -n "$pdir" ] && [ -f "$pdir/$d.json" ]; then args+=(--params "$pdir/$d.json"); fi
  perl -e 'alarm 280; exec @ARGV' "$bin" "${args[@]}" </dev/null >"$out.$d.log" 2>&1 || echo "FAIL $d (see $out.$d.log)"
done
# SHA-256 of each design's files (the glb, rskin, JBeam set and the rest), one line a file.
( cd "$out" && find . -type f | LC_ALL=C sort | xargs shasum -a 256 ) >"$out.sha256"
echo "wrote $out.sha256"
