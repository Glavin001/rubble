#!/bin/sh
# Regenerates the raw tables behind engine/docs/beamng-structure-study.md.
#
#   JBEAMS=~/Downloads/jbeams OUT=~/Development/rubble-artifacts/beamng-structure-study \
#     engine/tools/beamng-structure-study/run.sh
#
# JBEAMS holds the five unpacked mods (data only; nothing from them is committed).
# OUT gets dump/<car>.json (the cars' rows and meshes) and <car>.<section>.md.
# CPU only. A car whose directory is absent is skipped.
set -eu
JBEAMS="${JBEAMS:-$HOME/Downloads/jbeams}"
OUT="${OUT:-$HOME/Development/rubble-artifacts/beamng-structure-study}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT/dump"
if [ "${SKIP_DUMP:-0}" != "1" ]; then
  RUBBLE_STRUCTURE_OUT="$OUT/dump" \
  RUBBLE_CENSUS_CARS="etkk_kc6=$JBEAMS/etkk_lucasbe/vehicles,etkk,kc6.pc;etkk_kr6=$JBEAMS/etkk_lucasbe/vehicles,etkk,kr6.pc;ccf=$JBEAMS/thw_ccf2/vehicles,ccf,f4_sport_M.pc;vertex=$JBEAMS/innocent_gavril_vertex/vehicles,gavril_vertex,gavril_vertex.pc;fullsuv=$JBEAMS/fsw_fullsizesuv/vehicles,fullsuv,mid-sport.pc;boathauler=$JBEAMS/boathauler/vehicles,boathauler,gooseneck.pc" \
    cargo test --release --manifest-path "$HERE/../../Cargo.toml" -p engine-softbody --test authoring_census -- --ignored --nocapture structure_dump
fi
for f in "$OUT"/dump/*.json; do
  car="$(basename "$f" .json)"
  for s in parts attach body grade brace mech wheelpath glass flex stab types panels height counts; do
    python3 -I "$HERE/analyze.py" "$OUT/dump" "$car" --only "$s" > "$OUT/$car.$s.md"
  done
  echo "wrote $OUT/$car.*.md"
done
# The cross-car tables.
python3 -I "$HERE/analyze.py" "$OUT/dump" --only summary > "$OUT/summary.md"
echo "wrote $OUT/summary.md"
