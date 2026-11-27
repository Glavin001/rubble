#!/bin/bash
# The Camry side comparison (engine/docs/crash-reference/camry-side-comparison.md), regenerated
# step by step. From engine/, after
#   cargo build --release -p engine-softbody-lab --example crash_ref --example crash_outlines
#
#   tools/crash-reference/camry_side.sh runs      # crash_ref reports, CPU reference, run 0
#   tools/crash-reference/camry_side.sh render    # Rubble stills at the films' times and views
#   tools/crash-reference/camry_side.sh real      # the films' stills at the same times
#   tools/crash-reference/camry_side.sh sheets    # side-by-side sheets (to the artifacts folder)
#   tools/crash-reference/camry_side.sh curves    # lateral curves, real against Rubble
#
# ROOT: the local NHTSA downloads (gitignored, never read by a build or test). Every job runs
# under its own 290 s kill timer. Third-party footage stays out of git: stills and sheets go
# to out/ and ~/Development/rubble-artifacts/crash-reference/camry-side/.
set -u
cd "$(dirname "$0")/../.."
ROOT=${ROOT:-$HOME/Development/rubble/inspiration/ccsa-camry-2012}
OUT=out/camry-side
ART=${ART:-$HOME/Development/rubble-artifacts/crash-reference/camry-side}
ETK=local:etkk_tex/etkk/kc6.pc
export RUBBLE_LOCAL_MODS=${RUBBLE_LOCAL_MODS:-$HOME/Development/rubble/.claude/worktrees/rubble-crash-physics-reference-4d421e/engine/crates/engine-webgpu/__mods}
T=(perl -e 'alarm 290; exec @ARGV')
TIMES=-10,0,10,20,30,40,60,80,100,150,200,300
mkdir -p $OUT/runs $OUT/rubble

runs() {
  for s in S13 S14; do
    for c in edge discrete continuous; do
      "${T[@]}" target/release/examples/crash_ref $s --run 0 --contact $c --json > $OUT/runs/$s-$c.json 2> $OUT/runs/$s-$c.err </dev/null
      "${T[@]}" target/release/examples/crash_ref $s --run 0 --contact $c --set car=$ETK --json > $OUT/runs/$s-etk-$c.json 2> $OUT/runs/$s-etk-$c.err </dev/null
      echo "$s $c: $(tail -c 200 $OUT/runs/$s-$c.err | tr '\n' ' ') | etk: $(tail -c 200 $OUT/runs/$s-etk-$c.err | tr '\n' ' ')"
    done
  done
}

render() {
  for s in S13 S14; do
    "${T[@]}" target/release/examples/crash_ref --export $s > $OUT/$s.export.json </dev/null
    "${T[@]}" target/release/examples/crash_ref --export $s --set car=$ETK > $OUT/$s-etk.export.json </dev/null
  done
  # The impact point in each layout's world frame (the barrier face's centre at contact; the
  # pole's centre): read from the exports.
  python3 tools/crash-reference/side_compare.py rubble-cams S14 $OUT/S14.export.json --out $OUT/rubble/S14-film.json --at 0.9 -1.313
  python3 tools/crash-reference/side_compare.py rubble-cams S13 $OUT/S13.export.json --out $OUT/rubble/S13-film.json --at 1.07 -1.96
  python3 tools/crash-reference/side_compare.py rubble-cams S14 $OUT/S14-etk.export.json --out $OUT/rubble/S14-etk-film.json --at 0.9 0.077
  python3 tools/crash-reference/side_compare.py rubble-cams S13 $OUT/S13-etk.export.json --out $OUT/rubble/S13-etk-film.json --at 1.057 -0.795
  for j in "S13-film edge" "S14-film discrete" "S14-film edge" "S13-etk-film edge" "S14-etk-film discrete"; do
    set -- $j
    "${T[@]}" target/release/examples/crash_outlines --scenario $OUT/rubble/$1.json --cameras $OUT/rubble/$1.cameras.json --contact $2 --size 960x640 --out $OUT/rubble/$1-$2 > $OUT/rubble/$1-$2.log 2>&1 </dev/null
    echo "$1 $2: exit $?"
  done
}

real() {
  for v in "7517 v07517C001.avi" "7517 v07517C002.avi" "7517 v07517C005.avi" "7517 v07517C006.avi" "7517 v07517C004.avi" \
           "8560 v08560C001.avi" "8560 v08560C002.avi" "8560 v08560C005.avi" "8560 v08560C006.avi" "8560 v08560C004.avi" \
           "7523 v07523C001.avi" "7523 v07523C004.avi" "7523 v07523C002.avi" "7523 v07523C006.avi" "7523 v07523C003.avi" \
           "8558 v08558C001.avi" "8558 v08558C004.avi" "8558 v08558C002.avi" "8558 v08558C009.avi" "8558 v08558C003.avi"; do
    set -- $v
    "${T[@]}" python3 tools/crash-reference/side_compare.py real "$ROOT" $1 $2 -- $TIMES > /dev/null </dev/null || echo "$1 $2 failed"
  done
}

sheets() {
  for f in tools/crash-reference/camry-side-sheets/*.json; do
    "${T[@]}" python3 tools/crash-reference/side_compare.py sheet $f --out $OUT/sheets/$(basename $f .json).png --copy-to "$ART" </dev/null
  done
}

curves() {
  P=${PY:-python3}
  mkdir -p $OUT/curves
  $P tools/crash-reference/side_compare.py curves --test docs/crash-reference/nhtsa/7517.json --test docs/crash-reference/nhtsa/8560.json \
    --run "gen sedan, discrete=$OUT/runs/S14-discrete.json" --run "gen sedan, edge=$OUT/runs/S14-edge.json" --run "gen sedan, continuous=$OUT/runs/S14-continuous.json" \
    --run "ETK, discrete=$OUT/runs/S14-etk-discrete.json" --barrier barrier --title "S14 against NHTSA 7517, 8560 (struck car CG, +Y right)" --out $OUT/curves/S14-lateral.png
  $P tools/crash-reference/side_compare.py curves --test docs/crash-reference/nhtsa/7523.json --test docs/crash-reference/nhtsa/8558.json \
    --run "gen sedan, edge=$OUT/runs/S13-edge.json" --run "gen sedan, continuous=$OUT/runs/S13-continuous.json" \
    --run "ETK, edge=$OUT/runs/S13-etk-edge.json" --title "S13 against NHTSA 7523, 8558 (CG, +Y right)" --out $OUT/curves/S13-lateral.png
  mkdir -p "$ART" && cp $OUT/curves/*.png "$ART"/
}

for step in "$@"; do $step; done
