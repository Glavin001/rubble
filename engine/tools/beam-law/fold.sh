#!/usr/bin/env bash
# How the load path's crush cells fold in S11 and S12 (beam-yield-law.md §9): the sedan with
# load_path + crush_law, then each measurement of `CRUSH_CELL_EXP` (engine_vehiclegen::physics::
# crush_cell_exp) alone and together, through `front_fold` (CPU, edge contact, run 0; about 7 s a
# run). From engine/, after `cargo build --release -p engine-softbody-lab --example front_fold
# --example vehicle_gen_rig`:
#
#   tools/beam-law/fold.sh            → out/beam-law/fold/<variant>/<S>-edge-0-{bays,stations}.csv, -summary.txt
#   python3 -I tools/beam-law/fold_table.py      # the tables;  python3 -I tools/beam-law/fold_plot.py  # SVG plots
set -u
out=out/beam-law/fold
mkdir -p "$out"
params=$PWD/$out/lpx.json
perl -e 'alarm 120; exec @ARGV' python3 -I tools/sections/front_params.py sedan "$params" load_path crush_law </dev/null || exit 1
run() {
  local tag=$1 exp=$2
  for s in S11 S12; do
    CRUSH_CELL_EXP=$exp perl -e 'alarm 295; exec @ARGV' target/release/examples/front_fold "$s" --params "sedan=$params" --out "$PWD/$out/$tag" \
      </dev/null >/dev/null 2>"$out/$tag-$s.err" || echo "fail $tag $s"
  done
}
run lpx ""
run chord_k chord_k
run diag_tc diag_tc
run camry_sections camry_sections
run no_peak no_peak
run all chord_k,diag_tc,camry_sections
run all_no_peak chord_k,diag_tc,camry_sections,no_peak
echo "done: $out"
