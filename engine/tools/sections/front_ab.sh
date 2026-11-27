#!/usr/bin/env bash
# The generated sedan with a set of front rules (`Physics::front_rules`, lessons §25), measured:
# the rigged lattice's cuts, the slow push, S11 and S12 (CPU, edge contact, run 0) and S11's
# elastic energy by member. From engine/:
#
#   tools/sections/front_ab.sh TAG [rule ...] [crash=real_crash]
#   tools/sections/front_ab.sh off                      # every rule off (the car before)
#   tools/sections/front_ab.sh r1 subframe_mounts_off_crash_box
#   python3 tools/sections/front_ab.py off r1 ...       # the comparison table (and plots)
#
# Rules: subframe_mounts_off_crash_box crash_box_bay_alone crash_box_stroke yielding_stops
# hood_fold_row crush_from_section (any subset; `all` for every one). Writes
# out/gen/sedan_<TAG>/ (the mod folder, params.json, s11s12.jsonl, audit/). Every step is
# bounded (perl alarm); CPU only.
set -eu
tag=${1:?tag}
shift
dir=out/gen/sedan_$tag
mkdir -p "$dir/audit"
params=$PWD/$dir/params.json
perl -e 'alarm 120; exec @ARGV' python3 -I tools/sections/front_params.py sedan "$params" "$@" </dev/null
echo "$tag: $*"
alarm() { perl -e 'alarm shift; exec @ARGV' "$@" </dev/null; }
alarm 280 target/release/examples/vehicle_gen_rig sedan --params "$params" --out "$dir" >"$dir/rig.log" 2>&1
alarm 290 target/release/examples/crash_ref S11 S12 --params "sedan=$params" --json >"$dir/s11s12.jsonl" 2>"$dir/crash_ref.err" || true
for t in sedan_front_cuts sedan_front_push sedan_s11_stored_energy; do
  AUDIT_OUT=$PWD/$dir/audit RUBBLE_DESIGN_PARAMS="sedan=$params" alarm 290 \
    cargo test --release -q -p engine-softbody-lab --test sedan_front_audit -- --ignored --exact "$t" --nocapture --test-threads 1 \
    >"$dir/audit/$t.log" 2>&1 || echo "$t failed (see $dir/audit/$t.log)"
done
echo "done: $dir"
