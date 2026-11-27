#!/usr/bin/env bash
# The existing gates and the tests that hold the generated sedan's crash numbers, on a set of
# front rules (RUBBLE_FRONT_RULES: the designs' default rules for this run; `none` is the car
# before). From engine/:   tools/sections/front_gates.sh TAG RULES [TEST ...]
# Each test binary runs under a 295 s alarm; logs in out/gates/<TAG>/. The summary lists each
# binary's result and its failing tests.
set -u
tag=${1:?tag} rules=${2:?rules}
shift 2
dir=out/gates/$tag
mkdir -p "$dir"
all="vg:crash_zones vg:mechanical vg:lib au:generated_vehicle_lint au:every_design au:crash_zones au:truck_cab au:box_trailer au:gen3_rig au:sedan_energy au:crash_invariants au:gen_sedan_contact au:gen_sedan_paint au:continuous_contact"
list=${*:-$all}
for t in $list; do
  pkg=engine-softbody-authoring
  [ "${t%%:*}" = vg ] && pkg=engine-vehiclegen
  name=${t#*:}
  sel=(--test "$name")
  [ "$name" = lib ] && sel=(--lib)
  env=(RUBBLE_FRONT_RULES="$rules")
  [ "$name" = gen3_rig ] && env+=(GEN3_RIG_DESIGN=sedan)
  env "${env[@]}" perl -e 'alarm 295; exec @ARGV' cargo test --release -q -p "$pkg" "${sel[@]}" -- --test-threads 4 \
    </dev/null >"$dir/$name.log" 2>&1
  code=$?
  res=$(grep -E '^test result' "$dir/$name.log" | tail -1)
  fails=$(grep -E '^test .* FAILED$|^    [a-z_:]+$' "$dir/$name.log" | sed 's/^test //; s/ \.\.\. FAILED//; s/^ *//' | sort -u | tr '\n' ' ')
  echo "$tag $name: exit $code; ${res:-no result (timeout?)}; failed: ${fails:-none}"
done
