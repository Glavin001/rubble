#!/usr/bin/env bash
# One scenario and contact setting, 8 runs, on a front-rules variant (the sedan's and the 4x4's
# params from out/gen/hard_<TAG>/ or made here), one process a run (each under 295 s). From
# engine/:   tools/sections/front_one.sh TAG SCENARIO CONTACT [rule ...] [all] [crash=PRESET]
set -u
tag=${1:?tag} s=${2:?scenario} c=${3:?contact}
shift 3
dir=out/gen/hard_$tag
mkdir -p "$dir"
for d in sedan suv_4x4; do
  [ -f "$dir/$d.json" ] || perl -e 'alarm 120; exec @ARGV' python3 -I tools/sections/front_params.py "$d" "$PWD/$dir/$d.json" "$@" </dev/null || exit 1
done
p="sedan=$PWD/$dir/sedan.json,suv_4x4=$PWD/$dir/suv_4x4.json"
: >"$dir/$s-$c.jsonl"
for k in 0 1 2 3 4 5 6 7; do
  perl -e 'alarm 295; exec @ARGV' target/release/examples/crash_ref "$s" --runs 8 --run "$k" --contact "$c" --params "$p" --json \
    </dev/null >>"$dir/$s-$c.jsonl" 2>>"$dir/$s-$c.err" || true
done
echo "$tag $s $c: $(wc -l <"$dir/$s-$c.jsonl") runs"
