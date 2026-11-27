#!/usr/bin/env bash
# The owner's hard rules on a front-rules variant (deciding-tradeoffs): S11, S12, S01, S08 on the
# generated sedan and S02 on the 4x4, 8 runs in each contact setting, CPU. From engine/:
#
#   tools/sections/front_hard_rules.sh TAG [rule ...] [all] [crash=PRESET]
#   python3 -I tools/sections/front_hard_rules.py TAG [TAG ...]     # the summary
#
# Writes out/gen/hard_<TAG>/<scenario>-<contact>.jsonl (one report a run) and the params files.
set -u
tag=${1:?tag}
shift
dir=out/gen/hard_$tag
mkdir -p "$dir"
for d in sedan suv_4x4; do
  perl -e 'alarm 120; exec @ARGV' python3 -I tools/sections/front_params.py "$d" "$PWD/$dir/$d.json" "$@" </dev/null || exit 1
done
p="sedan=$PWD/$dir/sedan.json,suv_4x4=$PWD/$dir/suv_4x4.json"
for s in S11 S12 S01 S02 S08; do
  for c in discrete edge continuous; do
    : >"$dir/$s-$c.jsonl"
    for k in 0 1 2 3 4 5 6 7; do
      perl -e 'alarm 295; exec @ARGV' target/release/examples/crash_ref "$s" --runs 8 --run "$k" --contact "$c" --params "$p" --json \
        </dev/null >>"$dir/$s-$c.jsonl" 2>>"$dir/$s-$c.err" || true
    done
    echo "$s $c: $(wc -l <"$dir/$s-$c.jsonl") runs"
  done
done
