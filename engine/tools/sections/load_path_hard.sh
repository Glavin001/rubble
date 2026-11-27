#!/usr/bin/env bash
# The owner's hard rules on the load path (front_hard_rules.sh's set: S11, S12, S01, S08 on the
# sedan, S02 on the 4x4, 8 runs × discrete/edge/continuous), the 15 settings in parallel, each
# through front_one.sh (every run under its own 295 s alarm). From engine/:
#   tools/sections/load_path_hard.sh TAG [rule ...]      # e.g. lp load_path
#   python3 -I tools/sections/front_hard_rules.py TAG; cargo run --release -p engine-softbody-lab --example crash_verdict -- out/gen/hard_TAG
set -u
tag=${1:?tag}
shift
for s in S11 S12 S01 S02 S08; do
  for c in discrete edge continuous; do
    tools/sections/front_one.sh "$tag" "$s" "$c" "$@" >/dev/null 2>&1 &
    sleep 1
  done
done
wait
echo "done: $tag"
