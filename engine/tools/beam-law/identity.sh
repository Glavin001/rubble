#!/usr/bin/env bash
# The 8-run report identity (beam-yield-law.md §6): S11 and S02, 8 runs × discrete/edge/continuous,
# default designs (every opt-in off), one process a run under its own 295 s alarm, 12 at a time.
# From engine/:   tools/beam-law/identity.sh TAG     → out/beam-law/ident/TAG/<S>-<contact>-<run>.json
# Compare two tags: diff -r out/beam-law/ident/A out/beam-law/ident/B (the reports carry no clock).
set -u
tag=${1:?tag}
dir=out/beam-law/ident/$tag
mkdir -p "$dir"
one() {
  perl -e 'alarm 295; exec @ARGV' target/release/examples/crash_ref "$1" --runs 8 --run "$3" --contact "$2" --json \
    </dev/null >"$dir/$1-$2-$3.json" 2>/dev/null || echo "fail $1 $2 $3"
}
n=0
for s in S11 S02; do
  for c in discrete edge continuous; do
    for k in 0 1 2 3 4 5 6 7; do
      one "$s" "$c" "$k" &
      n=$((n + 1))
      if [ $((n % 12)) -eq 0 ]; then wait; fi
    done
  done
done
wait
echo "done: $dir ($(ls "$dir" | wc -l) reports)"
