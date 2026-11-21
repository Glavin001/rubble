#!/usr/bin/env bash
# One CI job's share of the engine host tests (no GPU). Run from engine/.
#
#   host-tests.sh rest             every host-tested crate but engine-softbody
#   host-tests.sh softbody N M     shard N of M of engine-softbody (host-test-shards.py)
#
# Together the jobs run what one command ran before:
#   cargo test --release -p engine-schema -p engine-exec -p engine-scenes -p engine-vehicle
#     -p engine-softbody -p engine-fracture -p engine-vehiclegen --no-default-features --no-fail-fast
# which took 23-35 minutes in one job (about 20 of building 130 test binaries and
# examples with thin LTO, 13 of running engine-softbody's tests).
#
# Writes host-tests.log and host-tests.code (the largest cargo exit code); the job
# "Engine host tests (no GPU)" joins every share's log and checks it against
# tests-expected-failures.txt. Here the share is checked alone (--partial), so a new
# failure shows in the job that ran it.
set -uo pipefail

scripts="$(cd "$(dirname "$0")" && pwd)"
group="${1:?usage: host-tests.sh rest | softbody N M}"
log=host-tests.log
: > "$log"
code=0
run() {
  echo "+ cargo $*" | tee -a "$log"
  cargo "$@" 2>&1 | tee -a "$log"
  local c=${PIPESTATUS[0]}
  if (( c > code )); then code=$c; fi
}
flags=(--release --no-default-features)

case "$group" in
  rest)
    run test "${flags[@]}" --no-fail-fast -p engine-schema -p engine-exec -p engine-scenes \
      -p engine-vehicle -p engine-fracture -p engine-vehiclegen
    ;;
  softbody)
    shard="${2:?shard}"
    total="${3:?shards}"
    plan="$(python3 "$scripts/host-test-shards.py" engine-softbody "$shard" "$total")" || exit 1
    tests=()
    examples=()
    doc=0
    while read -r kind arg; do
      case "$kind" in
        test) tests+=("$arg") ;;
        example) examples+=("--example=$arg") ;;
        doc) doc=1 ;;
        "") ;;
        *) echo "host-test-shards.py: unknown line '$kind $arg'" >&2; exit 1 ;;
      esac
    done <<< "$plan"
    if (( ${#tests[@]} )); then
      run test "${flags[@]}" --no-fail-fast -p engine-softbody "${tests[@]}"
    fi
    if (( doc )); then
      run test "${flags[@]}" --no-fail-fast -p engine-softbody --doc
    fi
    if (( ${#examples[@]} )); then
      run build "${flags[@]}" -p engine-softbody "${examples[@]}"
    fi
    ;;
  *)
    echo "usage: host-tests.sh rest | softbody N M" >&2
    exit 1
    ;;
esac

echo "$code" > host-tests.code
python3 "$scripts/expected-failures.py" --partial "$log" tests-expected-failures.txt "$code"
