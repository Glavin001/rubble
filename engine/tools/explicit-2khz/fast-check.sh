#!/bin/sh
# The explicit vehicle kernels' fast check: a change replayed on saved moments of the
# lab's multi-car run, each way from the same state, every state word compared and both
# ways timed. Seconds a moment, not the minutes of driving up to it.
#
#   tools/explicit-2khz/fast-check.sh --make [CARS ...]     (from engine/; once, and again
#       after a change of the buffers' layout)  saves the fixtures: CARS cars (default
#       1 2 4 8) driven to the moments below, into out/fixtures/cars<N>_t<T>.fx (EXTRA:
#       more vehicle_lab options for the run, e.g. EXTRA="--wheels-bodies 0")
#   tools/explicit-2khz/fast-check.sh SWITCH [FIXTURE ...]
#       SWITCH: a `vehicle_lab --ab-switch` value (perf=<mask>, groupsplit, wide, sleep,
#       sleeplist, ...; vehicle sleep's own checks: sleep-check.sh).
#       FIXTURE: files or name parts (default: every fixture). One line a fixture:
#       its name, ms a frame with the switch on and off, and the verdict.
#   FRAMES=n (default 4) frames each way; PROFILE=fast|release (default fast: no
#   link-time optimisation, incremental; the same optimisation level and the same
#   kernels); EXTRA="..." more vehicle_lab options (--ab-profile, --ab-split-ablate ...).
#
# Bit-exact results (the engine-wide setting, off by default in the lab since 2026-11-19)
# are on here: every A/B compares the kernels as built, bit for bit. BIT_EXACT=0 runs
# the check in the fast setting (its timings only: there a lossless switch can still
# differ in the last bits, the compilers fusing the two paths' arithmetic differently).
#
# This is the first rung. It does not replace the suites (GPU == CPU reference), the
# Chrome validation or the whole runs: see AGENTS.md, "Checking a kernel change".
here="$(cd "$(dirname "$0")" && pwd)"
engine="$(cd "$here/../.." && pwd)"
fixtures="${FIXTURES:-$engine/out/fixtures}"
profile="${PROFILE:-fast}"
car="${CAR:-$HOME/Development/rubble-artifacts/sedan-live/mods/sedan_new}"
cd "$engine" || exit 1
export RUBBLE_BIT_EXACT="${BIT_EXACT:-1}"
[ "$profile" = release ] && flag="--release" || flag="--profile $profile"
# shellcheck disable=SC2086
bin="${LAB_BIN:-$engine/target/$profile/examples/vehicle_lab}"
# The machine's one GPU job at a time: the whole check under one hold of the lock (a hold
# a fixture would wait behind another job's minutes each time), the build before it.
if [ -z "$FAST_CHECK_LOCKED" ]; then
  cargo build $flag -p engine-world --example vehicle_lab 2>&1 | grep -E "^error" -A12
  [ -x "$bin" ] || { echo "no $bin"; exit 1; }
  exec python3 "$engine/scripts/with-gpu-lock.py" --limit "${LIMIT:-590}" -- env RUBBLE_GPU_LOCK=0 FAST_CHECK_LOCKED=1 sh "$0" "$@"
fi
lab() {
  cars="$1"; shift
  "$bin" --mode gpu --measure mountain-perf --cars "$cars" --car "$car" --rockslide 0 --self-split 1 --timing step --bit-exact "$RUBBLE_BIT_EXACT" "$@"
}
if [ "$1" = "--make" ]; then
  shift
  mkdir -p "$fixtures"
  for n in ${*:-1 2 4 8}; do
    case "$n" in
      1) at="2.5,6,20"; secs=20.3 ;;
      32) at="1,2.5,4"; secs=4.3 ;;
      # (Eight cars: also the pile at rest, where vehicle sleep acts.)
      8) at="2,3.7,8,16,20,30"; secs=30.3 ;;
      *) at="2,3.7,8,16"; secs=16.3 ;;
    esac
    # (AT="3,3.4,4.2,5" SECS=5.3: moments of one's own, e.g. the first collisions' fast
    # relative motion, into FIXTURES=<another directory>.)
    at="${AT:-$at}"; secs="${SECS:-$secs}"
    # shellcheck disable=SC2086
    lab "$n" --ab "$at" --ab-switch "perf=0" --ab-frames 1 --save-fixtures "$fixtures" --secs "$secs" $EXTRA 2>&1 | grep -E "fixture|^Error"
  done
  exit 0
fi
switch="${1:?usage: fast-check.sh --make | SWITCH [FIXTURE ...]}"
shift
list=""
if [ $# -eq 0 ]; then
  list=$(ls "$fixtures"/*.fx 2>/dev/null)
else
  for want in "$@"; do
    if [ -f "$want" ]; then list="$list $want"; else list="$list $(ls "$fixtures"/*"$want"*.fx 2>/dev/null)"; fi
  done
fi
[ -n "$list" ] || { echo "no fixtures in $fixtures: run fast-check.sh --make"; exit 1; }
bad=0
start=$(date +%s)
for f in $list; do
  name=$(basename "$f" .fx)
  n=$(echo "$name" | sed -E 's/cars([0-9]+)_.*/\1/')
  # shellcheck disable=SC2086
  out=$(lab "$n" --from-fixture "$f" --ab-switch "$switch" --ab-frames "${FRAMES:-4}" $EXTRA 2>&1)
  line=$(echo "$out" | grep "A/B at" | tail -1 | sed -E 's/^ +//')
  if echo "$line" | grep -q "bit for bit: true"; then
    echo "ok   $name: $line"
  else
    bad=1
    echo "FAIL $name: ${line:-no verdict}"
    echo "$out" | grep -E "differ|records|^Error|panicked" | head -6 | sed 's/^/       /'
  fi
  echo "$out" | grep -E "kept candidates \(on\)" | sed 's/^ */     /'
  [ -n "$EXTRA" ] && echo "$out" | grep -E "profile at|explicit |split ablation|untimed|group of|sleep \\(" | sed 's/^/     /'
done
echo "fast check: $(( $(date +%s) - start )) s"
exit $bad
