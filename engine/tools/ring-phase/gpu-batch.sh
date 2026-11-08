#!/bin/sh
# The ring_phase_probe GPU runs (diag/ring-phase-through), one after another, as one GPU
# job, from engine/:
#   scripts/with-gpu-lock.py --limit 900 -- env RUBBLE_GPU_LOCK=0 sh tools/ring-phase/gpu-batch.sh ../out/ring-phase
# CASES="name VAR=... ..." runs a subset (see the list below).
set -u
out=${1:-../out/ring-phase}
mkdir -p "$out"
bin=target/release/examples/ring_phase_probe
run() {
  name=$1; shift
  echo "== $name: $*"
  env "$@" perl -e 'alarm 240; exec @ARGV' $bin > "$out/$name.txt" 2>&1
  grep -E "worst|rigid control|GPU World flat|stopping" "$out/$name.txt"
}
if [ -n "${CASES:-}" ]; then
  eval "run $CASES"
  exit 0
fi
if [ "${SET:-1}" = 2 ]; then
  # The window-lag checks: a static brick (no motion), a 2.5 t brick, the World at 240 Hz.
  run brick1-5-static PP_SCENE=wall PP_KMH=5 PP_SECS=4 PP_RUNUP=1.0 PP_COLS=1 PP_ROWS=1 PP_DEEP=1 PP_STATIC=1
  run brick1-5-heavy PP_SCENE=wall PP_KMH=5 PP_SECS=4 PP_RUNUP=1.0 PP_COLS=1 PP_ROWS=1 PP_DEEP=1 PP_DENSITY=10000
  run brick1-5-240hz PP_SCENE=wall PP_KMH=5 PP_SECS=4 PP_RUNUP=1.0 PP_COLS=1 PP_ROWS=1 PP_DEEP=1 PP_HZ=240 PP_EVERY=60
  run brick1-5-nopush PP_SCENE=wall PP_KMH=5 PP_SECS=4 PP_RUNUP=1.0 PP_COLS=1 PP_ROWS=1 PP_DEEP=1 PP_PUSH=0
  run brick1-5-s1 PP_SCENE=wall PP_KMH=5 PP_SECS=4 PP_RUNUP=1.0 PP_COLS=1 PP_ROWS=1 PP_DEEP=1 PP_SURFACE=1
  run wall-5-240hz PP_SCENE=wall PP_KMH=5 PP_SECS=4 PP_RUNUP=1.0 PP_HZ=240 PP_EVERY=60
  run wall-5-static PP_SCENE=wall PP_KMH=5 PP_SECS=4 PP_RUNUP=1.0 PP_STATIC=1
  exit 0
fi
run wall-2-s0 PP_SCENE=wall PP_KMH=2 PP_SECS=8 PP_RUNUP=0.6
run wall-5-s0-nopush PP_SCENE=wall PP_KMH=5 PP_SECS=6 PP_RUNUP=1.0 PP_PUSH=0
run wall-5-s1 PP_SCENE=wall PP_KMH=5 PP_SECS=6 PP_RUNUP=1.0 PP_SURFACE=1
run wall-10-s0 PP_SCENE=wall PP_KMH=10 PP_SECS=6 PP_RUNUP=1.0
run wall-10-s1 PP_SCENE=wall PP_KMH=10 PP_SECS=6 PP_RUNUP=1.0 PP_SURFACE=1
run wall-5-s0-nosleep PP_SCENE=wall PP_KMH=5 PP_SECS=6 PP_RUNUP=1.0 PP_SLEEP=0
run brick1-5-s0 PP_SCENE=wall PP_KMH=5 PP_SECS=5 PP_RUNUP=1.0 PP_COLS=1 PP_ROWS=1 PP_DEEP=1
run wall-5-s0-trace PP_SCENE=wall PP_KMH=5 PP_SECS=2.2 PP_RUNUP=1.0 PP_EVERY=1
run rigid-5 PP_SCENE=rigid PP_KMH=5 PP_SECS=6 PP_RUNUP=1.0
run rigid-10 PP_SCENE=rigid PP_KMH=10 PP_SECS=5 PP_RUNUP=1.0
run loose-20-s0 PP_SCENE=loose PP_KMH=20 PP_SECS=4 PP_RUNUP=3
run loose-20-s1 PP_SCENE=loose PP_KMH=20 PP_SECS=4 PP_RUNUP=3 PP_SURFACE=1
run flat-38 PP_SCENE=flat PP_KMH=38 PP_SECS=14 PP_RUNUP=60
