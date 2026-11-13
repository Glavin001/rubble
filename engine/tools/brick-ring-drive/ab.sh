#!/bin/bash
# Interleaved A/B of the brick ring drive natively (one GPU job at a time; each run under
# a kill timer): the ring alone (sleep on, no ball, the drive's 25 kg bricks) against the
# ring with a car driving in it, twice each; each run's median GPU ms per World step after
# the first 30 steps, and with BR_PHASES=1 the segment means (rigid passes against the
# vehicles' coupling and frame).
#
#   bash engine/tools/brick-ring-drive/ab.sh [car] [secs]     # car: sedan (default) or etk
set -u
cd "$(dirname "$0")/../.."
CAR=${1:-sedan}
SECS=${2:-8}
PROBE=target/release/examples/brick_ring_drive_probe
run() {
  env BR_SLEEP=1 BR_BALL=0 BR_DENSITY=100 BR_SECS=$SECS BR_RUNUP=60 "$@" \
    perl -e 'alarm 600; exec @ARGV' $PROBE 2>&1 | grep -E "GPU per|segments|rror"
}
for k in 1 2; do
  echo "== ring alone ($k)"; run BR_CAR=none
  echo "== ring + $CAR ($k)"; run BR_CAR=$CAR
done
