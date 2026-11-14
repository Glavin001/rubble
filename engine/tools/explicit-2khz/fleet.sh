#!/bin/sh
# The fleet benchmark: N generated sedans in the World, GPU time of a 1/60 s World step
# (native wgpu, each step's span from submit to completion).
#   fleet.sh apart "1 8 32" [bench binary]   # nothing to hit: the per-car cost
#   fleet.sh town  "1 4 8 16" [bench binary] # the crash town: houses and car contact
# One GPU job at a time (the machine's lock) under a kill timer. Check the GPU is idle
# first (ioreg -r -d 1 -w 0 -c IOAccelerator | grep -o '"Device Utilization %"=[0-9]*'):
# a run that overlaps another GPU job reads ten times too high.
set -e
here="$(cd "$(dirname "$0")/../.." && pwd)"
mode="${1:-apart}"
counts="${2:-1 8 32}"
bin="${3:-$here/target/release/examples/explicit_vehicles_bench}"
car="${FLEET_CAR:-$HOME/Development/rubble-artifacts/sedan-live/mods/sedan_new:sedan/sedan.pc}"
extra=""
[ "$mode" = apart ] && extra="EV_NO_CONTACT=1 EV_NO_VV=1"
# shellcheck disable=SC2086
python3 "$here/scripts/with-gpu-lock.py" --limit 580 -- env RUBBLE_GPU_LOCK=0 EV_ETK="$car" EV_SPAN=1 \
  EV_SECS="${FLEET_SECS:-3}" $extra "$bin" $counts 2>&1 |
  grep -E "^[1-9][0-9]* vehicles|Error|limit" | sed -E 's/; wall.*(; [0-9]+ bonds)/\1/'
