#!/bin/sh
# Native GPU split of the ETK in the brick ring around the strike (the after-wall cost):
# brick_ring_drive_probe with the car spawned 20 m out at 30 m/s, 4 s, BR_PHASES=1 (segment
# means over the run), n = 1 then n = 2; each run under tools/explicit-2khz/bounded.sh
# (the GPU lock, a 600 s kill). From engine/ after
#   cargo build --release -p engine-world --example brick_ring_drive_probe
cd "$(dirname "$0")/../.." || exit 1
for n in ${NS:-1 2}; do
  echo "== native ETK n=$n, struck at 30 m/s"
  BR_CAR=etk BR_N=$n BR_SLEEP=1 BR_BALL=0 BR_DENSITY=100 BR_RUNUP=${RUNUP:-20} BR_SPEED=${SPEED:-30} \
    BR_SECS=${SECS:-4} BR_FAR=1 BR_PHASES=1 \
    sh tools/explicit-2khz/bounded.sh 600 target/release/examples/brick_ring_drive_probe 2>&1 |
    grep -E "GPU per|segments|rror|t [0-9.]+ s|stopping" | cut -c1-600
done
