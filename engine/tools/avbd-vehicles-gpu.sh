#!/usr/bin/env bash
# GPU cost of the AVBD vehicle study (engine/docs/avbd-vehicles.md §6): the existing
# GPU AVBD soft path (avbd_soft_bench) against the explicit vehicles in the World
# (vehicle_scale_bench fleet), on the ETK. One GPU job at a time (the GPU lock), each
# under a 10-minute kill timer. CSV to stdout.
#
#   engine/tools/avbd-vehicles-gpu.sh > out.csv
set -u
cd "$(dirname "$0")/.."
B=./target/release/examples
ETK="$PWD/crates/engine-webgpu/__mods/etkk_tex"
run() { perl -e 'alarm 600; exec @ARGV' "$@"; }
echo "# AVBD soft path (parked ETK grid, rigid core off)"
for cfg in "1 120 10" "1 120 20" "1 120 50" "1 240 20" "16 120 10" "16 120 20" "16 120 50" "64 120 10" "64 120 20" "64 120 50"; do
  # shellcheck disable=SC2086
  RUBBLE_ETK="$ETK" run "$B/avbd_soft_bench" $cfg 120 2>&1 | tail -1
done
echo "# explicit 2 kHz vehicles in the World (ETK kr6, raycast wheels, fleet driving)"
for n in 1 16 64; do
  VS_CAR=etkk RUBBLE_ETK="$ETK" VS_SLEEP=0 run "$B/vehicle_scale_bench" fleet "$n" 2>&1 | grep -v "^rubble:" | tail -2
done
