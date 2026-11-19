#!/usr/bin/env bash
# The baseline scenario set, browser and native, under ONE GPU-lock acquisition
# (gpu-run.sh; each run has its own alarm too). Writes out/webgpu-profile/<run>/.
#
#   tools/webgpu-profile/baseline.sh            # everything (about 12 minutes)
#   ONLY="b-lab8 n-lab8" tools/webgpu-profile/baseline.sh
#
# Build first: the release examples (vehicle_lab, brick_ring_drive_probe), the
# kernel_replay test, and the page wasm (docs/profiling-webgpu.md).
set -uo pipefail
ENGINE="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ "${BASELINE_LOCKED:-}" != 1 ]]; then
  exec env BASELINE_LOCKED=1 "$ENGINE/tools/webgpu-profile/gpu-run.sh" --limit "${LIMIT:-1080}" -- "$0" "$@"
fi
if [[ -s "$HOME/.nvm/nvm.sh" ]]; then . "$HOME/.nvm/nvm.sh"; cd "$ENGINE/.." && nvm use >/dev/null; fi
OUT="$ENGINE/../out/webgpu-profile"; mkdir -p "$OUT"
BIN="$ENGINE/target/release"
want() { [[ -z "${ONLY:-}" ]] || [[ " $ONLY " == *" $1 "* ]]; }
bounded() { local secs="$1"; shift; perl -e 'alarm shift; exec @ARGV' "$secs" "$@"; }
browser() { local name="$1"; shift; want "$name" || return 0; echo "== $name"; bounded 300 node "$ENGINE/crates/engine-webgpu/profile-webgpu.mjs" --name "$name" --limit 5 "$@" 2>&1 | tee "$OUT/$name.log" | tail -60; }
native() { local name="$1"; shift; want "$name" || return 0; echo "== $name"; mkdir -p "$OUT/$name"; (cd "$ENGINE" && bounded 240 env RUBBLE_HOST_PROFILE=1 RUBBLE_KERNEL_REPLAY_OUT="$OUT/$name/native.json" "$@") > "$OUT/$name/native.log" 2>&1; tail -45 "$OUT/$name/native.log"; }

if want test; then echo "== kernel replay test"; (cd "$ENGINE" && bounded 120 "$(ls -t "$BIN"/deps/kernel_replay-* | grep -v '\.d$' | head -1)" --nocapture 2>&1 | tail -30); fi

LAB='mode=mountain&car=sedan&res=low&cars=8&terrain=ledges'
browser b-lab8 --page vehicle-lab.html --query "$LAB&timing=step" --warm 6 --secs 10 --trace 3 --replay 15
browser b-lab8-phases --page vehicle-lab.html --query "$LAB&timing=phases" --warm 6 --secs 8 --trace 0
# The ring with sleep off (every brick awake, as the native probe runs it).
RING='car=sedan&auto=1&sleep=0'
browser b-ring --page brick-ring-drive.html --query "$RING&timing=step" --warm 4 --secs 10 --trace 3 --replay 15
browser b-ring-phases --page brick-ring-drive.html --query "$RING&timing=phases" --warm 4 --secs 8 --trace 0
browser b-ring-labels --page brick-ring-drive.html --query "$RING&timing=step" --warm 4 --secs 10 --trace 0 --labels

native n-lab8 env RUBBLE_KERNEL_REPLAY_AT=360 "$BIN/examples/vehicle_lab" --mode gpu --measure mountain-perf --cars 8 --terrain ledges --timing step --secs 10
native n-lab8-phases "$BIN/examples/vehicle_lab" --mode gpu --measure mountain-perf --cars 8 --terrain ledges --timing phases --secs 6
native n-ring env BR_CAR=showcase BR_SECS=6 RUBBLE_KERNEL_REPLAY_AT=240 "$BIN/examples/brick_ring_drive_probe"
native n-ring-phases env BR_CAR=showcase BR_SECS=6 BR_PHASES=1 "$BIN/examples/brick_ring_drive_probe"
native n-ring110k env BR_CAR=none BR_SECS=3 RUBBLE_KERNEL_REPLAY_AT=60 "$BIN/examples/brick_ring_drive_probe"
echo "baseline done: $OUT"
