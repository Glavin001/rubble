#!/bin/sh
# The scenario matrix: the gate for every perf change (AGENTS.md, "Checking a kernel
# change"). Saved states of a dozen use cases (one soft car driving, tumbling, into a
# wall; piles of 2, 8 and 32; car against car; car into rigid bricks and under rocks; car
# into a node-beam barrier; a loaded truss alone), each replayed with the switch on and
# off from the same state in the same build: every state word compared, each frame timed.
#
#   tools/scenario-matrix/run.sh SWITCH [ROW ...]   (from engine/) one table and a verdict:
#       PASS only if every row is bit for bit and none is slower past the noise threshold.
#       SWITCH: any `vehicle_lab --ab-switch` value (perf=MASK, groupsplit, wide, ...).
#   tools/scenario-matrix/run.sh --baseline [ROW ...]   each row's ms a frame and its target
#   tools/scenario-matrix/run.sh --make [ROW ...]       (re)save the states into
#       out/scenario-matrix/ (minutes; once, and after a change of the buffers' layout)
#   FRAMES=n (default 10) frames a half, ROUNDS=r (default 3) the halves in turn; PROFILE_SPLIT=1 each row's frame by dispatch
#   kind (slower); PROFILE=fast|release (default fast); THRESHOLD=% and NOISE_MS=ms.
#
# Where it runs: on Modal by default, every row in a GPU container of its own at once
# (tools/modal/gpu_run.sh --matrix; the build is local Docker, free). No local GPU and
# no lock: any number of agents at once. Each row's on and off halves share one
# container, so the verdict's speed comparison holds (directionally: a T4 through
# Vulkan, not the Mac); bit for bit is exact. The saved states upload once (by hash).
#   MODAL_OPTS="--gpu L4 --allow-cold-compile --timeout 1500" passes gpu_run options
#   (a kernel version new on Modal's driver needs --allow-cold-compile once).
#   tools/scenario-matrix/run.sh --local ...   the Mac's GPU instead (under the GPU lock):
#       for final Metal timings.
#
# Rows and how the states are made: docs/measurements/scenario-matrix.md.
here="$(cd "$(dirname "$0")" && pwd)"
engine="$(cd "$here/../.." && pwd)"
profile="${PROFILE:-fast}"
cd "$engine" || exit 1
if [ "$1" = "--local" ]; then
  shift
elif [ -z "$SCENARIO_MATRIX_LOCKED" ]; then
  # shellcheck disable=SC2086
  exec "$engine/tools/modal/gpu_run.sh" ${MODAL_OPTS:-} --matrix -- "$@"
fi
[ "$profile" = release ] && flag="--release" || flag="--profile $profile"
export LAB_BIN="${LAB_BIN:-$engine/target/$profile/examples/vehicle_lab}"
if [ -z "$SCENARIO_MATRIX_LOCKED" ]; then
  # shellcheck disable=SC2086
  cargo build $flag -p engine-world --example vehicle_lab 2>&1 | grep -E "^error" -A12
  [ -x "$LAB_BIN" ] || { echo "no $LAB_BIN"; exit 1; }
  # The machine's one GPU job at a time: the whole matrix under one hold of the lock.
  exec python3 "$engine/scripts/with-gpu-lock.py" --limit "${LIMIT:-590}" -- env RUBBLE_GPU_LOCK=0 SCENARIO_MATRIX_LOCKED=1 python3 "$here/matrix.py" "$@"
fi
exec python3 "$here/matrix.py" "$@"
