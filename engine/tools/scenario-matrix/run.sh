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
#   tools/scenario-matrix/run.sh --world [ROW ...]      the World's own cost a step (the rigid
#       step, the coupling's passes, the host) on the scenario rows: fails a row past
#       WORLD_MS (default 2.5 ms; about 30 s). Needs no saved states.
#   tools/scenario-matrix/run.sh --make [ROW ...]       (re)save the states into
#       out/scenario-matrix/ (minutes; once, and after a change of the buffers' layout)
#   FRAMES=n (default 10) frames a half, ROUNDS=r (default 3) the halves in turn; PROFILE_SPLIT=1 each row's frame by dispatch
#   kind (slower); PROFILE=fast|release (default fast); THRESHOLD=% and NOISE_MS=ms.
#
# Where it runs: the Mac's GPU, under the GPU lock (the default).
#   tools/scenario-matrix/run.sh --cuda SWITCH [ROW ...]   every row in one T4 container
#       (1 core, 2 GiB) on Modal, on CUDA (engine-cuda's explicit_replay; device code built
#       locally with nvcc, so a kernel edit costs a local build, ~4 min, then ~30 s and half
#       a cent on Modal; MODAL_OPTS="--pack N" spreads the rows over N containers). It
#       replays each row's portable fixture (out/scenario-matrix/portable/ROW.pfx: run
#       `python3 tools/scenario-matrix/matrix.py --make-portable` under the GPU lock once
#       after --make). Bit for bit holds within CUDA (ENGINE_CUDA_FMAD=0 for strict floats
#       when a switch runs other modules: nvcc fuses multiply-adds per module); timing is
#       directional (a T4, this host's launch overheads).
#   tools/scenario-matrix/run.sh --modal ...   opt-in: every row in a T4 container of its
#       own on Modal through Vulkan (tools/modal/gpu_run.sh --matrix; build in local
#       Docker). Not for kernel edits: NVIDIA compiles each new kernel version on Modal's
#       hosts (one explicit-beams pipeline took over 28 minutes, past the cap), so a
#       changed kernel costs dollars before a row runs. Use --cuda.
#   MODAL_OPTS="--gpu L4 --allow-cold-compile --timeout 1500" passes gpu_run options.
#
# Rows and how the states are made: docs/measurements/scenario-matrix.md.
here="$(cd "$(dirname "$0")" && pwd)"
engine="$(cd "$here/../.." && pwd)"
profile="${PROFILE:-fast}"
cd "$engine" || exit 1
if [ "$1" = "--cuda" ]; then
  shift
  # shellcheck disable=SC2086
  exec "$engine/tools/modal/gpu_run.sh" ${MODAL_OPTS:-} --matrix-cuda -- "$@"
fi
if [ "$1" = "--modal" ]; then
  shift
  # shellcheck disable=SC2086
  exec "$engine/tools/modal/gpu_run.sh" ${MODAL_OPTS:-} --matrix -- "$@"
fi
[ "$1" = "--local" ] && shift  # the default; kept for scripts that pass it
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
