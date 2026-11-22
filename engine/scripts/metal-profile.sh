#!/bin/sh
# Native Metal step profile: where every microsecond of a step goes
# (engine/docs/profiling-metal.md). Builds `metal_profile` and runs it as the machine's
# one GPU job (scripts/gpu-lock.sh) under a kill timer.
#
#   engine/scripts/metal-profile.sh [metal_profile args ...]
#     --scene "rigid:Brick Ring (9k)" --scene cars:8 --warm 60 --steps 60 --reps 7
#     --split 5 --no-replay --tag NAME --out DIR
#   LIMIT=seconds (default 600) kill timer; PROFILE=release|fast (default release).
#   RUBBLE_CAPTURE_STEP=N [RUBBLE_CAPTURE_OUT=out/x.gputrace]: also a .gputrace of step N.
#   RUBBLE_METAL_LABELS=1: debug groups in every command buffer (Instruments, captures).
#   RUBBLE_METAL_DEBUG_LIB=1: kernels with line tables and recorded source (shader profiler).
# Outputs: out/metal-profile[/TAG]/<scene>.{json,txt,trace.json} at the repository root.
here="$(cd "$(dirname "$0")" && pwd)"
engine="$(cd "$here/.." && pwd)"
profile="${PROFILE:-release}"
cd "$engine" || exit 1
[ "$profile" = release ] && flag="--release" || flag="--profile $profile"
# shellcheck disable=SC2086
cargo build $flag -p engine-scenes --bin metal_profile 2>&1 | grep -E "^error" -A12
bin="$engine/target/$profile/metal_profile"
[ -x "$bin" ] || { echo "no $bin" >&2; exit 1; }
exec "$here/gpu-lock.sh" --limit "${LIMIT:-600}" -- "$bin" "$@"
