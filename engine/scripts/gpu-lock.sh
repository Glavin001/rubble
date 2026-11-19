#!/bin/sh
# Run a command as the machine's one GPU job, bounded by a kill timer.
#
#   engine/scripts/gpu-lock.sh [--limit SECONDS] -- COMMAND [ARG ...]
#
# Takes both GPU locks the repo's tools use, then runs COMMAND:
#   1. ~/.rubble-gpu-lock, a directory made with mkdir (atomic) holding this script's
#      PID in ~/.rubble-gpu-lock/pid. The profiling sessions (Metal and WebGPU) share
#      it. If the directory exists and its PID is dead it is stale and removed; while
#      its PID lives, this script waits on that PID (kill -0 every 2 s), never on a
#      process name.
#   2. ~/.rubble/gpu.lock, the flock native binaries take (hold_gpu_lock), through
#      scripts/with-gpu-lock.py, which also kills COMMAND's process group after
#      --limit seconds (default 600). COMMAND runs with RUBBLE_GPU_LOCK=0 so its own
#      binaries do not wait on the flock this run already holds.
# The directory is removed when the run ends, also on failure or interrupt (trap).
limit=600
if [ "$1" = "--limit" ]; then limit="$2"; shift 2; fi
[ "$1" = "--" ] && shift
if [ $# -eq 0 ]; then sed -n '2,17p' "$0"; exit 2; fi
here="$(cd "$(dirname "$0")" && pwd)"
lock="${RUBBLE_GPU_LOCK_DIR:-$HOME/.rubble-gpu-lock}"
waited=0
while ! mkdir "$lock" 2>/dev/null; do
  holder="$(cat "$lock/pid" 2>/dev/null)"
  if [ -n "$holder" ] && ! kill -0 "$holder" 2>/dev/null; then
    echo "gpu-lock: removing stale $lock (pid $holder is gone)" >&2
    rm -rf "$lock"
    continue
  fi
  if [ -z "$holder" ]; then
    # Made but its PID not yet written: give the holder a moment, then call it stale.
    waited=$((waited + 1))
    if [ $waited -gt 5 ]; then rm -rf "$lock"; waited=0; fi
    sleep 1
    continue
  fi
  echo "gpu-lock: waiting for pid $holder ($lock)" >&2
  while kill -0 "$holder" 2>/dev/null; do sleep 2; done
done
echo $$ > "$lock/pid"
trap 'rm -rf "$lock"' EXIT
trap 'rm -rf "$lock"; exit 130' INT TERM HUP
python3 "$here/with-gpu-lock.py" --limit "$limit" -- env RUBBLE_GPU_LOCK=0 "$@"
