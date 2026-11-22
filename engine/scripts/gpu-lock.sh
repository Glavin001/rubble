#!/bin/sh
# Run a command as the machine's one GPU job, bounded by a kill timer.
#
#   engine/scripts/gpu-lock.sh [--limit SECONDS] -- COMMAND [ARG ...]
#
# The machine-wide GPU lock is the flock on ~/.rubble/gpu.lock that native binaries take
# (hold_gpu_lock) and scripts/with-gpu-lock.py holds for a command; this goes through
# with-gpu-lock.py, which waits for the lock, runs COMMAND in its own process group and
# kills the group after --limit seconds (default 600). COMMAND runs with
# RUBBLE_GPU_LOCK=0 so its own binaries do not wait on the flock this run already holds.
# Expect to queue behind other sessions' GPU jobs: run it in the background.
limit=600
if [ "$1" = "--limit" ]; then limit="$2"; shift 2; fi
[ "$1" = "--" ] && shift
if [ $# -eq 0 ]; then sed -n '2,11p' "$0"; exit 2; fi
here="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$here/with-gpu-lock.py" --limit "$limit" -- env RUBBLE_GPU_LOCK=0 "$@"
