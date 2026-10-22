#!/bin/zsh
# A machine-wide GPU queue for the learned-impact sessions: one MPS/Metal job at a time
# (training, timing, GPU tests). Atomic mkdir lock with the owner's pid; a dead owner's lock
# is taken over. Waits (polling every 20 s) until the GPU is free, runs the command, releases.
#   scripts/gpu_lock.sh <label> -- <command...>
set -u
LOCK=${RUBBLE_GPU_LOCK:-/tmp/rubble-gpu.lock}
label=$1; shift; [ "$1" = "--" ] && shift
while ! mkdir "$LOCK" 2>/dev/null; do
  owner=$(cat "$LOCK/pid" 2>/dev/null)
  if [ -n "$owner" ] && ! kill -0 "$owner" 2>/dev/null; then
    echo "[gpu-lock] taking over a stale lock of pid $owner ($(cat "$LOCK/label" 2>/dev/null))"
    rm -rf "$LOCK"; continue
  fi
  sleep 20
done
echo $$ > "$LOCK/pid"; echo "$label" > "$LOCK/label"; date +%s > "$LOCK/since"
trap 'rm -rf "$LOCK"' EXIT INT TERM
echo "[gpu-lock] $label holds the GPU"
"$@"
