#!/usr/bin/env bash
# Runs a command as the machine's one GPU job, shared with other sessions and agents:
#
#   tools/webgpu-profile/gpu-run.sh [--limit SECONDS] -- command args...
#
# 1. Takes the session lock `~/.rubble-gpu-lock` (an atomic `mkdir`, the PID inside). A lock
#    whose PID is dead is stale and removed; otherwise waits on that PID (never by name).
# 2. Runs the command under `scripts/with-gpu-lock.py` (the repo's flock on
#    ~/.rubble/gpu.lock and a hard kill of the whole process group after --limit, default
#    600 s). Native binaries take that flock themselves (`hold_gpu_lock`), so the child
#    gets RUBBLE_GPU_LOCK=0 (the parent already holds it).
# 3. Removes the session lock on exit, failure or interrupt.
set -euo pipefail
ENGINE="$(cd "$(dirname "$0")/../.." && pwd)"
LIMIT=600
if [[ "${1:-}" == "--limit" ]]; then LIMIT="$2"; shift 2; fi
[[ "${1:-}" == "--" ]] && shift
LOCK="$HOME/.rubble-gpu-lock"
while ! mkdir "$LOCK" 2>/dev/null; do
  holder="$(cat "$LOCK/pid" 2>/dev/null || true)"
  if [[ -z "$holder" ]]; then
    sleep 1; holder="$(cat "$LOCK/pid" 2>/dev/null || true)"
  fi
  if [[ -z "$holder" ]] || ! kill -0 "$holder" 2>/dev/null; then
    echo "gpu-run: removing stale lock (pid ${holder:-none})" >&2
    rm -rf "$LOCK"
    continue
  fi
  echo "gpu-run: waiting for GPU job pid $holder" >&2
  while kill -0 "$holder" 2>/dev/null; do sleep 2; done
done
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT INT TERM
python3 "$ENGINE/scripts/with-gpu-lock.py" --limit "$LIMIT" -- env RUBBLE_GPU_LOCK=0 "$@"
