#!/usr/bin/env bash
# Runs a command as the machine's one GPU job:
#
#   tools/webgpu-profile/gpu-run.sh [--limit SECONDS] -- command args...
#
# Through `scripts/with-gpu-lock.py`: the machine-wide flock on ~/.rubble/gpu.lock (every
# session's GPU jobs queue on it) and a hard kill of the whole process group after --limit
# (default 600 s). Native binaries take that flock themselves (`hold_gpu_lock`), so the
# child gets RUBBLE_GPU_LOCK=0 (the wrapper already holds it).
set -euo pipefail
ENGINE="$(cd "$(dirname "$0")/../.." && pwd)"
LIMIT=600
if [[ "${1:-}" == "--limit" ]]; then LIMIT="$2"; shift 2; fi
[[ "${1:-}" == "--" ]] && shift
exec python3 "$ENGINE/scripts/with-gpu-lock.py" --limit "$LIMIT" -- env RUBBLE_GPU_LOCK=0 "$@"
