#!/usr/bin/env bash
# The vehicle lab's real-time check, each mode in turn (headless Chrome, one GPU job at a
# time): World steps/s, FPS, frame p50/p95 and the GPU split, then the lab's table.
#   engine/tools/vehicle-lab/probe-modes.sh [extra query, e.g. car=sedan]
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
EXTRA="${1:+&$1}"
for q in "shoot:script=shoot-sphere" "compactor:script=compactor-top" "crash:mode=crash" "mountain:mode=mountain" "drive:script=drive-wall"; do
  bash "$HERE/run.sh" --probe --secs "${SECS:-7}" "$q$EXTRA" 2>&1 | grep -E "ready|[357]s |plate|first contact|mountain seed|^beams|^penetration|rror"
done
