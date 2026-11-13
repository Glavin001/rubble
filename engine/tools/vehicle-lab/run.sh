#!/usr/bin/env bash
# Runs the vehicle lab's recorder (engine-webgpu/vehicle-lab-record.mjs) as the machine's
# one GPU job, with the repo's Node: `engine/tools/vehicle-lab/run.sh [recorder args]`.
#
#   run.sh --probe shoot-sphere:car=sedan          # real-time check, console errors fail
#   run.sh shoot-box compactor-top mountain        # clips + stats into ~/Development/rubble-artifacts/vehicle-lab
#   run.sh --restarts crash:mode=crash             # in-place restart timings
set -euo pipefail
ENGINE="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ -s "$HOME/.nvm/nvm.sh" ]]; then
  # shellcheck source=/dev/null
  . "$HOME/.nvm/nvm.sh"
  (cd "$ENGINE/.." && nvm use >/dev/null)
  cd "$ENGINE/.." && nvm use >/dev/null
fi
exec python3 "$ENGINE/scripts/with-gpu-lock.py" --limit "${LIMIT:-900}" -- node "$ENGINE/crates/engine-webgpu/vehicle-lab-record.mjs" "$@"
