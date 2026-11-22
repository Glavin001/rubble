#!/usr/bin/env bash
# Profiles a page in headless Chrome as the machine's one GPU job (gpu-run.sh: the session
# lock, the repo's flock, a hard time limit), with the repo's Node:
#
#   tools/webgpu-profile/run.sh --page vehicle-lab.html --query 'mode=mountain&cars=8&timing=step' --name lab8
#   tools/webgpu-profile/run.sh --page brick-ring-drive.html --query 'car=sedan&timing=phases' --replay 15 --name ring
#
# Writes out/webgpu-profile/<name>/{summary.json,table.txt,trace.json} (see
# crates/engine-webgpu/profile-webgpu.mjs for every option). Compare two runs with
# tools/webgpu-profile/diff.mjs A/summary.json B/summary.json. Build the page's wasm first
# (docs/profiling-webgpu.md).
set -euo pipefail
ENGINE="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ -s "$HOME/.nvm/nvm.sh" ]]; then
  # shellcheck source=/dev/null
  . "$HOME/.nvm/nvm.sh"
  cd "$ENGINE/.." && nvm use >/dev/null
fi
exec "$ENGINE/tools/webgpu-profile/gpu-run.sh" --limit "${LIMIT:-660}" -- node "$ENGINE/crates/engine-webgpu/profile-webgpu.mjs" "$@"
