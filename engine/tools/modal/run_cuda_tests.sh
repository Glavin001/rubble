#!/usr/bin/env bash
# engine-cuda GPU tests with only the GPU seconds on Modal: build the test
# binaries in local Docker (free), then run them on a single-use Modal GPU.
# A run is skipped when the same archive already passed with the same
# arguments (engine/target/cuda-green); --force runs it anyway.
#
# Arguments go to cuda_tests.py (--gpu, --perf, --filter, --sync, --sanitize,
# --env). Build-time switches (ENGINE_CUDA_LOAD_CACHE, ...) come from the
# environment.
#
#   engine/tools/modal/run_cuda_tests.sh
#   engine/tools/modal/run_cuda_tests.sh --perf --gpu L40S
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here/../../.."
force=0
args=()
for a in "$@"; do
  if [ "$a" = "--force" ]; then force=1; else args+=("$a"); fi
done
archive=engine/target/cuda-tests.tar.gz
engine/tools/cuda/build_tests_in_docker.sh "$archive"

# Identity of this run: the binaries, the arguments and build-time ENGINE_* vars.
key=$( { shasum -a 256 "$archive" | cut -d' ' -f1; printf '%s\n' ${args[@]+"${args[@]}"}; env | grep '^ENGINE_' | sort || true; } | shasum -a 256 | cut -d' ' -f1)
green=engine/target/cuda-green
if [ "$force" = 0 ] && [ -f "$green" ] && grep -qx "$key" "$green"; then
  echo "skipped: this archive already passed with these arguments (--force to rerun)"
  exit 0
fi
engine/tools/modal/modal.sh run engine/tools/modal/cuda_tests.py --archive "$archive" ${args[@]+"${args[@]}"}
echo "$key" >> "$green"
