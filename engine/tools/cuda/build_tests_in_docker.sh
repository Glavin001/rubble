#!/usr/bin/env bash
# Build engine-cuda's test binaries in local Docker (free) and package them for
# the Modal GPU runner. Cargo's registry and target dir live in Docker volumes,
# so later builds are incremental.
#
#   engine/tools/cuda/build_tests_in_docker.sh [OUT]   # default engine/target/cuda-tests.tar.gz
#   ENGINE_CUDA_LOAD_CACHE=ca engine/tools/cuda/build_tests_in_docker.sh   # ENGINE_* pass through
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
out="${1:-engine/target/cuda-tests.tar.gz}"
image=rubble-cuda-build:slang-2026.19-cuda-12.8
if ! docker image inspect "$image" >/dev/null 2>&1; then
  echo "building $image (once; a few minutes)" >&2
  docker build --platform linux/amd64 -q -t "$image" "$here" >/dev/null
fi
envs=()
while IFS='=' read -r k _; do
  case "$k" in ENGINE_*) envs+=(-e "$k") ;; esac
done < <(env)
tty=()
if [ -t 1 ]; then tty=(-t); fi
start=$(date +%s)
docker run --rm --platform linux/amd64 ${tty[@]+"${tty[@]}"} ${envs[@]+"${envs[@]}"} \
  -v "$repo:/repo" \
  -v rubble-cuda-cargo:/cache/cargo \
  -v rubble-cuda-target:/cache/target \
  -e CARGO_HOME=/cache/cargo -e CARGO_TARGET_DIR=/cache/target \
  "$image" python3 engine/tools/modal/cuda_tests.py --package "$out"
echo "wrote $out ($(du -h "$repo/$out" | cut -f1)) in $(( $(date +%s) - start ))s" >&2
