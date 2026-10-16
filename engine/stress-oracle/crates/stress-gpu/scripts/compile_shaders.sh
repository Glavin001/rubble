#!/usr/bin/env bash
# Compile every shaders/*.slang that has entry points to Metal Shading Language
# (shaders/generated/<name>.metal, run natively through wgpu's Metal passthrough) and
# WGSL (shaders/generated/<name>.wgsl, for WebGPU), with the pinned Slang compiler in
# Docker (tools/slang/Dockerfile). The output is checked in, so building the solver needs
# neither Slang nor Docker; build.rs turns the .metal into a .metallib with Xcode.
#   scripts/compile_shaders.sh           regenerate
#   scripts/compile_shaders.sh --check   fail if the checked-in output is stale
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=stress-gpu-slang:2026.19
if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
  docker build -q -t "$IMAGE" tools/slang > /dev/null
fi
out=shaders/generated
check=0
if [ "${1:-}" = "--check" ]; then
  check=1
  out=$(mktemp -d)
fi
mkdir -p "$out"
out=$(cd "$out" && pwd)
for src in shaders/*.slang; do
  name=$(basename "$src" .slang)
  # Modules imported by entry-point files (no [shader(...)] attribute) are not compiled alone.
  if ! grep -q '\[shader(' "$src"; then
    continue
  fi
  for target in metal wgsl; do
    docker run --rm -v "$PWD:/work" -v "$out:/out" "$IMAGE" \
      "/work/$src" -I /work/shaders -target "$target" -line-directive-mode none -o "/out/$name.$target"
  done
done
if [ "$check" = 1 ]; then
  if ! diff -r "$out" shaders/generated; then
    echo "generated shaders are stale: run scripts/compile_shaders.sh" >&2
    exit 1
  fi
  echo "generated shaders are current"
fi
