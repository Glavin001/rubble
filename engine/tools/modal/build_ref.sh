#!/usr/bin/env bash
# engine-cuda's explicit_replay built from a commit, for the two-build A/B on Modal
# (gpu_run.sh --base REF [--head REF]; tools/scenario-matrix/run.sh --cuda --base REF).
#
#   engine/tools/modal/build_ref.sh REF OUT_FILE FIXTURE...
#
# The commit's tree (git archive: committed files only, LFS files as pointers, which the
# CUDA build does not read) is built in the CUDA build image (x86_64, nvcc), with device
# code for the modules the fixtures need as that commit names them (its own
# `explicit_replay --modules`; every module when it predates that). Cached by commit,
# ENGINE_* settings and fixtures under engine/target/gpu-run-refs/: building the same base
# again costs nothing.
#
# Between commits only what changed is built again (README, "Build speed"): one tree for
# every commit and worktree (~/.cache/rubble-gpu-run-refs/tree), updated by content
# (rsync -c: a file whose bytes changed gets a new modification time, the others keep
# theirs), one cargo target for it, and nvcc's fatbins kept by content in the volume
# gpu_run.sh's CUDA builds share. So a commit that changed a kernel compiles that
# kernel's modules and links; one that changed a crate builds that crate and its
# dependents. One build at a time on this machine (a lock directory beside the tree).
# On Apple silicon the build runs natively (engine/tools/modal/build: aarch64 slangc and
# nvcc, the x86_64 binary cross-compiled); GPU_RUN_X86=1 uses the x86_64 image instead.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
[ $# -ge 3 ] || { echo "usage: build_ref.sh REF OUT_FILE FIXTURE..." >&2; exit 2; }
ref="$1" out="$2"; shift 2
sha="$(git -C "$repo" rev-parse --verify "$ref^{commit}")" || { echo "build_ref: no commit $ref" >&2; exit 2; }
key="$( { env | grep '^ENGINE_' | grep -v '^ENGINE_CUDA_EXPLICIT_MODULES=' | sort; shasum -a 256 "$@" | cut -c1-64; } | shasum -a 256 | cut -c1-12)"
cache="$repo/engine/target/gpu-run-refs/${sha:0:12}-$key"
if [ -x "$cache/explicit_replay" ]; then
  echo "build_ref: $ref (${sha:0:12}) cached" >&2
  mkdir -p "$(dirname "$out")"
  cp "$cache/explicit_replay" "$out"
  exit 0
fi
mkdir -p "$cache"
refs="${XDG_CACHE_HOME:-$HOME/.cache}/rubble-gpu-run-refs"
lock="$refs/.lock"
mkdir -p "$refs"
waited=0
until mkdir "$lock" 2>/dev/null; do
  if [ "$waited" -ge 1800 ]; then echo "build_ref: $lock held for 30 min; remove it if no build runs" >&2; exit 2; fi
  [ "$waited" -eq 0 ] && echo "build_ref: waiting for another commit's build ($lock)" >&2
  sleep 5; waited=$((waited + 5))
done
tree="$refs/tree" stage="$refs/stage"
trap 'rm -rf "$lock" "$stage"' EXIT
rm -rf "$stage"
mkdir -p "$stage" "$tree"
# The commit's files, then the tree updated by content: no -t, so a rewritten file is
# stamped now (newer than any build output) and an unchanged one keeps its time.
git -C "$repo" archive "$sha" | tar -x -C "$stage"
rsync -rlpc --delete --exclude /fx "$stage/" "$tree/"
rm -rf "$tree/fx" && mkdir -p "$tree/fx"
for f in "$@"; do cp "$f" "$tree/fx/"; done
if [ "$(uname -m)" = arm64 ] && [ "${GPU_RUN_X86:-0}" != 1 ]; then
  platform=linux/arm64
  image="rubble-gpurun-build:arm64-$(shasum -a 256 "$here/build/Dockerfile" | cut -c1-12)"
  image_dir="$here/build" target_vol=rubble-gpurun-refs-arm-target
  bin=/cache/target/x86_64-unknown-linux-gnu/release/examples/explicit_replay strip=x86_64-linux-gnu-strip
else
  platform=linux/amd64
  image=rubble-cuda-build:slang-2026.19-cuda-12.8
  image_dir="$repo/engine/tools/cuda" target_vol=rubble-gpurun-refs-x86-target
  bin=/cache/target/release/examples/explicit_replay strip=strip
fi
if ! docker image inspect "$image" >/dev/null 2>&1; then
  docker build --platform "$platform" -q -t "$image" "$image_dir" >/dev/null
fi
build_env=()
while IFS='=' read -r k _; do
  case "$k" in ENGINE_CUDA_EXPLICIT_MODULES|ENGINE_CUDA_FATBIN_CACHE) ;; ENGINE_*) build_env+=(-e "$k") ;; esac
done < <(env)
echo "build_ref: building $ref (${sha:0:12}) in Docker (once per commit)" >&2
t0=$(date +%s)
docker run --rm --platform "$platform" ${build_env[@]+"${build_env[@]}"} \
  -v "$tree:/repo" -v "$cache:/out" \
  -v rubble-cuda-cargo:/cache/cargo -v "$target_vol:/cache/target" \
  -v rubble-cuda-fatbins:/cache/fatbins -v "$here/fatbin_cache.py:/opt/rubble/fatbin_cache.py:ro" \
  -e CARGO_HOME=/cache/cargo -e CARGO_TARGET_DIR=/cache/target \
  -e ENGINE_CUDA_FATBIN_CACHE=/cache/fatbins -e ENGINE_CUDA_REQUIRE_PTX=1 \
  -e BIN="$bin" -e STRIP_BIN="$strip" \
  -e CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER="env LD_LIBRARY_PATH=/usr/x86_64-linux-gnu/lib" \
  -e CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 6)}" -e RUBBLE_BUILD_JOBS="${RUBBLE_BUILD_JOBS:-12}" \
  "$image" bash -c '
    set -e
    python3 /opt/rubble/fatbin_cache.py /cache/fatbins
    cd /repo/engine
    mods=$(ENGINE_CUDA_EXPLICIT_MODULES=none cargo run -q --release -p engine-cuda --example explicit_replay -- --modules /repo/fx/*.pfx 2>/dev/null | tail -1) || mods=""
    echo "build_ref: modules ${mods:-(all: this commit has no --modules)}" >&2
    ENGINE_CUDA_EXPLICIT_MODULES="$mods" cargo build -q --release -p engine-cuda --example explicit_replay
    "$STRIP_BIN" --strip-debug -o /out/explicit_replay "$BIN"'
echo "build_ref: $ref built in $(( $(date +%s) - t0 )) s" >&2
mkdir -p "$(dirname "$out")"
cp "$cache/explicit_replay" "$out"
