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
# again costs nothing. Cargo's registry and the dependencies' builds are shared between
# commits (one target volume); the workspace's own crates rebuild for each commit (the
# tree's files are stamped with the extraction time, so nothing stale is reused). One
# build at a time on this machine (a lock directory).
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
lock="$repo/engine/target/gpu-run-refs/.lock"
mkdir -p "$repo/engine/target/gpu-run-refs"
waited=0
until mkdir "$lock" 2>/dev/null; do
  if [ "$waited" -ge 1800 ]; then echo "build_ref: $lock held for 30 min; remove it if no build runs" >&2; exit 2; fi
  [ "$waited" -eq 0 ] && echo "build_ref: waiting for another commit's build ($lock)" >&2
  sleep 5; waited=$((waited + 5))
done
tree="$repo/engine/target/gpu-run-refs/tree"
trap 'rm -rf "$lock" "$tree"' EXIT
rm -rf "$tree"
mkdir -p "$tree/fx" "$cache"
# -m: every file stamped now, newer than any build output in the shared target.
git -C "$repo" archive "$sha" | tar -x -m -C "$tree"
for f in "$@"; do cp "$f" "$tree/fx/"; done
image=rubble-cuda-build:slang-2026.19-cuda-12.8
if ! docker image inspect "$image" >/dev/null 2>&1; then
  docker build --platform linux/amd64 -q -t "$image" "$repo/engine/tools/cuda" >/dev/null
fi
build_env=()
while IFS='=' read -r k _; do
  case "$k" in ENGINE_CUDA_EXPLICIT_MODULES) ;; ENGINE_*) build_env+=(-e "$k") ;; esac
done < <(env)
echo "build_ref: building $ref (${sha:0:12}) in Docker (once per commit)" >&2
t0=$(date +%s)
docker run --rm --platform linux/amd64 ${build_env[@]+"${build_env[@]}"} \
  -v "$tree:/repo" -v "$cache:/out" \
  -v rubble-cuda-cargo:/cache/cargo -v rubble-gpurun-refs-target:/cache/target \
  -e CARGO_HOME=/cache/cargo -e CARGO_TARGET_DIR=/cache/target \
  -e CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-6}" -e RUBBLE_BUILD_JOBS="${RUBBLE_BUILD_JOBS:-12}" \
  "$image" bash -c '
    set -e
    cd /repo/engine
    mods=$(ENGINE_CUDA_EXPLICIT_MODULES=none cargo run -q --release -p engine-cuda --example explicit_replay -- --modules /repo/fx/*.pfx 2>/dev/null | tail -1) || mods=""
    echo "build_ref: modules ${mods:-(all: this commit has no --modules)}" >&2
    ENGINE_CUDA_EXPLICIT_MODULES="$mods" cargo build -q --release -p engine-cuda --example explicit_replay
    strip --strip-debug -o /out/explicit_replay /cache/target/release/examples/explicit_replay'
echo "build_ref: $ref built in $(( $(date +%s) - t0 )) s" >&2
mkdir -p "$(dirname "$out")"
cp "$cache/explicit_replay" "$out"
