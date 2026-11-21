#!/usr/bin/env bash
# engine-cuda's explicit_replay built from a commit, for the two-build A/B on Modal
# (tools/scenario-matrix/run.sh --cuda --base REF [--head REF]; gpu_run.sh --base/--head).
#
#   engine/tools/modal/build_ref.sh REF OUT_FILE FIXTURE...   build (or reuse) and copy out
#   engine/tools/modal/build_ref.sh --list                    the builds kept, newest first
#   engine/tools/modal/build_ref.sh --prune N                 keep the newest N builds
#
# The library: engine/target/gpu-run-refs/<commit>-<settings>/ (gitignored), one
# directory a build, holding `explicit_replay` and `info.txt` (the commit, its subject and
# branches, the settings, the fixtures it was built for, when). A commit is built once
# for each settings and fixture set: ENGINE_CUDA_SASS (75 for the T4, 89 for the L4),
# ENGINE_CUDA_FMAD (strict floats or not), any other ENGINE_* setting, and the saved
# states (they decide which kernel modules get device code). Comparing two commits
# built before costs no build.
#
# Only commits: the working tree is built by gpu_run.sh itself and not kept (uncommitted
# code has no name). Commit an experiment, then compare it.
#
# The commit's tree comes from git archive (committed files; LFS files stay pointers,
# which the CUDA build does not read) and is built in the CUDA build image (x86_64,
# nvcc) with device code for the modules the fixtures need as that commit names them
# (its own `explicit_replay --modules`; every module when it predates that). Cargo's
# registry and the dependencies' builds are shared between commits (one target volume);
# the workspace's own crates rebuild for each (the tree's files are stamped with the
# extraction time, so nothing stale is reused). One build at a time on this machine.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
lib="$repo/engine/target/gpu-run-refs"

if [ "${1:-}" = "--list" ]; then
  [ -d "$lib" ] || { echo "no builds yet ($lib)"; exit 0; }
  found=0
  # shellcheck disable=SC2045
  for d in $(ls -t "$lib"); do
    [ -f "$lib/$d/info.txt" ] || continue
    found=1
    echo "== $lib/$d"
    sed 's/^/   /' "$lib/$d/info.txt"
  done
  [ "$found" = 1 ] || echo "no builds yet ($lib)"
  exit 0
fi
if [ "${1:-}" = "--prune" ]; then
  keep="${2:?--prune N}"
  [ -d "$lib" ] || exit 0
  n=0
  # shellcheck disable=SC2045
  for d in $(ls -t "$lib"); do
    [ -f "$lib/$d/info.txt" ] || continue
    n=$((n + 1))
    if [ "$n" -gt "$keep" ]; then echo "removing $d"; rm -rf "${lib:?}/$d"; fi
  done
  exit 0
fi

[ $# -ge 3 ] || { echo "usage: build_ref.sh REF OUT_FILE FIXTURE... | --list | --prune N" >&2; exit 2; }
ref="$1" out="$2"; shift 2
sha="$(git -C "$repo" rev-parse --verify "$ref^{commit}")" || { echo "build_ref: no commit $ref" >&2; exit 2; }
settings="$(env | grep '^ENGINE_' | grep -v '^ENGINE_CUDA_EXPLICIT_MODULES=' | sort | tr '\n' ' ')"
fixtures="$(shasum -a 256 "$@" | awk '{print substr($1,1,12) "  " $2}')"
key="$( { env | grep '^ENGINE_' | grep -v '^ENGINE_CUDA_EXPLICIT_MODULES=' | sort; shasum -a 256 "$@" | cut -c1-64; } | shasum -a 256 | cut -c1-12)"
cache="$lib/${sha:0:12}-$key"
write_info() {
  {
    echo "commit:   $sha"
    echo "subject:  $(git -C "$repo" log -1 --format=%s "$sha")"
    echo "branches: $(git -C "$repo" branch --all --contains "$sha" --format='%(refname:short)' 2>/dev/null | head -5 | tr '\n' ' ')"
    echo "asked as: $ref"
    echo "settings: ${settings:-(defaults)}"
    echo "modules:  $(cut -c1-200 "$cache/modules.txt" 2>/dev/null)"
    echo "fixtures:"
    printf '%s\n' "$fixtures" | sed 's/^/  /'
    echo "built:    $1"
  } > "$cache/info.txt"
}
if [ -x "$cache/explicit_replay" ]; then
  [ -f "$cache/info.txt" ] || write_info "(before info.txt was kept)"
  echo "build_ref: $ref (${sha:0:12}) from the library: $cache" >&2
  touch "$cache"
  mkdir -p "$(dirname "$out")"
  cp "$cache/explicit_replay" "$out"
  exit 0
fi
lock="$lib/.lock"
mkdir -p "$lib"
waited=0
until mkdir "$lock" 2>/dev/null; do
  if [ "$waited" -ge 1800 ]; then echo "build_ref: $lock held for 30 min; remove it if no build runs" >&2; exit 2; fi
  [ "$waited" -eq 0 ] && echo "build_ref: waiting for another commit's build ($lock)" >&2
  sleep 5; waited=$((waited + 5))
done
tree="$lib/.tree"
trap 'rm -rf "$lock" "$tree"' EXIT
rm -rf "$tree" "$cache"
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
echo "build_ref: building $ref (${sha:0:12}) in Docker (once per commit and settings)" >&2
t0=$(date +%s)
if ! docker run --rm --platform linux/amd64 ${build_env[@]+"${build_env[@]}"} \
  -v "$tree:/repo" -v "$cache:/out" \
  -v rubble-cuda-cargo:/cache/cargo -v rubble-gpurun-refs-target:/cache/target \
  -e CARGO_HOME=/cache/cargo -e CARGO_TARGET_DIR=/cache/target \
  -e CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-6}" -e RUBBLE_BUILD_JOBS="${RUBBLE_BUILD_JOBS:-12}" \
  "$image" bash -c '
    set -e
    cd /repo/engine
    mods=$(ENGINE_CUDA_EXPLICIT_MODULES=none cargo run -q --release -p engine-cuda --example explicit_replay -- --modules /repo/fx/*.pfx 2>/dev/null | tail -1) || mods=""
    echo "build_ref: modules ${mods:-(all: this commit has no --modules)}" >&2
    echo "$mods" > /out/modules.txt
    ENGINE_CUDA_EXPLICIT_MODULES="$mods" cargo build -q --release -p engine-cuda --example explicit_replay
    strip --strip-debug -o /out/explicit_replay /cache/target/release/examples/explicit_replay'; then
  rm -rf "$cache"
  echo "build_ref: the build of $ref failed" >&2
  exit 1
fi
secs=$(( $(date +%s) - t0 ))
write_info "$(date '+%Y-%m-%d %H:%M:%S') in ${secs} s"
echo "build_ref: $ref built in $secs s: $cache" >&2
mkdir -p "$(dirname "$out")"
cp "$cache/explicit_replay" "$out"
