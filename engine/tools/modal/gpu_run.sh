#!/usr/bin/env bash
# Run wgpu tests and examples on a Modal cloud GPU (Vulkan on the NVIDIA driver)
# instead of this Mac's GPU: build in local Docker (x86_64), upload only the
# binaries and their small data files, run in a single-use container, stream the
# output and return the exit code. Safe to call from several worktrees at once.
# See engine/tools/modal/README.md.
#
#   engine/tools/modal/gpu_run.sh [options] -- <cargo selection> [-- <run args>]
#
#   engine/tools/modal/gpu_run.sh -- -p engine-webgpu --test explicit_beams_bitwise
#   engine/tools/modal/gpu_run.sh --variant old:RUBBLE_COUPLING_LIST=0 --variant new: \
#     -- -p engine-world --example vehicle_lab -- --mode gpu --measure mountain-perf --secs 4
#
# Options:
#   --gpu T4|L4|L40S     GPU (default T4; L40S only for a reason, ~3x the price)
#   --timeout SECS       hard cap on the container (default 600, max 1800)
#   --queue SECS         how long to wait for a free GPU before giving up (default
#                        300; queueing is not billed)
#   --env K=V            a variable for every run (repeatable)
#   --variant NAME:K=V,K2=V2   run the selection once per variant in the same
#                        container (repeatable; `NAME:` is the unchanged run)
#   --repeat N           run all variants N times (interleaved), for timing spread
#   --monitor SECS       a GPU/CPU use line every SECS while a process runs
#                        (default 30; 0 off)
#   --timing             print the directional-timing banner without variants
#   --no-shader-cache    compile every pipeline cold (the shared NVIDIA shader cache
#                        is neither read nor written)
#   --data PATH          an extra repo-relative file or directory to ship (small)
#   --sedan DIR          the generated sedan's JBeam export (default
#                        ~/Development/rubble-artifacts/sedan-live/mods/sedan_new);
#                        shipped when an argument says @sedan, which becomes its
#                        path in the container (/scenarios/sedan_new)
#   --build-only         build and pack, do not call Modal
#   --keep               keep the run directory (engine/target/gpu-run/<id>)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"

gpu=T4 cap=600 queue=300 monitor=30 shader_cache=1 repeat=1 timing=0 build_only=0 keep=0
sedan="$HOME/Development/rubble-artifacts/sedan-live/mods/sedan_new"
envs=() variants=() data=()
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) gpu="$2"; shift 2 ;;
    --timeout) cap="$2"; shift 2 ;;
    --queue) queue="$2"; shift 2 ;;
    --env) envs+=("$2"); shift 2 ;;
    --variant) variants+=("$2"); shift 2 ;;
    --repeat) repeat="$2"; shift 2 ;;
    --monitor) monitor="$2"; shift 2 ;;
    --timing) timing=1; shift ;;
    --no-shader-cache) shader_cache=0; shift ;;
    --data) data+=(--data "$2"); shift 2 ;;
    --sedan) sedan="$2"; shift 2 ;;
    --build-only) build_only=1; shift ;;
    --keep) keep=1; shift ;;
    --) shift; break ;;
    -h|--help) sed -n '2,39p' "$0"; exit 0 ;;
    *) echo "gpu_run: unknown option $1 (the cargo selection goes after --)" >&2; exit 2 ;;
  esac
done
[ $# -gt 0 ] || { echo "gpu_run: nothing to run; see --help" >&2; exit 2; }
case "$gpu" in T4|L4|L40S) ;; *) echo "gpu_run: --gpu must be T4, L4 or L40S" >&2; exit 2 ;; esac
case "$cap" in ''|*[!0-9]*) echo "gpu_run: --timeout takes seconds" >&2; exit 2 ;; esac
if [ "$cap" -lt 30 ] || [ "$cap" -gt 1800 ]; then echo "gpu_run: --timeout must be 30..1800 s" >&2; exit 2; fi

# The adapter probe every run starts with (engine-webgpu's gpu_adapter bin): a branch
# cut before it was added must merge or cherry-pick it first.
if ! grep -q '^name = "gpu_adapter"' "$repo/engine/crates/engine-webgpu/Cargo.toml" \
   || [ ! -f "$repo/engine/crates/engine-webgpu/src/bin/gpu_adapter.rs" ]; then
  echo "gpu_run: this branch lacks engine-webgpu's gpu_adapter bin (src/bin/gpu_adapter.rs and its [[bin]] entry in engine/crates/engine-webgpu/Cargo.toml); merge or cherry-pick it from tools/modal-gpu-runner" >&2
  exit 2
fi

# One directory per invocation: nothing shared between concurrent runs.
id="$(date +%Y%m%d-%H%M%S)-$$-$RANDOM"
run="$repo/engine/target/gpu-run/$id"
mkdir -p "$run"
cleanup() { if [ "$keep" = 0 ]; then rm -rf "$run"; fi; }
trap cleanup EXIT

# `@sedan`: stage the export's JBeam text only (files.json, *.jbeam, *.pc; never the
# .glb/.rskin) at the container path, and point the arguments there.
args=()
need_sedan=0
for a in "$@"; do
  case "$a" in *@sedan*) need_sedan=1; a="${a//@sedan//scenarios/sedan_new}" ;; esac
  args+=("$a")
done
for e in ${envs[@]+"${envs[@]}"} ${variants[@]+"${variants[@]}"}; do
  case "$e" in *@sedan*) need_sedan=1 ;; esac
done
if [ "$need_sedan" = 1 ]; then
  [ -f "$sedan/files.json" ] || { echo "gpu_run: no JBeam export at $sedan (--sedan DIR)" >&2; exit 2; }
  dest="$run/stage/scenarios/sedan_new"
  mkdir -p "$dest"
  cp "$sedan/files.json" "$dest/"
  (cd "$sedan" && find . -type f \( -name '*.jbeam' -o -name '*.pc' \) -print0 \
    | while IFS= read -r -d '' f; do mkdir -p "$dest/$(dirname "$f")"; cp "$f" "$dest/$f"; done)
fi

# Build: x86_64 Ubuntu 22.04 (the CUDA build image: Rust + slangc), cargo's target in a
# volume of this worktree's own (two worktrees share /repo as their path inside the
# container, so a shared target dir could take one's stale outputs for the other's).
image=rubble-cuda-build:slang-2026.19-cuda-12.8
if ! docker image inspect "$image" >/dev/null 2>&1; then
  echo "gpu_run: building $image (once; a few minutes)" >&2
  docker build --platform linux/amd64 -q -t "$image" "$repo/engine/tools/cuda" >/dev/null
fi
wt_key="$(printf '%s' "$repo" | shasum -a 256 | cut -c1-12)"
t0=$(date +%s)
echo "gpu_run: building in Docker (target volume rubble-gpurun-target-$wt_key)" >&2
docker run --rm --platform linux/amd64 \
  -v "$repo:/repo" \
  -v "$run:/out" \
  -v rubble-cuda-cargo:/cache/cargo \
  -v "rubble-gpurun-target-$wt_key:/cache/target" \
  -e CARGO_HOME=/cache/cargo -e CARGO_TARGET_DIR=/cache/target \
  -e CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-6}" -e RUBBLE_BUILD_JOBS="${RUBBLE_BUILD_JOBS:-6}" \
  "$image" python3 /repo/engine/tools/modal/gpu_run_pack.py /out ${data[@]+"${data[@]}"} -- "${args[@]}"
t1=$(date +%s)
echo "gpu_run: build + pack $((t1 - t0)) s, archive $(du -h "$run/archive.tar.gz" | cut -f1)" >&2
rm -rf "$run/stage"
if [ "$build_only" = 1 ]; then
  keep=1
  echo "gpu_run: --build-only: $run/archive.tar.gz" >&2
  exit 0
fi

# The run's spec (JSON) for gpu_run.py.
python3 - "$run/spec.json" "$gpu" "$cap" "$repeat" "$timing" "$monitor" "$queue" "$shader_cache" "$id" \
  "${#envs[@]}" ${envs[@]+"${envs[@]}"} ${variants[@]+"${variants[@]}"} <<'PY'
import json, sys
out, gpu, cap, repeat, timing, monitor, queue, shader_cache, run_id, n, *rest = sys.argv[1:]
n = int(n)
def kv(spec):
    # K=V,K2=V2; a comma part without `=` belongs to the value before it (EV_AB_MASKS=1,2).
    parts = []
    for p in spec.split(","):
        if "=" in p or not parts:
            parts.append(p)
        else:
            parts[-1] += "," + p
    d = {}
    for part in filter(None, (p.strip() for p in parts)):
        k, eq, v = part.partition("=")
        if not eq or not k:
            raise SystemExit(f"gpu_run: {part!r} is not K=V")
        d[k] = v.replace("@sedan", "/scenarios/sedan_new")
    return d
env = {}
for e in rest[:n]:
    env.update(kv(e))
variants = []
for v in rest[n:]:
    name, colon, spec = v.partition(":")
    if not colon:
        raise SystemExit(f"gpu_run: --variant {v!r} is not NAME:K=V,...")
    variants.append({"name": name, "env": kv(spec)})
json.dump({"gpu": gpu, "cap": int(cap), "repeat": int(repeat), "timing": timing == "1", "monitor": int(monitor), "queue": int(queue),
           "shader_cache": shader_cache == "1", "id": run_id,
           "env": env, "variants": variants}, open(out, "w"))
PY

# The Modal CLI in Docker with only this run's directory and the runner script
# mounted: nothing else of the machine is reachable or uploaded. The login volume is
# read-only.
cli=rubble-modal-cli:1.6.2
if ! docker image inspect "$cli" >/dev/null 2>&1; then
  docker build -q -t "$cli" "$here" >/dev/null
fi
# Named, so an interrupted gpu_run.sh stops its own Modal call (SIGINT: the Modal
# CLI then stops the app and its container) instead of leaving it to the cap.
cli_name="rubble-gpurun-$id"
stop_cli() {
  if docker inspect "$cli_name" >/dev/null 2>&1; then
    docker kill --signal INT "$cli_name" >/dev/null 2>&1 || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      docker inspect "$cli_name" >/dev/null 2>&1 || break
      sleep 1
    done
    docker rm -f "$cli_name" >/dev/null 2>&1 || true
  fi
}
trap 'stop_cli; cleanup' EXIT
trap 'exit 130' INT TERM
set +e
docker run --rm --name "$cli_name" \
  -v "$run:/run-dir" \
  -v "$here/gpu_run.py:/app/gpu_run.py:ro" \
  -v rubble-modal-config:/config:ro \
  -e MODAL_CONFIG_PATH=/config/modal.toml \
  -w /app \
  "$cli" run /app/gpu_run.py --archive /run-dir/archive.tar.gz --spec /run-dir/spec.json
cli_code=$?
set -e
code=$cli_code
if [ -f "$run/exit_code" ]; then code=$(cat "$run/exit_code"); fi
t2=$(date +%s)
echo "gpu_run: modal $((t2 - t1)) s, total $((t2 - t0)) s, exit $code" >&2
exit "$code"
