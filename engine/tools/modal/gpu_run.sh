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
#   --cpus N             CPU cores for the container (default 8: the driver compiles
#                        pipelines on the CPU, one core each, several at once)
#   --allow-cold-compile run even when this kernel version is new on the host's
#                        driver (minutes of compiling; without it the run stops
#                        at once with the estimate, exit 3)
#   --no-warm            skip the parallel pipeline warm-up before the selection
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
#   --fanout             one container per executable of the selection (each test
#                        binary), all at once; one summary and exit code
#   --fanout-tests       one container per test (names listed at build time), all
#                        at once: the shortest wall time for a suite
#   --matrix             the scenario matrix, one container per row (what
#                        tools/scenario-matrix/run.sh calls): the arguments after
#                        `--` are matrix.py's (SWITCH | --baseline | --make, rows)
#   --build-only         build and pack, do not call Modal
#   --keep               keep the run directory (engine/target/gpu-run/<id>)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"

gpu=T4 cpus=8 allow_cold=0 warm=1 cap=600 queue=300 monitor=30 shader_cache=1 repeat=1 timing=0 build_only=0 keep=0
fanout=""
sedan="$HOME/Development/rubble-artifacts/sedan-live/mods/sedan_new"
envs=() variants=() data=() pack_opts=()
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) gpu="$2"; shift 2 ;;
    --timeout) cap="$2"; shift 2 ;;
    --cpus) cpus="$2"; shift 2 ;;
    --allow-cold-compile) allow_cold=1; shift ;;
    --no-warm) warm=0; shift ;;
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
    --fanout) fanout=exe; shift ;;
    --fanout-tests) fanout=tests; pack_opts=(--list-tests); shift ;;
    --matrix) fanout=matrix; shift ;;
    --) shift; break ;;
    -h|--help) sed -n '2,52p' "$0"; exit 0 ;;
    *) echo "gpu_run: unknown option $1 (the cargo selection goes after --)" >&2; exit 2 ;;
  esac
done
[ $# -gt 0 ] || { echo "gpu_run: nothing to run; see --help" >&2; exit 2; }

# The scenario matrix: its own selection (the lab), data (matrix.py and the scenes),
# car and knobs; the arguments are matrix.py's. Its rows are picked here (matrix.py
# --rows) and each runs in a container of its own.
matrix_args=() matrix_rows=()
if [ "$fanout" = matrix ]; then
  matrix_args=("$@")
  matrix_py="$repo/engine/tools/scenario-matrix/matrix.py"
  rows_out="$(python3 "$matrix_py" --rows "$@")" || exit 2
  # shellcheck disable=SC2206
  matrix_rows=($rows_out)
  data+=(--data engine/tools/scenario-matrix)
  if [ -n "${CAR:-}" ]; then sedan="$CAR"; fi
  envs+=("CAR=@sedan")
  for k in FRAMES ROUNDS RETRIES THRESHOLD NOISE_MS WARM_MS PROFILE_SPLIT; do
    v="$(printenv "$k" || true)"
    if [ -n "$v" ]; then envs+=("$k=$v"); fi
  done
  mkdir -p "$repo/engine/out/scenario-matrix"
  set -- -p engine-world --example vehicle_lab
fi
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
#
# On Apple silicon (except the CUDA route: nvcc is x86_64-only) the build runs natively:
# an arm64 Linux image cross-compiling to x86_64 (engine/tools/modal/build), about 3x
# faster than the x86_64 image under Rosetta (vehicle_lab from empty: 112 s vs 328 s).
wt_key="$(printf '%s' "$repo" | shasum -a 256 | cut -c1-12)"
cuda_route=0
case " $* " in *" engine-cuda "*|*" -p engine-cuda"*) cuda_route=1 ;; esac
if [ "$(uname -m)" = arm64 ] && [ "$cuda_route" = 0 ]; then
  platform=linux/arm64
  image="rubble-gpurun-build:arm64-$(shasum -a 256 "$here/build/Dockerfile" | cut -c1-12)"
  image_dir="$here/build"
  target_vol="rubble-gpurun-arm-target-$wt_key"
else
  platform=linux/amd64
  image=rubble-cuda-build:slang-2026.19-cuda-12.8
  image_dir="$repo/engine/tools/cuda"
  target_vol="rubble-gpurun-target-$wt_key"
fi
if ! docker image inspect "$image" >/dev/null 2>&1; then
  echo "gpu_run: building $image (once; a minute or a few)" >&2
  docker build --platform "$platform" -q -t "$image" "$image_dir" >/dev/null
fi
t0=$(date +%s)
echo "gpu_run: building in Docker ($platform, target volume $target_vol)" >&2
# ENGINE_* variables reach the build (engine-cuda: ENGINE_CUDA_EXPLICIT_MODULES,
# ENGINE_CUDA_SASS, ...). The CUDA route embeds SASS for the GPU it runs on only.
case " $* " in *" engine-cuda "*|*" -p engine-cuda"*)
  if [ -z "${ENGINE_CUDA_SASS:-}" ]; then
    if [ "$gpu" = T4 ]; then export ENGINE_CUDA_SASS=75; else export ENGINE_CUDA_SASS=89; fi
  fi ;;
esac
build_env=()
while IFS='=' read -r k _; do
  case "$k" in ENGINE_*) build_env+=(-e "$k") ;; esac
done < <(env)
# Native builds use every core (the OS shares them with whatever else runs); the
# emulated one keeps its old default.
if [ "$platform" = linux/arm64 ]; then jobs_default=$(sysctl -n hw.ncpu 2>/dev/null || echo 8); rjobs_default=8
# (The CUDA route's build script runs nvcc a module a job: a kernel edit of the parity
# tests' 21 modules took 301 s at 4 jobs, 222 s at 8 and 211 s at 12, peak 6-7 GB, on a
# 16-core M3 Max with Docker given 16 cores and 25 GB; 2026-11-19.)
else jobs_default=6; rjobs_default=12; fi
docker run --rm --platform "$platform" ${build_env[@]+"${build_env[@]}"} \
  -v "$repo:/repo" \
  -v "$run:/out" \
  -v rubble-cuda-cargo:/cache/cargo \
  -v "$target_vol:/cache/target" \
  -e CARGO_HOME=/cache/cargo -e CARGO_TARGET_DIR=/cache/target \
  -e CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-$jobs_default}" -e RUBBLE_BUILD_JOBS="${RUBBLE_BUILD_JOBS:-$rjobs_default}" \
  "$image" python3 /repo/engine/tools/modal/gpu_run_pack.py /out ${pack_opts[@]+"${pack_opts[@]}"} ${data[@]+"${data[@]}"} -- "${args[@]}"
t1=$(date +%s)
echo "gpu_run: build + pack $((t1 - t0)) s, archive $(du -h "$run/archive.tar.gz" | cut -f1)" >&2
rm -rf "$run/stage"
if [ "$build_only" = 1 ]; then
  keep=1
  echo "gpu_run: --build-only: $run/archive.tar.gz" >&2
  exit 0
fi

# The run's spec (JSON) for gpu_run.py.
python3 - "$run/spec.json" "$gpu" "$cap" "$repeat" "$timing" "$monitor" "$queue" "$shader_cache" "$id" "$cpus" "$allow_cold" "$warm" \
  "$fanout" "${#matrix_args[@]}" ${matrix_args[@]+"${matrix_args[@]}"} "${#matrix_rows[@]}" ${matrix_rows[@]+"${matrix_rows[@]}"} \
  "${#envs[@]}" ${envs[@]+"${envs[@]}"} ${variants[@]+"${variants[@]}"} <<'PY'
import json, sys
out, gpu, cap, repeat, timing, monitor, queue, shader_cache, run_id, cpus, allow_cold, warm, fanout, *rest = sys.argv[1:]
na = int(rest.pop(0)); matrix_args, rest = rest[:na], rest[na:]
nr = int(rest.pop(0)); matrix_rows, rest = rest[:nr], rest[nr:]
n = int(rest.pop(0))
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
           "shader_cache": shader_cache == "1", "id": run_id, "cpus": int(cpus),
           "allow_cold": allow_cold == "1", "warm": warm == "1",
           "env": env, "variants": variants,
           "fanout": fanout, "matrix_args": matrix_args, "matrix_rows": matrix_rows}, open(out, "w"))
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
# Heartbeat for the Modal client (gpu_run.py's watcher): touched while this shell
# lives; the loop ends with it, even when the shell is SIGKILLed (no trap runs then),
# and the client then stops its app and containers within about a minute.
touch "$run/heartbeat"
( while kill -0 $$ 2>/dev/null; do touch "$run/heartbeat" 2>/dev/null; sleep 5; done ) &
trap 'stop_cli; cleanup' EXIT
trap 'exit 130' INT TERM
# The matrix's saved states (read to upload by hash; --make writes the new ones back).
states_mount=()
if [ "$fanout" = matrix ]; then
  states_mount=(-v "$repo/engine/out/scenario-matrix:/run-dir/states")
fi
set +e
docker run --rm --name "$cli_name" \
  -v "$run:/run-dir" ${states_mount[@]+"${states_mount[@]}"} \
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
