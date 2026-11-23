#!/usr/bin/env bash
# Build a cargo selection on Modal's CPUs instead of this machine (remote_build.py):
# only the source files that changed are sent; the packed archive lands in the GPU
# runner's archive volume (rubble-gpu-run-archives), never on this machine.
#
#   engine/tools/modal/remote_build.sh [--data PATH]... [--stage DIR] [--result FILE] \
#     -- <cargo selection>
#
#   --data PATH    passed to gpu_run_pack.py (a repo-relative file or directory to ship)
#   --stage DIR    files from outside the repo to ship, laid out at their container paths
#                  (gpu_run.sh's OUT/stage: the car's JBeam export)
#   --result FILE  where to write the result (archive key, manifest, timings) as JSON
#
# ENGINE_* variables reach the build (ENGINE_CUDA_SASS, ENGINE_CUDA_FMAD,
# ENGINE_CUDA_EXPLICIT_MODULES, ...). The tree key is this worktree's path: each worktree
# has its own source copy and cargo target on Modal, and all share the kernel caches.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
pack_opts=() stage="" result_out=""
while [ $# -gt 0 ]; do
  case "$1" in
    --data) pack_opts+=(--data "$2"); shift 2 ;;
    --stage) stage="$2"; shift 2 ;;
    --result) result_out="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "remote_build: unknown option $1" >&2; exit 2 ;;
  esac
done
[ $# -gt 0 ] || { echo "usage: remote_build.sh [--data P] [--stage DIR] [--result F] -- <cargo selection>" >&2; exit 2; }
key="$(printf '%s' "$repo" | shasum -a 256 | cut -c1-12)"
run="$repo/engine/target/remote-build/$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$run"
t0=$(date +%s)
# The tree: every tracked file under engine/ and every untracked one not ignored (two
# lists: `-co --exclude-standard` together drops tracked files such as Cargo.lock), except
# the large assets no build reads. sha256 a file.
(cd "$repo" && { git ls-files -z -- engine; git ls-files -z -o --exclude-standard -- engine
  # Cargo.lock is gitignored in this repo: the worktree's own lockfile, so the remote
  # build resolves the same crate versions as a local one.
  [ -f engine/Cargo.lock ] && printf 'engine/Cargo.lock\0'; } \
  | python3 -c '
import hashlib, os, sys
for p in sys.stdin.buffer.read().split(b"\0"):
    if not p:
        continue
    q = p.decode()
    try:
        st = os.stat(q)
    except FileNotFoundError:
        continue
    # (3D models and Blender files: no build reads them, and they were 100 of 147 MB.)
    if st.st_size > 8 << 20 or q.endswith((".glb", ".blend", ".blend1")):
        continue
    print(hashlib.sha256(open(q, "rb").read()).hexdigest() + "\t" + q)
') > "$run/manifest.tsv"
echo "remote_build: $(wc -l < "$run/manifest.tsv" | tr -d ' ') files hashed in $(( $(date +%s) - t0 )) s" >&2
env_json="$(python3 -c 'import json, os; print(json.dumps({k: v for k, v in os.environ.items() if k.startswith("ENGINE_")}))')"
stage_mount=()
if [ -n "$stage" ]; then stage_mount=(-v "$stage:/stage-local:ro"); fi
python3 - "$run/spec.json" "$key" "$env_json" "${#pack_opts[@]}" ${pack_opts[@]+"${pack_opts[@]}"} "$([ -n "$stage" ] && echo /stage-local)" "$@" <<'PY'
import json, sys
out, key, env, n, *rest = sys.argv[1:]
n = int(n)
pack_opts, rest = rest[:n], rest[n:]
stage, sel = rest[0], rest[1:]
import os
# RB_MODULES_FROM: repo-relative portable fixtures, one a line (gpu_run.sh --matrix-cuda
# sets it): the builder lists the kernel modules they launch (explicit_replay --modules).
modules_from = [l for l in os.environ.get("RB_MODULES_FROM", "").splitlines() if l]
json.dump({"key": key, "root": "/repo-local", "manifest": "/run-dir/manifest.tsv", "result": "/run-dir/result.json",
           "env": json.loads(env), "selection": sel, "pack_opts": pack_opts, "stage_dir": stage,
           "modules_from_local": modules_from}, open(out, "w"))
PY
cli=rubble-modal-cli:1.6.2
docker image inspect "$cli" >/dev/null 2>&1 || docker build -q -t "$cli" "$here" >/dev/null
mounts=(-v "$run:/run-dir" -v "$repo:/repo-local:ro" -v "$repo/engine/tools/cuda:/ctx/cuda:ro"
  -v "$here/remote_build.py:/app/remote_build.py:ro" ${stage_mount[@]+"${stage_mount[@]}"}
  -v rubble-modal-config:/config:ro -e MODAL_CONFIG_PATH=/config/modal.toml -w /app)
# The build function is a deployed app (a warm container outlives one build and serves
# the next); deployed again when remote_build.py or the build image's sources change.
deployed="$repo/engine/target/remote-build/deployed.sha"
want="$(cat "$here/remote_build.py" "$repo/engine/tools/cuda/Dockerfile" "$repo/engine/tools/cuda/install_linux_toolchain.sh" | shasum -a 256 | cut -c1-16)"
if [ "$(cat "$deployed" 2>/dev/null)" != "$want" ]; then
  echo "remote_build: deploying the build function (its code changed)" >&2
  docker run --rm "${mounts[@]}" "$cli" deploy /app/remote_build.py >&2
  echo "$want" > "$deployed"
fi
docker run --rm --name "rubble-rb-$$" "${mounts[@]}" --entrypoint python "$cli" \
  /app/remote_build.py --spec /run-dir/spec.json
echo "remote_build: total $(( $(date +%s) - t0 )) s" >&2
if [ -n "$result_out" ]; then cp "$run/result.json" "$result_out"; fi
