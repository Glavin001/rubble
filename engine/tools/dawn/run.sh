#!/usr/bin/env bash
# The web World's physics on Dawn (Chrome's WebGPU implementation, in Node) on a Modal T4:
#   engine/tools/dawn/run.sh [--web DIR --car DIR] -- [world_ab.mjs args]
#
# By default this worktree's code is built on Modal's CPUs (remote_build.sh --web: the
# release web build's kernels and World parts, kernels embedded, and the generated sedan
# from this code), so this machine compiles nothing, and the bundle stays on Modal.
#   --web DIR   instead, pack an existing release web build (an engine/crates/engine-webgpu
#               directory with generated/world-wasm) ...
#   --car DIR   ... and this generated sedan export (files.json, JBeam, .pc)
#   args        --cars N --scene mountain|flat --terrain NAME --until S --ab MASK
#               --frames N --steps N --time-frames N
#
# Prints the harness's output live (each kernel's compile as the lab reports it) and its
# RESULT (engine/out/dawn/last.json). DAWN_NODE (default 22) picks Node's version;
# DAWN_NODE_FLAGS passes Node flags.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
engine="$(cd "$here/../.." && pwd)"
web="" car=""
while [ $# -gt 0 ]; do
  case "$1" in
    --web) web="$2"; shift 2 ;;
    --car) car="$2"; shift 2 ;;
    --) shift; break ;;
    *) break ;;
  esac
done
out="$engine/out/dawn"
mkdir -p "$out"
if [ -z "$web" ]; then
  "$engine/tools/modal/remote_build.sh" --web --result "$out/build.json"
  name="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["bundle"])' "$out/build.json")"
  bundle_path=""
else
  [ -f "$web/generated/world-wasm/engine_world_wasm_bg.wasm" ] || { echo "dawn/run.sh: no World wasm in $web/generated/world-wasm" >&2; exit 2; }
  [ -n "$car" ] && [ -f "$car/files.json" ] || { echo "dawn/run.sh: --web needs --car DIR (a generated sedan export)" >&2; exit 2; }
  stage="$out/stage"
  rm -rf "$stage"
  mkdir -p "$stage/bundle/fracture" "$stage/bundle/generated" "$stage/car"
  cp "$web/rubble-world.mjs" "$web/world-loop.mjs" "$stage/bundle/"
  cp "$web/fracture/gpu-wait.mjs" "$stage/bundle/fracture/"
  cp -R "$web/generated/world-wasm" "$stage/bundle/generated/"
  rm -f "$stage/bundle/generated/world-wasm/engine_world_wasm_bg.asyncify.wasm"
  (cd "$car" && python3 -c '
import json, os, shutil, sys
names = json.load(open("files.json"))
for f in ["files.json"] + [n for n in names if n.endswith((".jbeam", ".pc"))]:
    os.makedirs(os.path.dirname(os.path.join(sys.argv[1], f)) or sys.argv[1], exist_ok=True)
    shutil.copy(f, os.path.join(sys.argv[1], f))
' "$stage/car")
  (cd "$stage" && find bundle car -type f | LC_ALL=C sort > files.txt \
    && COPYFILE_DISABLE=1 tar -cf - -T files.txt | gzip -n > "$out/bundle.tar.gz")
  name="$(shasum -a 256 "$out/bundle.tar.gz" | cut -c1-24).tar.gz"
  bundle_path=/out/bundle.tar.gz
fi
python3 - "$out/spec.json" "$name" "$bundle_path" "$@" <<'PY'
import json, os, sys
out, name, path, *args = sys.argv[1:]
json.dump({"bundle": name, "bundle_path": path, "result_path": "/out/last.json", "args": args,
           "node_flags": os.environ.get("DAWN_NODE_FLAGS", "").split(),
           # DAWN_CAP: the harness's limit in seconds (default 900; the container's is 1800).
           "cap": int(os.environ.get("DAWN_CAP", "900"))}, open(out, "w"))
PY
docker image inspect rubble-modal-cli:1.6.2 >/dev/null 2>&1 || docker build -q -t rubble-modal-cli:1.6.2 "$engine/tools/modal" >/dev/null
docker run --rm --name "rubble-dawn-$$" -v "$out:/out" \
  -v "$here/dawn_run.py:/app/dawn_run.py:ro" -v "$here/world_ab.mjs:/app/world_ab.mjs:ro" -v "$here/loader.mjs:/app/loader.mjs:ro" \
  -v rubble-modal-config:/config:ro -e MODAL_CONFIG_PATH=/config/modal.toml -e DAWN_NODE="${DAWN_NODE:-22}" -w /app \
  rubble-modal-cli:1.6.2 run /app/dawn_run.py --spec /out/spec.json
