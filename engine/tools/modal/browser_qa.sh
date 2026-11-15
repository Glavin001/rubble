#!/usr/bin/env bash
# Browser QA of a deployed page (the vehicle lab) on a Modal NVIDIA GPU: headless
# Chrome, WebGPU on the driver's Vulkan, the lab's probe protocol
# (engine/crates/engine-webgpu/vehicle-lab-record.mjs --qa). See README.md.
#
#   engine/tools/modal/browser_qa.sh --base https://<deployment> --share '<share link>' \
#     --version <commit> --out <dir> [--gpu T4] [--timeout 1800] [--mods __mods/sedan_new] \
#     -- 'mountain-gen:mode=mountain&car=gen' --secs 45 --stills 0.3,4,15,30 --beams-at 31 \
#     [--- next step ...]
#
# Steps are separated by `---`; each is the recorder's `--qa` arguments. The share link
# (Vercel Authentication) is passed to Modal only, never printed; the report masks it.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
gpu=T4 cap=1800 base= share= version= out= mods=
while [ $# -gt 0 ]; do
  case "$1" in
    --base) base="$2"; shift 2 ;;
    --share) share="$2"; shift 2 ;;
    --version) version="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    --gpu) gpu="$2"; shift 2 ;;
    --timeout) cap="$2"; shift 2 ;;
    --mods) mods="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "browser_qa: unknown option $1" >&2; exit 2 ;;
  esac
done
[ -n "$base" ] && [ -n "$share" ] && [ -n "$version" ] && [ -n "$out" ] || { echo "browser_qa: --base, --share, --version and --out are required" >&2; exit 2; }
if [ "$cap" -gt 1800 ]; then echo "browser_qa: --timeout at most 1800" >&2; exit 2; fi
id="$(date +%Y%m%d-%H%M%S)-$$"
run="$repo/engine/target/browser-qa/$id"
mkdir -p "$run/out" "$out"
cp "$repo/engine/crates/engine-webgpu/vehicle-lab-record.mjs" "$run/"
python3 - "$run/spec.json" "$gpu" "$cap" "$base" "$share" "$version" "$mods" "$id" "$@" <<'PY'
import json, sys
out, gpu, cap, base, share, version, mods, rid, *rest = sys.argv[1:]
steps, cur = [], []
for a in rest:
    if a == "---":
        steps.append(cur); cur = []
    else:
        cur.append(a)
if cur:
    steps.append(cur)
json.dump({"gpu": gpu, "cap": int(cap), "base": base.rstrip("/"), "share": share, "version": version,
           "mods": mods, "id": rid, "steps": steps, "script_path": "/run-dir/vehicle-lab-record.mjs"},
          open(out, "w"))
PY
name="rubble-browserqa-$id"
trap 'docker kill --signal INT "$name" >/dev/null 2>&1 || true; rm -f "$run/spec.json"' EXIT
set +e
docker run --rm --name "$name" \
  -v "$run:/run-dir" \
  -v "$here/browser_qa.py:/app/browser_qa.py:ro" \
  -v rubble-modal-config:/config:ro -e MODAL_CONFIG_PATH=/config/modal.toml -w /app \
  rubble-modal-cli:1.6.2 run /app/browser_qa.py --spec /run-dir/spec.json --out /run-dir/out
code=$?
set -e
cp -R "$run/out/." "$out/" 2>/dev/null || true
[ -f "$run/out/exit_code" ] && code=$(cat "$run/out/exit_code")
echo "browser_qa: results in $out (exit $code)" >&2
exit "$code"
