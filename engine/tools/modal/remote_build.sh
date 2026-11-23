#!/usr/bin/env bash
# Build a cargo selection on Modal's CPUs instead of this machine (remote_build.py):
# only the source files that changed are sent; the packed archive lands in the GPU
# runner's archive volume. Prints the archive's key.
#
#   engine/tools/modal/remote_build.sh -- -p engine-cuda --example explicit_replay
#
# ENGINE_* variables reach the build (ENGINE_CUDA_SASS, ENGINE_CUDA_FMAD,
# ENGINE_CUDA_EXPLICIT_MODULES, ...). The tree key is this worktree's path: each worktree
# has its own source copy and cargo target on Modal, and all share the kernel caches.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
[ "${1:-}" = "--" ] && shift
[ $# -gt 0 ] || { echo "usage: remote_build.sh -- <cargo selection>" >&2; exit 2; }
key="$(printf '%s' "$repo" | shasum -a 256 | cut -c1-12)"
run="$repo/engine/target/remote-build/$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$run"
t0=$(date +%s)
# The tree: every tracked or untracked, not ignored file under engine/, except the large
# assets (LFS fracture scenes and the like) no build reads. sha256 a file.
(cd "$repo" && git ls-files -z -co --exclude-standard -- engine \
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
    if st.st_size > 8 << 20:
        continue
    print(hashlib.sha256(open(q, "rb").read()).hexdigest() + "\t" + q)
') > "$run/manifest.tsv"
echo "remote_build: $(wc -l < "$run/manifest.tsv" | tr -d ' ') files hashed in $(( $(date +%s) - t0 )) s" >&2
env_json="$(python3 -c 'import json, os; print(json.dumps({k: v for k, v in os.environ.items() if k.startswith("ENGINE_")}))')"
python3 - "$run/spec.json" "$key" "$env_json" "$@" <<'PY'
import json, sys
out, key, env, *sel = sys.argv[1:]
json.dump({"key": key, "root": "/repo-local", "manifest": "/run-dir/manifest.tsv", "result": "/run-dir/result.json",
           "env": json.loads(env), "selection": sel, "pack_opts": []}, open(out, "w"))
PY
cli=rubble-modal-cli:1.6.2
docker image inspect "$cli" >/dev/null 2>&1 || docker build -q -t "$cli" "$here" >/dev/null
docker run --rm --name "rubble-rb-$$" \
  -v "$run:/run-dir" \
  -v "$repo:/repo-local:ro" \
  -v "$repo/engine/tools/cuda:/ctx/cuda:ro" \
  -v "$here/remote_build.py:/app/remote_build.py:ro" \
  -v rubble-modal-config:/config:ro -e MODAL_CONFIG_PATH=/config/modal.toml \
  -w /app "$cli" run /app/remote_build.py --spec /run-dir/spec.json
echo "remote_build: total $(( $(date +%s) - t0 )) s" >&2
cat "$run/result.json" 2>/dev/null; echo
