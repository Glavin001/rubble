#!/usr/bin/env bash
# Builds the WebGPU demos into one static site: the demo scenes (demo.html), the vehicle
# demo (vehicle.html), soft-body vehicles (softbody.html), heightfield terrain (terrain.html), the meteor impact (impact.html), the meteor run (showcase.html), Broadside (broadside.html), the motion demo (motion.html:
# CCD, characters, bullets), the fracture lab (fracture/) and a landing page (index.html).
# The one command Vercel, CI and local builds use. Runs on Linux and macOS; no GPU needed.
#
#   bash engine/scripts/build-web.sh            # output: engine/web-dist/
#   OUT=/tmp/site bash engine/scripts/build-web.sh
#
# Needs Rust (stable, with the wasm32-unknown-unknown target) and Git LFS assets. slangc
# and wasm-pack are used from PATH (or $SLANGC) when present, otherwise the pinned
# releases are downloaded into $RUBBLE_TOOLS (default ~/.cache/rubble-tools).
#
# RUBBLE_BUILD_CACHE=<dir> keeps work between builds: cargo's target dir (both
# workspaces), slangc's outputs, the vibe-city packs and the Asyncify build, each keyed
# by its inputs so a hit is the same bytes. On Vercel it is node_modules/.cache/rubble-build:
# Vercel's static builder keeps every **/node_modules/** between builds (and a branch's
# first build starts from production's cache).
set -euo pipefail

SLANG_VERSION=2026.19   # engine/TOOLCHAIN.md
WASM_PACK_VERSION=0.14.0
# Binaryen's wasm-opt makes the stress solver's Asyncify build (browsers without JSPI).
BINARYEN_VERSION=123

ENGINE="$(cd "$(dirname "$0")/.." && pwd)"
WEB="$ENGINE/crates/engine-webgpu"
OUT="${OUT:-$ENGINE/web-dist}"
TOOLS="${RUBBLE_TOOLS:-$HOME/.cache/rubble-tools}"
# step NAME: start a timed step; the summary at the end lists each step's seconds.
STEP_NAME=""
STEP_START=$SECONDS
STEP_TIMES=()
step() {
  if [[ -n "$STEP_NAME" ]]; then
    STEP_TIMES+=("$(printf '%5ds  %s' $((SECONDS - STEP_START)) "$STEP_NAME")")
  fi
  STEP_NAME="$1"
  STEP_START=$SECONDS
  if [[ -n "$1" ]]; then echo "==> $1"; fi
}

CACHE="${RUBBLE_BUILD_CACHE:-}"
if [[ -z "$CACHE" && -n "${VERCEL:-}" ]]; then CACHE="$ENGINE/../node_modules/.cache/rubble-build"; fi
if [[ -n "$CACHE" ]]; then
  mkdir -p "$CACHE"
  CACHE="$(cd "$CACHE" && pwd)"
  export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$CACHE/cargo-target}"
  # Cargo never deletes stale artifacts (old dependency versions, a new toolchain), and
  # Vercel's build cache holds 1 GB. Past the cap, start the target dir over: one cold
  # build, then it fits again.
  CAP_MB="${RUBBLE_TARGET_CAP_MB:-1600}"
  if [[ -d "$CARGO_TARGET_DIR" ]] \
    && (( $(du -sm "$CARGO_TARGET_DIR" | cut -f1) > CAP_MB )); then
    echo "==> $CARGO_TARGET_DIR is over ${CAP_MB} MB: starting it over"
    rm -rf "$CARGO_TARGET_DIR"
  fi
  export RUBBLE_SLANG_CACHE="$CACHE/slang"
  echo "==> Build cache: $CACHE"
fi

# cached NAME KEY DIR CMD...: with a build cache, a kept DIR for (NAME, KEY) is copied
# back instead of running CMD; otherwise CMD runs and DIR is kept for next time.
cached() {
  local name="$1" key="$2" dir="$3"
  shift 3
  local entry="$CACHE/$name/$key"
  if [[ -n "$CACHE" && -d "$entry" ]]; then
    echo "    (cached: $name $key)"
    rm -rf "$dir"
    mkdir -p "$(dirname "$dir")"
    cp -R "$entry" "$dir"
    return
  fi
  "$@"
  if [[ -n "$CACHE" ]]; then
    rm -rf "$CACHE/$name"
    mkdir -p "$CACHE/$name"
    cp -R "$dir" "$entry.tmp"
    mv "$entry.tmp" "$entry"
  fi
}
# A short content hash of stdin.
key() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum; else shasum -a 256; fi | cut -c1-16
}

if [[ -f "$HOME/.cargo/env" ]]; then
  # shellcheck source=/dev/null
  . "$HOME/.cargo/env"
fi
step "Tools (wasm32 target, slangc, wasm-pack)"
# From engine/: its rust-toolchain.toml (stable) is the one the build uses. The repo
# root's pins a nightly with rustc-dev, which a rustup call there would install.
cd "$ENGINE"
if ! rustup target list --installed 2>/dev/null | grep -q wasm32-unknown-unknown; then
  echo "==> rustup target add wasm32-unknown-unknown"
  rustup target add wasm32-unknown-unknown
fi

os="$(uname -s)"
arch="$(uname -m)"

# slangc: Slang → SPIR-V for wgsl_export (and the build scripts of the WebGPU crates).
if [[ -z "${SLANGC:-}" ]] && ! command -v slangc >/dev/null 2>&1 \
  && [[ ! -x "$HOME/.local/bin/slangc" ]]; then
  case "$os-$arch" in
    Linux-x86_64) asset="slang-$SLANG_VERSION-linux-x86_64-glibc-2.27.tar.gz" ;;
    Linux-aarch64) asset="slang-$SLANG_VERSION-linux-aarch64.tar.gz" ;;
    Darwin-arm64) asset="slang-$SLANG_VERSION-macos-aarch64.tar.gz" ;;
    Darwin-x86_64) asset="slang-$SLANG_VERSION-macos-x86_64.tar.gz" ;;
    *) echo "no slangc release for $os-$arch; install slangc $SLANG_VERSION" >&2; exit 1 ;;
  esac
  dir="$TOOLS/slang-$SLANG_VERSION"
  if [[ ! -x "$dir/bin/slangc" ]]; then
    echo "==> Downloading $asset"
    mkdir -p "$dir"
    curl -fsSL "https://github.com/shader-slang/slang/releases/download/v$SLANG_VERSION/$asset" \
      | tar -xz -C "$dir"
  fi
  export SLANGC="$dir/bin/slangc"
fi
echo "==> slangc: ${SLANGC:-$(command -v slangc || echo "$HOME/.local/bin/slangc")}"

# wasm-pack: the vehicle and fracture wasm modules.
if ! command -v wasm-pack >/dev/null 2>&1; then
  case "$os-$arch" in
    Linux-x86_64) target="x86_64-unknown-linux-musl" ;;
    Linux-aarch64) target="aarch64-unknown-linux-musl" ;;
    Darwin-arm64) target="aarch64-apple-darwin" ;;
    Darwin-x86_64) target="x86_64-apple-darwin" ;;
    *) echo "no wasm-pack release for $os-$arch; cargo install wasm-pack" >&2; exit 1 ;;
  esac
  name="wasm-pack-v$WASM_PACK_VERSION-$target"
  if [[ ! -x "$TOOLS/$name/wasm-pack" ]]; then
    echo "==> Downloading $name"
    mkdir -p "$TOOLS"
    curl -fsSL "https://github.com/wasm-bindgen/wasm-pack/releases/download/v$WASM_PACK_VERSION/$name.tar.gz" \
      | tar -xz -C "$TOOLS"
  fi
  export PATH="$TOOLS/$name:$PATH"
fi

# The fracture lab's assets are in Git LFS; pointer files would load as broken GLBs.
for glb in "$WEB"/fracture/assets/*.glb; do
  if head -c 40 "$glb" | grep -q "git-lfs"; then
    echo "$glb is a Git LFS pointer: enable Git LFS (git lfs pull, or the host's LFS setting)" >&2
    exit 1
  fi
done

step "WGSL kernels, vehicle presets and vehicle wasm (wgsl_export)"
# Without the `gpu` feature: the exporter only needs the WGSL tables, not native wgpu.
cargo run --release -p engine-webgpu --no-default-features --bin wgsl_export
step "Fracture lab wasm"
wasm-pack build crates/engine-fracture-wasm --release --target web \
  --out-dir ../engine-webgpu/generated/fracture-wasm --no-typescript
step "World API wasm (Broadside: engine-world-wasm)"
CARGO_PROFILE_RELEASE_DEBUG=0 wasm-pack build crates/engine-world-wasm --release --target web \
  --out-dir ../engine-webgpu/generated/world-wasm --no-typescript
step "Stress-solver wasm (the Fracture lab's stress bond mode)"
CARGO_PROFILE_RELEASE_DEBUG=0 wasm-pack build stress-oracle/crates/stress-web --release --target web \
  --out-dir ../../../crates/engine-webgpu/generated/stress-web --no-typescript
step "Stress-solver wasm, Asyncify build (browsers without JSPI: Safari)"
# The solver waits on GPU maps through the import rubble-gpu-wait.yield_to_browser
# (engine-gpu-wait). Chrome suspends the stack there with JSPI; elsewhere this build's
# Asyncify unwinds it (fracture/gpu-wait.mjs). Only that import may unwind. Indirect calls
# are followed (the blast_parity world reaches its waits through trait objects). Asyncify
# cannot instrument externref, which only wasm-bindgen's shims use (`* externref shim`,
# `__externref_*`): they call into JavaScript and never wait, so they are left out.
case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) binaryen_target=arm64-macos ;;
  Darwin-x86_64) binaryen_target=x86_64-macos ;;
  Linux-x86_64) binaryen_target=x86_64-linux ;;
  Linux-aarch64) binaryen_target=aarch64-linux ;;
  *) echo "no Binaryen release for $(uname -s)-$(uname -m): install wasm-opt $BINARYEN_VERSION" >&2; exit 1 ;;
esac
WASM_OPT="$TOOLS/binaryen-version_$BINARYEN_VERSION/bin/wasm-opt"
if [[ ! -x "$WASM_OPT" ]]; then
  echo "==> Downloading Binaryen $BINARYEN_VERSION"
  mkdir -p "$TOOLS"
  curl -fsSL "https://github.com/WebAssembly/binaryen/releases/download/version_$BINARYEN_VERSION/binaryen-version_$BINARYEN_VERSION-$binaryen_target.tar.gz" \
    | tar -xz -C "$TOOLS"
fi
STRESS_WASM="$ENGINE/crates/engine-webgpu/generated/stress-web/stress_web_bg.wasm"
# The features rustc's wasm32 target emits, all in Safari 15+ (nothing newer may appear).
ASYNCIFY_ARGS=(--enable-bulk-memory --enable-sign-ext --enable-mutable-globals
  --enable-nontrapping-float-to-int --enable-multivalue --enable-reference-types --asyncify
  --pass-arg=asyncify-imports@rubble-gpu-wait.yield_to_browser '--pass-arg=asyncify-removelist@*externref*' -O2)
asyncify() {
  mkdir -p "$STRESS_ASYNCIFY"
  "$WASM_OPT" "$STRESS_WASM" "${ASYNCIFY_ARGS[@]}" -o "$STRESS_ASYNCIFY/stress_web_bg.asyncify.wasm"
}
STRESS_ASYNCIFY="$(mktemp -d)/asyncify"
cached asyncify "$( (cat "$STRESS_WASM"; echo "$BINARYEN_VERSION ${ASYNCIFY_ARGS[*]}") | key)" \
  "$STRESS_ASYNCIFY" asyncify
cp "$STRESS_ASYNCIFY/stress_web_bg.asyncify.wasm" "${STRESS_WASM%.wasm}.asyncify.wasm"
step "Scene snapshots (scene_export, host build)"
cargo run --release -p engine-scenes --no-default-features --bin scene_export

step "vibe-city scene packs (engine/scenes/vibe-city, bun)"
if command -v bun >/dev/null 2>&1; then BUN=(bun); else BUN=(npx --yes bun@1.2.4); fi
VIBE="$ENGINE/scenes/vibe-city"
vibe_city() { (cd "$VIBE" && "${BUN[@]}" install --frozen-lockfile && "${BUN[@]}" run export.ts); }
# The export is deterministic (seeded Math.random) and reads only its tracked files and
# the packages bun.lock pins.
vibe_key="$( (cd "$VIBE" && git ls-files -z | sort -z | xargs -0 cat; "${BUN[@]}" --version) | key)"
cached vibe-city "$vibe_key" "$WEB/generated/vibe-city" vibe_city
step "Dune buggy render mesh (engine/scenes/dune-buggy, bun)"
(cd "$ENGINE/scenes/dune-buggy" && "${BUN[@]}" install --frozen-lockfile && "${BUN[@]}" run bake-mesh.mjs)

step "Assembling $OUT"
rm -rf "$OUT"
mkdir -p "$OUT"
cp "$WEB"/index.html "$WEB"/demo.html "$WEB"/vehicle.html "$WEB"/softbody.html "$WEB"/motion.html "$WEB"/terrain.html "$WEB"/impact.html "$WEB"/showcase.html "$WEB"/broadside.html "$OUT"/
cp "$WEB"/*.mjs "$OUT"/
cp -R "$WEB"/generated "$OUT"/generated
cp -R "$WEB"/fracture "$OUT"/fracture
find "$OUT" \( -name '*.test.mjs' -o -name '.gitignore' -o -name '*-check.mjs' \) -delete
rm -rf "$OUT"/generated/*/.gitignore "$OUT"/generated/*/package.json
echo "==> Done: $OUT ($(du -sh "$OUT" | cut -f1))"

if [[ -n "$CACHE" ]]; then echo "==> Build cache: $(du -sh "$CACHE" | cut -f1)"; fi
step ""
echo "==> Step times (total ${SECONDS}s)"
printf '%s\n' "${STEP_TIMES[@]}"
