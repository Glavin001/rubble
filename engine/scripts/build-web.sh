#!/usr/bin/env bash
# Builds the WebGPU demos into one static site: the demo scenes (demo.html), the vehicle
# demo (vehicle.html), the fracture lab (fracture/) and a landing page (index.html).
# The one command Vercel, CI and local builds use. Runs on Linux and macOS; no GPU needed.
#
#   bash engine/scripts/build-web.sh            # output: engine/web-dist/
#   OUT=/tmp/site bash engine/scripts/build-web.sh
#
# Needs Rust (stable, with the wasm32-unknown-unknown target) and Git LFS assets. slangc
# and wasm-pack are used from PATH (or $SLANGC) when present, otherwise the pinned
# releases are downloaded into $RUBBLE_TOOLS (default ~/.cache/rubble-tools).
set -euo pipefail

SLANG_VERSION=2026.19   # engine/TOOLCHAIN.md
WASM_PACK_VERSION=0.14.0

ENGINE="$(cd "$(dirname "$0")/.." && pwd)"
WEB="$ENGINE/crates/engine-webgpu"
OUT="${OUT:-$ENGINE/web-dist}"
TOOLS="${RUBBLE_TOOLS:-$HOME/.cache/rubble-tools}"

if [[ -f "$HOME/.cargo/env" ]]; then
  # shellcheck source=/dev/null
  . "$HOME/.cargo/env"
fi
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

cd "$ENGINE"
echo "==> WGSL kernels, vehicle presets and vehicle wasm (wgsl_export)"
cargo run --release -p engine-webgpu --bin wgsl_export
echo "==> Fracture lab wasm"
wasm-pack build crates/engine-fracture-wasm --release --target web \
  --out-dir ../engine-webgpu/generated/fracture-wasm --no-typescript
echo "==> Scene snapshots (scene_export, host build)"
cargo run --release -p engine-scenes --no-default-features --bin scene_export

echo "==> Assembling $OUT"
rm -rf "$OUT"
mkdir -p "$OUT"
cp "$WEB"/index.html "$WEB"/demo.html "$WEB"/vehicle.html "$OUT"/
cp "$WEB"/*.mjs "$OUT"/
cp -R "$WEB"/generated "$OUT"/generated
cp -R "$WEB"/fracture "$OUT"/fracture
find "$OUT" \( -name '*.test.mjs' -o -name '.gitignore' \) -delete
rm -rf "$OUT"/generated/*/.gitignore "$OUT"/generated/*/package.json
echo "==> Done: $OUT ($(du -sh "$OUT" | cut -f1))"
