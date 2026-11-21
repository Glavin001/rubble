#!/usr/bin/env bash
# Builds the WebGPU demos into one static site: the demo scenes (demo.html), the vehicle
# demo (vehicle.html), soft-body vehicles (softbody.html), heightfield terrain (terrain.html), the meteor impact (impact.html), the meteor run (showcase.html), Broadside (broadside.html), the timber yard (timber.html), the large-maps city (city.html), the motion demo (motion.html:
# CCD, characters, bullets), the fracture lab (fracture/), the generated vehicle models viewer (models/), the live vehicle designer (vehicle-designer.html), Rubble's generated sedan
# (__mods/sedan_new, made by running its generator) and a landing page (index.html).
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
#
# RUBBLE_WEB_PARTS="<part> ..." builds only those parts (default: all four, in this
# order, which is what Vercel and local builds do). CI builds kernels, world and stress
# in three parallel jobs, merges their engine-webgpu/generated/ folders and then builds
# the site part (.github/workflows/ci.yml): the same commands, the same site.
#   kernels  WGSL kernels, vehicle presets and vehicle wasm (wgsl_export)
#   world    vehicle-designer wasm, world wasm and the world wasm's Asyncify build
#   stress   fracture scene wasm, stress-solver wasm and its Asyncify build
#   site     scene snapshots, the generated sedan, scene packs, the assembled site
#            (needs the other three parts' outputs in engine-webgpu/generated/)
set -euo pipefail

PARTS=" ${RUBBLE_WEB_PARTS:-kernels world stress site} "
for web_part in $PARTS; do
  case "$web_part" in
    kernels | world | stress | site) ;;
    *) echo "RUBBLE_WEB_PARTS: unknown part '$web_part' (kernels, world, stress, site)" >&2; exit 1 ;;
  esac
done
# part NAME: whether this run builds that part.
part() { [[ "$PARTS" == *" $1 "* ]]; }

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
# The summary a build ends with.
step_times() {
  step ""
  echo "==> Step times (total ${SECONDS}s)"
  printf '%s\n' "${STEP_TIMES[@]}"
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
# The site part compiles neither kernels nor wasm.
if part kernels || part world || part stress; then
  . "$ENGINE/scripts/slangc.sh"
fi

# wasm-pack: the vehicle and fracture wasm modules.
if { part kernels || part world || part stress; } && ! command -v wasm-pack >/dev/null 2>&1; then
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

if part kernels; then
  step "WGSL kernels, vehicle presets and vehicle wasm (wgsl_export)"
  # Without the `gpu` feature: the exporter only needs the WGSL tables, not native wgpu.
  cargo run --release -p engine-webgpu --no-default-features --bin wgsl_export
fi
if part stress; then
  step "Fracture scene wasm (the unit tests; the page loads it inside the world wasm)"
  wasm-pack build crates/engine-fracture-wasm --release --target web \
    --out-dir ../engine-webgpu/generated/fracture-wasm --no-typescript
fi
if part world; then
  step "World wasm (engine-world-wasm: the pages' physics host, the fracture lab's scenes, Broadside)"
  # the vehicle generator (engine-vehiclegen) for the vehicle designer page
  wasm-pack build crates/engine-vehiclegen-wasm --release --target web \
    --out-dir ../engine-webgpu/generated/vehiclegen-wasm --no-typescript
  CARGO_PROFILE_RELEASE_DEBUG=0 wasm-pack build crates/engine-world-wasm --release --target web \
    --out-dir ../engine-webgpu/generated/world-wasm --no-typescript
fi
if part stress; then
  step "Stress-solver wasm (the Fracture lab's blast_parity, load path and local impact modes)"
  CARGO_PROFILE_RELEASE_DEBUG=0 wasm-pack build stress-oracle/crates/stress-web --release --target web \
    --out-dir ../../../crates/engine-webgpu/generated/stress-web --no-typescript
fi
STRESS_WASM="$ENGINE/crates/engine-webgpu/generated/stress-web/stress_web_bg.wasm"
WORLD_WASM="$ENGINE/crates/engine-webgpu/generated/world-wasm/engine_world_wasm_bg.wasm"
if part world || part stress; then
step "Stress-solver and world wasm, Asyncify builds (browsers without JSPI: Safari)"
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
# The features rustc's wasm32 target emits, all in Safari 15+ (nothing newer may appear).
ASYNCIFY_ARGS=(--enable-bulk-memory --enable-sign-ext --enable-mutable-globals
  --enable-nontrapping-float-to-int --enable-multivalue --enable-reference-types --asyncify
  --pass-arg=asyncify-imports@rubble-gpu-wait.yield_to_browser '--pass-arg=asyncify-removelist@*externref*' -O2)
asyncify() {
  mkdir -p "$STRESS_ASYNCIFY"
  "$WASM_OPT" "$STRESS_WASM" "${ASYNCIFY_ARGS[@]}" -o "$STRESS_ASYNCIFY/stress_web_bg.asyncify.wasm"
}
if part stress; then
  STRESS_ASYNCIFY="$(mktemp -d)/asyncify"
  cached asyncify "$( (cat "$STRESS_WASM"; echo "$BINARYEN_VERSION ${ASYNCIFY_ARGS[*]}") | key)" \
    "$STRESS_ASYNCIFY" asyncify
  cp "$STRESS_ASYNCIFY/stress_web_bg.asyncify.wasm" "${STRESS_WASM%.wasm}.asyncify.wasm"
fi
# Broadside's world wasm waits on the GPU the same way: its Asyncify build too.
world_asyncify() {
  mkdir -p "$WORLD_ASYNCIFY"
  "$WASM_OPT" "$WORLD_WASM" "${ASYNCIFY_ARGS[@]}" -o "$WORLD_ASYNCIFY/engine_world_wasm_bg.asyncify.wasm"
}
if part world; then
  WORLD_ASYNCIFY="$(mktemp -d)/asyncify"
  cached world-asyncify "$( (cat "$WORLD_WASM"; echo "$BINARYEN_VERSION ${ASYNCIFY_ARGS[*]}") | key)" \
    "$WORLD_ASYNCIFY" world_asyncify
  cp "$WORLD_ASYNCIFY/engine_world_wasm_bg.asyncify.wasm" "${WORLD_WASM%.wasm}.asyncify.wasm"
fi
fi # part world || part stress

if ! part site; then
  step_times
  exit 0
fi
# The site takes engine-webgpu/generated/ as it is: the other parts' outputs must be
# there (built above, or by the runs that built those parts).
for f in wgsl/manifest.json vehicle_wasm fracture-wasm vehiclegen-wasm \
  "${WORLD_WASM#"$WEB/generated/"}" world-wasm/engine_world_wasm_bg.asyncify.wasm \
  "${STRESS_WASM#"$WEB/generated/"}" stress-web/stress_web_bg.asyncify.wasm; do
  if [[ ! -e "$WEB/generated/$f" ]]; then
    echo "error: generated/$f is missing: build the part that makes it (RUBBLE_WEB_PARTS)" >&2
    exit 1
  fi
done
step "Scene snapshots (scene_export, host build)"
cargo run --release -p engine-scenes --no-default-features --bin scene_export

step "Generated sedan (the generator and rigger, run)"
# Rubble's generated sedan is built here from its code (engine-vehiclegen's design,
# engine-softbody's rigger: the vehicle_gen_rig example), never committed: the BeamNG mod
# folder the vehicle lab and softbody.html load from __mods/sedan_new (JBeam, .pc, .glb,
# .rskin and files.json). A local $OUT/__mods that is a link to one's own mods folder
# (which may hold community cars, local only) is kept as it is and used instead.
MODS_LINK=""
if [[ -L "$OUT/__mods" ]]; then
  MODS_LINK="$(readlink "$OUT/__mods")"
  echo "    $OUT/__mods links to $MODS_LINK: using that local folder (the sedan is not generated)"
else
  SEDAN_DIR="$(mktemp -d)/sedan_new"
  gen_sedan() {
    cargo run --release -p engine-softbody --example vehicle_gen_rig -- sedan --out "$SEDAN_DIR"
  }
  # The design is built in memory from the generator's own parameters: its output
  # depends only on these crates' sources, the lock file and the compiler.
  sedan_key="$( ( (git ls-files -z crates/engine-vehiclegen crates/engine-softbody \
    crates/engine-vehicle crates/engine-schema Cargo.toml Cargo.lock rust-toolchain.toml \
    ':(exclude,glob)crates/*/tests/**' ':(exclude,glob)crates/*/examples/**'
    printf '%s\0' crates/engine-softbody/examples/vehicle_gen_rig.rs) | sort -z | xargs -0 cat
    rustc --version) | key)"
  cached sedan "$sedan_key" "$SEDAN_DIR" gen_sedan
fi

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
if [[ -n "$MODS_LINK" ]]; then
  ln -s "$MODS_LINK" "$OUT/__mods"
else
  mkdir -p "$OUT/__mods"
  mv "$SEDAN_DIR" "$OUT/__mods/sedan_new"
  # Every file the folder's files.json lists must be in it (the loader fetches each).
  for f in $(tr ',' '\n' < "$OUT/__mods/sedan_new/files.json" | tr -d '[]"'); do
    if [[ ! -f "$OUT/__mods/sedan_new/$f" ]]; then
      echo "error: __mods/sedan_new/files.json lists $f, which the export does not contain" >&2
      exit 1
    fi
  done
fi
cp "$WEB"/index.html "$WEB"/demo.html "$WEB"/vehicle.html "$WEB"/softbody.html "$WEB"/motion.html "$WEB"/terrain.html "$WEB"/impact.html "$WEB"/showcase.html "$WEB"/broadside.html "$WEB"/timber.html "$WEB"/city.html "$WEB"/one-world.html "$WEB"/brick-ring-drive.html "$WEB"/vehicle-lab.html "$WEB"/explicit-world.html "$WEB"/crash-orbit.html "$WEB"/showcase-car.html "$WEB"/vehicle-designer.html "$OUT"/
cp "$WEB"/*.mjs "$OUT"/
cp -R "$WEB"/generated "$OUT"/generated
cp -R "$WEB"/fracture "$OUT"/fracture
# the vehicle lab's scripted scenarios (vehicle-lab.html?script=<name>)
cp -R "$WEB"/vehicle-lab-scripts "$OUT"/vehicle-lab-scripts
# the vehicle-gen model viewer, with the generated vehicles' .glb files and thumbnails
mkdir -p "$OUT"/models/vehicles
cp "$ENGINE"/tools/vehicle-gen/viewer/index.html "$OUT"/models/index.html
cp "$ENGINE"/scenes/vehicles/index.json "$OUT"/models/vehicles/index.json
for v in "$ENGINE"/scenes/vehicles/*/; do
  n="$(basename "$v")"
  [[ -f "$v/$n.glb" ]] || continue
  mkdir -p "$OUT/models/vehicles/$n/renders"
  cp "$v/$n.glb" "$OUT/models/vehicles/$n/"
  if [[ -f "$v/renders/${n}_front34.jpg" ]]; then cp "$v/renders/${n}_front34.jpg" "$OUT/models/vehicles/$n/renders/"; fi
done
find "$OUT" \( -name '*.test.mjs' -o -name '.gitignore' -o -name '*-check.mjs' \) -delete
rm -rf "$OUT"/generated/*/.gitignore "$OUT"/generated/*/package.json
# Every page the landing page links to must be in the site.
missing=0
for page in $(grep -oE 'href="\./[^"?#]+\.html' "$OUT"/index.html | sed 's#href="\./##' | sort -u); do
  if [[ ! -f "$OUT/$page" ]]; then echo "error: index.html links $page, which the site does not contain" >&2; missing=1; fi
done
[[ $missing == 0 ]] || exit 1
echo "==> Done: $OUT ($(du -sh "$OUT" | cut -f1))"

if [[ -n "$CACHE" ]]; then echo "==> Build cache: $(du -sh "$CACHE" | cut -f1)"; fi
step_times
