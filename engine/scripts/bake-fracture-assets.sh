#!/usr/bin/env bash
# Rebuild the fracture lab's assets from Shardwright, reproducibly.
#
# Shardwright (the `prefracture` CLI, crate frac-cli) is taken from either
#   --shardwright PATH (or SHARDWRIGHT_PATH)  an existing local clone, built as it is
#                                            (a warning when it is not at the pinned commit), or
#   SHARDWRIGHT_REV=<commit>                 github.com/Glavin001/Shardwright at that commit
#                                            (cloned and built once under engine/.tools/);
#                                            default: the pinned commit below.
# Each asset is baked from its Shardwright source mesh and bake config, packaged as one
# self-contained GLB (`export-glb --uncompressed`), and exported for the stress oracle
# (`fracture_stress_export`, level 0). Bakes are deterministic for a given Shardwright
# commit, source, config and seed; the script prints each output's SHA-256 so a rebuild
# can be compared with the recorded one.
#
#   engine/scripts/bake-fracture-assets.sh [--shardwright PATH] [--out DIR] [--install]
#   engine/scripts/bake-fracture-assets.sh --shardwright ~/Development/Shardwright
#
# Without --install the GLBs go to DIR (default engine/.tools/baked/); --install copies
# them over engine/crates/engine-webgpu/fracture/assets/ and the oracle packs. Different
# Shardwright commits fracture differently, so installing changes the lab's assets.
set -euo pipefail
# The Shardwright commit the lab's rebuilt assets come from (export-glb first appears there).
PINNED_SHARDWRIGHT_REV=e2ea334cbd06181b11a2d793e30582b949af2ece
ENGINE=$(cd "$(dirname "$0")/.." && pwd)
OUT="$ENGINE/.tools/baked"
INSTALL=0
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT=$2; shift 2 ;;
    --shardwright) SHARDWRIGHT_PATH=$2; shift 2 ;;
    --install) INSTALL=1; shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done

# name  source (in the Shardwright checkout)       config                                         levels for the oracle
ASSETS=(
  "veneer-house benchmarks/assets/veneer_house.glb benchmarks/configs/building_lite.toml 0"
  "veneer-house-hull16 benchmarks/assets/veneer_house.glb $ENGINE/crates/engine-webgpu/fracture/assets/veneer-house-hull16.toml 0"
  "masonry-arch benchmarks/assets/masonry_arch.glb benchmarks/configs/building_lite.toml 0"
)

if [ -n "${SHARDWRIGHT_PATH:-}" ]; then
  SW=$(cd "$SHARDWRIGHT_PATH" && pwd)
  REV="$(git -C "$SW" rev-parse HEAD)$(git -C "$SW" diff --quiet && git -C "$SW" diff --cached --quiet || echo -dirty)"
  if [ "$REV" != "$PINNED_SHARDWRIGHT_REV" ]; then
    echo "warning: $SW is at $REV, not the pinned $PINNED_SHARDWRIGHT_REV: its bakes may differ from the recorded assets" >&2
  fi
else
  SHARDWRIGHT_REV=${SHARDWRIGHT_REV:-$PINNED_SHARDWRIGHT_REV}
  SW="$ENGINE/.tools/shardwright-$SHARDWRIGHT_REV"
  if [ ! -d "$SW/.git" ]; then
    git clone --filter=blob:none https://github.com/Glavin001/Shardwright.git "$SW"
  fi
  git -C "$SW" fetch -q origin "$SHARDWRIGHT_REV" || true
  git -C "$SW" checkout -q "$SHARDWRIGHT_REV"
  REV=$SHARDWRIGHT_REV
fi
(cd "$SW" && cargo build --release -q -p frac-cli)
PF="$SW/target/release/prefracture"
"$PF" export-glb --help > /dev/null 2>&1 || { echo "this Shardwright ($REV) has no export-glb" >&2; exit 1; }

mkdir -p "$OUT"
echo "Shardwright $REV"
for entry in "${ASSETS[@]}"; do
  read -r name source config level <<< "$entry"
  case "$config" in /*) ;; *) config="$SW/$config" ;; esac
  work="$OUT/$name.bake"
  rm -rf "$work"
  mkdir -p "$work"
  "$PF" bake --input "$SW/$source" --config "$config" --out "$work/" | tail -1
  stem=$(basename "$source" .glb)
  "$PF" export-glb --input "$work/$stem.glb" --physics "$work/$stem.fracphys" --out "$OUT/$name.glb" --uncompressed > /dev/null
  cargo run -q --manifest-path "$ENGINE/Cargo.toml" -p engine-fracture --bin fracture_stress_export -- \
    "$OUT/$name.glb" "$OUT/$name-l$level.json" "$level"
  shasum -a 256 "$work/$stem.fracphys" "$OUT/$name.glb" | sed "s#$OUT/##"
  if [ "$INSTALL" = 1 ]; then
    cp "$OUT/$name.glb" "$ENGINE/crates/engine-webgpu/fracture/assets/$name.glb"
    [ "$name" = veneer-house ] && cp "$OUT/$name-l$level.json" "$ENGINE/stress-oracle/packs/veneer-house-l$level.json"
  fi
done
