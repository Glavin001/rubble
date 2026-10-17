#!/usr/bin/env bash
# Full destruction of a Shardwright asset in the standalone stress-ref world: bake (or
# reuse) the asset, export it to shardwright-stress/1 with its component names, and run
# idle, ram, projectile and component removal (examples/shardwright_destruction.rs).
#
#   scripts/shardwright_verify.sh [OUT_DIR]
#
# Env: SHARDWRIGHT (checkout, default ShardWright beside the PhysX checkout), FRACPHYS and
# ASSET_JSON (a baked .fracphys and its .asset.json; default: bake into OUT_DIR/bake),
# CASES (default "idle ram projectile remove"), DURATION (s, default 2; idle 1),
# THREADS (rayon threads, default 8), EXTRA (more example options, e.g.
# "--feature layer_contact=off" for parity with stress-gpu).
#
# A second backend's observations compare against these with
#   stress-ref compare OUT_DIR/ref/<case>.scene.json <theirs.json> OUT_DIR/ref/<case>.obs.json
set -euo pipefail
cd "$(dirname "$0")/.."
ORACLE=$PWD
OUT=$(mkdir -p "${1:-/tmp/shardwright-verify}" && cd "${1:-/tmp/shardwright-verify}" && pwd)
# Beside the main PhysX checkout (also from a worktree: git's common dir is the main one).
SW=${SHARDWRIGHT:-$(cd "$(git rev-parse --git-common-dir)/../.." && pwd)/ShardWright}
CASES=${CASES:-idle ram projectile remove}
DURATION=${DURATION:-2}
THREADS=${THREADS:-8}

if [ -z "${FRACPHYS:-}" ]; then
    FRACPHYS=$OUT/bake/two_storey_building.fracphys
    ASSET_JSON=$OUT/bake/two_storey_building.asset.json
    if [ ! -f "$FRACPHYS" ]; then
        (cd "$SW" && cargo build --release -p frac-cli)
        "$SW/target/release/prefracture" bake --input "$SW/benchmarks/assets/two_storey_building.glb" \
            --meta "$SW/benchmarks/assets/two_storey_building.glb.meta.json" \
            --config "$SW/benchmarks/configs/support_clusters.toml" --out "$OUT/bake"
    fi
fi

# The exporter's path dependencies assume Shardwright beside the main PhysX checkout;
# build a copy pointed at $SW so this works from any worktree.
rm -rf "$OUT/export-src" && cp -R tools/shardwright-export "$OUT/export-src"
sed -i.bak "s#\.\./\.\./\.\./\.\./\.\./Shardwright#$SW#" "$OUT/export-src/Cargo.toml"
CARGO_TARGET_DIR=$OUT/export-target cargo build --release --manifest-path "$OUT/export-src/Cargo.toml"
EXPORT=$OUT/two_storey_L2.json
"$OUT/export-target/release/shard-export" "$FRACPHYS" "$EXPORT" 2 ${ASSET_JSON:-}

cargo build --release -p stress-ref --example shardwright_destruction
mkdir -p "$OUT/ref"
for c in $CASES; do
    d=$DURATION; [ "$c" = idle ] && d=1
    RAYON_NUM_THREADS=$THREADS ./target/release/examples/shardwright_destruction "$EXPORT" "$c" \
        --duration "$d" --out "$OUT/ref" ${EXTRA:-} 2> "$OUT/ref/$c.log" | tee "$OUT/ref/$c.summary"
done
