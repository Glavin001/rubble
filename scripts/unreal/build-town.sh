#!/usr/bin/env bash
# Write the town RubbleDemo's `fracture:town` loads (unreal/RubbleDemo/Content/Rubble/Town,
# not committed): rubble's base town pack (engine/scenes/town/vibe-town.json, Git LFS, first
# copied from vibe-land's Vibe Town) with its Main Street tower beside the grocery replaced by
# engine-fracture's glass office tower (glass_tower.rs, town_pack); and the glass tower and
# pavilion scene, `fracture:glasstower` (glass-tower.json, glass_tower_pack).
#
#   scripts/unreal/build-town.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE="$ROOT/engine/scenes/town/vibe-town.json"
DST="$ROOT/unreal/RubbleDemo/Content/Rubble/Town"
if head -c 64 "$BASE" | grep -q "git-lfs"; then
  echo "build-town: $BASE is an LFS pointer: run git lfs pull" >&2
  exit 1
fi
mkdir -p "$DST"
(cd "$ROOT/engine" && cargo run --release -q -p engine-fracture --bin town_pack -- "$BASE" "$DST/vibe-town.json")
(cd "$ROOT/engine" && cargo run --release -q -p engine-fracture --bin glass_tower_pack -- "$DST/glass-tower.json")
echo "build-town: $DST/vibe-town.json ($(du -h "$DST/vibe-town.json" | cut -f1))"
