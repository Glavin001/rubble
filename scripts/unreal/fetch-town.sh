#!/usr/bin/env bash
# Copy vibe-land's Vibe Town ScenePack (structures/vibe-town/out) into RubbleDemo's generated
# content, where `fracture:town` loads it. Not committed (30 MB of JSON, and vibe-land owns it).
#
#   scripts/unreal/fetch-town.sh [variant]   # vibe-town (default), vibe-town-low, vibe-town-hero
#
# VIBE_LAND points at the vibe-land checkout (default ~/Development/vibe-land). Rebuild the pack
# there with `node structures/vibe-town/build-town.mjs`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VIBE_LAND="${VIBE_LAND:-$HOME/Development/vibe-land}"
VARIANT="${1:-vibe-town}"
SRC="$VIBE_LAND/structures/vibe-town/out"
DST="$ROOT/unreal/RubbleDemo/Content/Rubble/Town"
mkdir -p "$DST"
for ext in json meta.json visuals.json slots; do
  if [[ -f "$SRC/$VARIANT.$ext" ]]; then
    cp "$SRC/$VARIANT.$ext" "$DST/vibe-town.$ext"
  fi
done
[[ -f "$DST/vibe-town.json" ]] || { echo "fetch-town: no $SRC/$VARIANT.json" >&2; exit 1; }
echo "fetch-town: $VARIANT -> $DST ($(du -h "$DST/vibe-town.json" | cut -f1))"
