#!/usr/bin/env bash
# Copy vibe-land's 2K Poly Haven city textures (CC0) into RubbleDemo's generated content, at the
# same /Game/VibeCity paths so the assets' own references hold: the HD texture arrays (one layer
# per material in vibe-land's shared/assets/materials/catalog.json), their layer sources, and the
# CC0 fracture-surface sets. Also copies the catalog, which maps a pack material's textureKey to
# its layer and tile size, and writes each layer's mean albedo (vibe-land's generated texture-set
# table) to texture-means.json, so a material's colour tints its layer. Not committed.
#
# The assets come from vibe-land's own Unreal import (run there once:
# scripts/unreal/import-city-materials.sh --hd). VIBE_LAND defaults to ~/Development/vibe-land.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VIBE_LAND="${VIBE_LAND:-$HOME/Development/vibe-land}"
SRC="$VIBE_LAND/unreal/VibeCity/Content/VibeCity/Textures"
DST="$ROOT/unreal/RubbleDemo/Content/VibeCity/Textures"
[[ -f "$SRC/T_CityAlbedoHDArray.uasset" ]] || {
  echo "fetch-city-textures: no $SRC/T_CityAlbedoHDArray.uasset (run vibe-land's scripts/unreal/import-city-materials.sh --hd)" >&2
  exit 1
}
mkdir -p "$DST/Layers" "$DST/CC0" "$ROOT/unreal/RubbleDemo/Content/Rubble/Town"
cp "$SRC"/T_City{Albedo,Normal,ARM}HDArray.uasset "$DST/"
cp "$SRC"/Layers/T_City{AlbedoHD,ArmHD,NormalHD}_*.uasset "$DST/Layers/"
cp "$SRC"/CC0/*.uasset "$DST/CC0/"
cp "$VIBE_LAND/shared/assets/materials/catalog.json" "$ROOT/unreal/RubbleDemo/Content/Rubble/Town/catalog.json"
python3 - "$VIBE_LAND/client/src/scene/cityTextureSets.generated.ts" "$ROOT/unreal/RubbleDemo/Content/Rubble/Town/texture-means.json" <<'PY'
import json, re, sys
means = [[float(v) for v in m] for m in re.findall(r"meanLinear: \[([\d.]+), ([\d.]+), ([\d.]+)\]", open(sys.argv[1]).read())]
assert means, "no meanLinear entries"
json.dump({"meanLinear": means}, open(sys.argv[2], "w"))
PY
echo "fetch-city-textures: $(du -sh "$DST" | cut -f1) -> $DST"
