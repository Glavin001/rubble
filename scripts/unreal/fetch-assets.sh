#!/usr/bin/env bash
# Pull RubbleDemo's Quixel Megascans (unreal/RubbleDemo/Assets/megascans.json: free Fab listings)
# from the private Glavin001/rubble-assets repo (Git LFS) into unreal/RubbleDemo/SourceArt/Megascans,
# which git ignores here: Fab's licence does not allow redistributing them, and this repo is public.
# Without access, add each listing in the manifest to your Fab library and unzip its download into
# SourceArt/Megascans/<key>/. Then scripts/unreal/build-content.sh imports them.
#
#   scripts/unreal/fetch-assets.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DST="$ROOT/unreal/RubbleDemo/SourceArt/Megascans"
REPO="${RUBBLE_ASSETS_REPO:-https://github.com/Glavin001/rubble-assets.git}"
if [[ -d "$DST/.git" ]]; then
  git -C "$DST" pull --ff-only
  git -C "$DST" lfs pull
else
  rm -rf "$DST"
  git clone "$REPO" "$DST"
  git -C "$DST" lfs pull
fi
echo "fetch-assets: $(find "$DST" -name '*.jpg' -o -name '*.png' -o -name '*.exr' -o -name '*.gltf' -o -name '*.glb' -o -name '*.bin' | grep -v '/.git/' | wc -l | tr -d ' ') files in $DST"
