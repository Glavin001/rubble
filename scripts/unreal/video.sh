#!/usr/bin/env bash
# Film Vibe Town's destruction (Rubble.Video.Town: fixed 1/30 s frames, clouds and volumetric fog
# on) and encode the frames to target/unreal/town-destruction.mp4 (H.264, 30 fps). Needs ffmpeg.
#
#   scripts/unreal/video.sh [--no-build]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUBBLE_VOLUMETRICS=1 "$ROOT/scripts/unreal/test.sh" "$@" --only Rubble.Video.Town
FRAMES="$ROOT/unreal/RubbleDemo/Saved/RubbleShots/video"
OUT="$ROOT/target/unreal/town-destruction.mp4"
mkdir -p "$(dirname "$OUT")"
ffmpeg -loglevel error -y -framerate 30 -i "$FRAMES/town_%04d.png" -c:v libx264 -pix_fmt yuv420p -crf 20 -movflags +faststart "$OUT"
echo "video: $OUT"
