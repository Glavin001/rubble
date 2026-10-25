#!/usr/bin/env bash
# Film a scene's destruction (fixed 1/30 s frames, clouds and volumetric fog on) and encode the
# frames to target/unreal/<film>-destruction.mp4 (H.264, 30 fps). Needs ffmpeg.
#   town        Rubble.Video.Town: Elm Park's houses, then the glass tower felled onto the grocery
#   glasstower  Rubble.Video.GlassTower: the fracture lab's glass tower and pavilion, shot at its base
#
#   scripts/unreal/video.sh [--no-build] [town|glasstower]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARGS=()
FILM=town
for a in "$@"; do
  case "$a" in
    town|glasstower) FILM="$a" ;;
    *) ARGS+=("$a") ;;
  esac
done
case "$FILM" in
  town) TEST=Rubble.Video.Town; FRAMES="$ROOT/unreal/RubbleDemo/Saved/RubbleShots/video/town_%04d.png" ;;
  glasstower) TEST=Rubble.Video.GlassTower; FRAMES="$ROOT/unreal/RubbleDemo/Saved/RubbleShots/video/glasstower/frame_%04d.png" ;;
esac
RUBBLE_TEST_TIMEOUT=${RUBBLE_TEST_TIMEOUT:-1800} RUBBLE_VOLUMETRICS=1 "$ROOT/scripts/unreal/test.sh" ${ARGS[@]+"${ARGS[@]}"} --only "$TEST"
OUT="$ROOT/target/unreal/$FILM-destruction.mp4"
mkdir -p "$(dirname "$OUT")"
ffmpeg -loglevel error -y -framerate 30 -i "$FRAMES" -c:v libx264 -pix_fmt yuv420p -crf 20 -movflags +faststart "$OUT"
echo "video: $OUT"
