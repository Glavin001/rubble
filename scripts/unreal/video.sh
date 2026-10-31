#!/usr/bin/env bash
# Film a scene's destruction (fixed 1/30 s frames, clouds and volumetric fog on) and encode the
# frames to target/unreal/<film>-destruction.mp4 (H.264, 30 fps). Needs ffmpeg.
#   town        Rubble.Video.Town: Elm Park's houses, then the glass tower felled onto the grocery
#   glasstower  Rubble.Video.GlassTower: the fracture lab's glass tower and pavilion, shot at its base,
#               one collapse from five cameras: target/unreal/glasstower-<camera>.mp4 each, and
#               glasstower-multicam.mp4 (the four close cameras side by side)
#   softcrash   Rubble.Video.SoftCrash: the soft-body head-on and wall crash tests, slowed around
#               the impact, each from five cameras: target/unreal/softcrash-<crash>-<camera>.mp4,
#               and softcrash-<crash>-multicam.mp4
#
#   scripts/unreal/video.sh [--no-build] [town|glasstower|softcrash]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARGS=()
FILM=town
for a in "$@"; do
  case "$a" in
    town|glasstower|softcrash) FILM="$a" ;;
    *) ARGS+=("$a") ;;
  esac
done
case "$FILM" in
  town) TEST=Rubble.Video.Town; FRAMES="$ROOT/unreal/RubbleDemo/Saved/RubbleShots/video/town_%04d.png" ;;
  glasstower) TEST=Rubble.Video.GlassTower; FRAMES="$ROOT/unreal/RubbleDemo/Saved/RubbleShots/video/glasstower" ;;
  softcrash) TEST=Rubble.Video.SoftCrash; FRAMES="$ROOT/unreal/RubbleDemo/Saved/RubbleShots/video/softcrash" ;;
esac
# Five cameras of 22-30 s each at full quality take well over the default half hour.
RUBBLE_TEST_TIMEOUT=${RUBBLE_TEST_TIMEOUT:-7200} RUBBLE_VOLUMETRICS=1 "$ROOT/scripts/unreal/test.sh" ${ARGS[@]+"${ARGS[@]}"} --only "$TEST"
mkdir -p "$ROOT/target/unreal"
encode() { ffmpeg -loglevel error -y -framerate 30 -i "$1" -c:v libx264 -pix_fmt yuv420p -crf 20 -movflags +faststart "$2"; echo "video: $2"; }
if [ "$FILM" = glasstower ]; then
  for cam in "$FRAMES"/*/; do
    cam=$(basename "$cam")
    encode "$FRAMES/$cam/frame_%04d.png" "$ROOT/target/unreal/glasstower-$cam.mp4"
  done
  # The four close cameras in a 2x2 grid, each at half size.
  V=() ; for cam in inside roof pavilion street; do V+=(-framerate 30 -i "$FRAMES/$cam/frame_%04d.png"); done
  ffmpeg -loglevel error -y "${V[@]}" -filter_complex \
    "[0]scale=iw/2:-2[a];[1]scale=iw/2:-2[b];[2]scale=iw/2:-2[c];[3]scale=iw/2:-2[d];[a][b][c][d]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0" \
    -c:v libx264 -pix_fmt yuv420p -crf 20 -movflags +faststart "$ROOT/target/unreal/glasstower-multicam.mp4"
  echo "video: $ROOT/target/unreal/glasstower-multicam.mp4"
elif [ "$FILM" = softcrash ]; then
  # One film per crash and camera (<crash>-<camera>), and per crash its four cameras in a 2x2 grid.
  for cam in "$FRAMES"/*/; do
    cam=$(basename "$cam")
    encode "$FRAMES/$cam/frame_%04d.png" "$ROOT/target/unreal/softcrash-$cam.mp4"
  done
  for crash in headon wall; do
    [ -d "$FRAMES/$crash-side" ] && [ -d "$FRAMES/$crash-bonnet" ] || continue
    V=() ; for cam in side closeup overhead bonnet; do V+=(-framerate 30 -i "$FRAMES/$crash-$cam/frame_%04d.png"); done
    ffmpeg -loglevel error -y "${V[@]}" -filter_complex \
      "[0]scale=iw/2:-2[a];[1]scale=iw/2:-2[b];[2]scale=iw/2:-2[c];[3]scale=iw/2:-2[d];[a][b][c][d]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0" \
      -c:v libx264 -pix_fmt yuv420p -crf 20 -movflags +faststart "$ROOT/target/unreal/softcrash-$crash-multicam.mp4"
    echo "video: $ROOT/target/unreal/softcrash-$crash-multicam.mp4"
  done
else
  encode "$FRAMES" "$ROOT/target/unreal/$FILM-destruction.mp4"
fi
