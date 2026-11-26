#!/usr/bin/env bash
# Two generated sedans head on, filmed on the CPU reference with continuous contact, once with
# rigid impact zones only and once with non-rigid ones first (DAT_SOFT), rendered in Blender
# (EEVEE) and put side by side: one MP4 for reviewing the crash.
#
#   engine/tools/c2c/film-zones.sh [KMH] [FRAMES] [OUT_DIR] [OFFSET_M]
#
# KMH each car (default 100); FRAMES at 4 ms (default 150: 0.6 s, 5 s of video at 7.5x slow
# motion); OUT_DIR default ~/Development/rubble-artifacts/c2c-zones. Each job is bounded
# (FILM_LIMIT seconds, default 170); the films and renders of the two run in parallel.
set -euo pipefail
ENGINE="$(cd "$(dirname "$0")/../.." && pwd)"
KMH="${1:-100}"; FRAMES="${2:-150}"; OUT="${3:-$HOME/Development/rubble-artifacts/c2c-zones}"; OFF="${4:-0}"
LIMIT="${FILM_LIMIT:-170}"
BLENDER="${BLENDER:-/Applications/Blender.app/Contents/MacOS/Blender}"
BIN="$ENGINE/target/release/examples/vehicle_film"
(cd "$ENGINE" && cargo build --release -q -p engine-softbody-lab --example vehicle_film)
tag="c2c_${KMH}kmh_off${OFF}"
mkdir -p "$OUT"
bound() { perl -e 'alarm shift; exec @ARGV' "$LIMIT" "$@" </dev/null; }
run() { # soft label
  local d="$OUT/$tag/soft$1"
  rm -rf "$d"; mkdir -p "$d"
  (cd "$ENGINE" && FILM_KMH="$KMH" FILM_OFFSET="$OFF" FILM_FRAMES="$FRAMES" FILM_CONTINUOUS=1 DAT_SOFT="$1" \
    bound "$BIN" c2c "$d/film" sedan > "$d/film.log" 2>&1)
  bound "$BLENDER" -b --python "$ENGINE/tools/vehicle-gen/render_film.py" -- "$d/film" "$d/clip.mp4" \
    --samples 8 --size 960x540 > "$d/render.log" 2>&1
  ffmpeg -y -loglevel error -i "$d/clip.mp4" -vf "drawtext=text='$2':x=20:y=20:fontsize=30:fontcolor=white:box=1:boxcolor=black@0.5" "$d/labelled.mp4"
}
run 0 "rigid impact zones" & a=$!
run 8 "non-rigid impact zones (8 rounds)" & b=$!
wait $a; wait $b
ffmpeg -y -loglevel error -i "$OUT/$tag/soft0/labelled.mp4" -i "$OUT/$tag/soft8/labelled.mp4" \
  -filter_complex hstack=inputs=2 -c:v libx264 -pix_fmt yuv420p -crf 23 "$OUT/$tag.mp4"
echo "wrote $OUT/$tag.mp4"
