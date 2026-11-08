#!/bin/sh
# Film a rigged design on the CPU reference and render the clips with
# Blender (engine/docs/generated-vehicles.md). From the repo root:
#   sh engine/tools/vehicle-gen/tools/film_all.sh [design] [artifacts dir]
# Frames go to out/film/<design>/<scene> (gitignored); the mp4s and stills to
# the artifacts dir (default ~/Development/rubble-artifacts/vehicle-gen-physics).
set -e
D=${1:-sedan}
ART=${2:-$HOME/Development/rubble-artifacts/vehicle-gen-physics}
ROOT=$(pwd)
B=${BLENDER:-/Applications/Blender.app/Contents/MacOS/Blender}
mkdir -p "$ART/stills"
(cd engine && cargo build --release -p engine-softbody --example vehicle_film)
for S in drive wall pole c2c; do
  F="$ROOT/out/film/$D/$S"
  [ -f "$F/scene.json" ] || "$ROOT/engine/target/release/examples/vehicle_film" "$S" "$F" "$D"
  (cd engine/tools/vehicle-gen && python3 ../../scripts/with-gpu-lock.py --limit 900 -- \
    "$B" -b --python render_film.py -- "$F" "$ART/$D-$S.mp4" --samples 32 --stills "$ART/stills" > "$F/render.log" 2>&1)
  echo "$D $S: $(tail -1 "$F/scene.json" >/dev/null; grep -o '"broken": \[[^]]*\]' "$F/scene.json") -> $ART/$D-$S.mp4"
done
for S in wall c2c; do
  F="$ROOT/out/film/$D/$S"
  (cd engine/tools/vehicle-gen && python3 ../../scripts/with-gpu-lock.py --limit 900 -- \
    "$B" -b --python render_film.py -- "$F" "$ART/$D-$S-orbit.mp4" --orbit 8 --samples 32 --stills "$ART/stills" > "$F/render-orbit.log" 2>&1)
  echo "$D $S orbit -> $ART/$D-$S-orbit.mp4"
done
