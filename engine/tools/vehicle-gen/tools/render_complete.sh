#!/bin/sh
# The "is everything there?" stills of a solid-build sedan: the exploded view by system and what
# each removable part reveals (hood, each door, boot lid, bumpers, a fender, a wheel), a roof
# cutaway and the underside.
#
#     tools/render_complete.sh <car.glb> <out_dir> [samples]
#
# A door is taken off with what rides it (glass, card, handle, mirror, intrusion beam, window
# regulator); its hinges and latch stay on the body. Needs Blender (BLENDER, or the macOS app).
set -e
GLB=$1
OUT=$2
SAMPLES=${3:-32}
BLENDER=${BLENDER:-/Applications/Blender.app/Contents/MacOS/Blender}
HERE=$(dirname "$0")
door() { echo "all,-door_$1,-glass_door_$1,-door_trim_$1,-handle_$1,-door_beam_$1,-window_regulator_$1,-mirror_$2"; }
"$BLENDER" -b --python "$HERE/render_parts.py" -- "$GLB" "$OUT" \
  "exploded_systems=systems:1,0.45,0.9" \
  "exploded_systems_rear=systems:-1,0.3,-0.9" \
  "hood_off=all,-hood:0.45,1,0.75:1.9:0,0.65,1.55" \
  "door_FL_off=$(door FL L):1,0.3,0.45:2.3:0.6,0.7,0.35" \
  "door_RL_off=$(door RL -):1,0.3,-0.35:2.3:0.6,0.7,-0.65" \
  "door_FR_off=$(door FR R):-1,0.3,0.45:2.3:-0.6,0.7,0.35" \
  "door_RR_off=$(door RR -):-1,0.3,-0.35:2.3:-0.6,0.7,-0.65" \
  "trunk_off=all,-trunk_lid:0.3,0.9,-1:2.1:0,0.85,-2.0" \
  "bumper_front_off=all,-bumper_front,-grille,-plate_front:0.5,0.35,1:2.0:0,0.5,2.2" \
  "bumper_rear_off=all,-bumper_rear:0.5,0.3,-1:2.0:0,0.5,-2.35" \
  "fender_FL_off=all,-fender_FL:1,0.45,0.75:2.2:0.7,0.6,1.5" \
  "wheel_FL_off=all,-rim_FL,-tyre_FL,-hub_cap_FL:1,0.15,0.35:3.4:0.8,0.36,1.42" \
  "wheel_RL_off=all,-rim_RL,-tyre_RL,-hub_cap_RL:1,0.15,-0.35:3.4:0.8,0.36,-1.42" \
  "roof_cutaway=all,-headliner,cut@y<1.02:0.4,1,0.3:1.45:0,0.6,-0.3" \
  "underside=all:0.25,-1,0.15" \
  --samples "$SAMPLES" --res 1500x900
