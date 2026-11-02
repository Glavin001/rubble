#!/bin/sh
# The acceptance runs (engine/docs/explicit-beams-acceptance): the four test
# vehicles, idle / drive at 100 km/h / 100 km/h into a wall, at 120 fps, 1 to 256
# vehicles on Rubble's GPU and Rigs of Rods (ROR_BIN: the local reference harness,
# inspiration/ror-harness), 5 trials, each run on a quiet machine. Agora S and L are
# RigsOfRods/content (GPLv3, not in this repository): CONTENT names their folder.
# Run from engine/: sh docs/explicit-beams-acceptance/run.sh
set -e
D=docs/explicit-beams-acceptance
: "${ROR_BIN:?set ROR_BIN to the ror_ref harness binary}"
: "${CONTENT:?set CONTENT to the RigsOfRods/content folder}"
cargo build --release -p engine-webgpu --bin explicit_beams_scaling
for v in "agora_s:$CONTENT/agora/95bbUID-agoras.truck" \
         "agora_l:$CONTENT/agora/95bbUID-agoral.truck" \
         "sedan:$D/vehicles/rubble_sedan_ror.truck" \
         "jbeam_car:$D/vehicles/rubble_hires_jbeam.truck"; do
    name=${v%%:*}
    file=${v#*:}
    mkdir -p "$D/$name"
    ./target/release/explicit_beams_scaling "$file" ror="$ROR_BIN" \
        scenarios=idle,drive,impact counts=1,2,4,8,16,32,64,128,256 rubble_counts= \
        fps=120 speed=27.78 drag=0 trials=5 quiet_wait=86400 out="$D/$name" \
        > "$D/$name/stdout.txt" 2> "$D/$name/stderr.txt"
done
