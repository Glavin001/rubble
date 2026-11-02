#!/bin/sh
# Accuracy only (acceptance item 2): one vehicle per run, each test vehicle and
# scenario, an ensemble of 8 nudged runs per engine (spawn speed + 0.1 mm/s × k),
# Rubble's GPU against Rigs of Rods (explicit_beams_scaling: Accuracy). Each
# vehicle is killed after 15 minutes. One trial, no quiet-machine gate: the frame times in these results
# are not acceptance timings (run.sh measures those).
# Run from engine/: ROR_BIN=... CONTENT=... sh docs/explicit-beams-acceptance/accuracy/run.sh
set -e
D=docs/explicit-beams-acceptance
: "${ROR_BIN:?set ROR_BIN to the ror_ref harness binary}"
: "${CONTENT:?set CONTENT to the RigsOfRods/content folder}"
cargo build --release -p engine-webgpu --bin explicit_beams_scaling
for v in "jbeam_car:$D/vehicles/rubble_hires_jbeam.truck" \
         "agora_s:$CONTENT/agora/95bbUID-agoras.truck" \
         "agora_l:$CONTENT/agora/95bbUID-agoral.truck" \
         "sedan:$D/vehicles/rubble_sedan_ror.truck"; do
    name=${v%%:*}
    file=${v#*:}
    # Done already (rerun: delete its summary.md).
    [ -f "$D/accuracy/$name/summary.md" ] && continue
    mkdir -p "$D/accuracy/$name"
    perl -e "alarm 900; exec @ARGV" nice -n 10 ./target/release/explicit_beams_scaling "$file" ror="$ROR_BIN" \
        scenarios=idle,drive,impact counts=1 rubble_counts= \
        fps=120 speed=27.78 drag=0 trials=1 nudges=8 max_load=1000 max_gpu=100000 out="$D/accuracy/$name" \
        > "$D/accuracy/$name/stdout.txt" 2> "$D/accuracy/$name/stderr.txt"
done
