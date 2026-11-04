#!/bin/sh
# The acceptance runs (engine/docs/explicit-beams-acceptance): the four test
# vehicles, idle / drive at 100 km/h / 100 km/h into a wall, at 120 fps, 1 to 256
# vehicles on Rubble's GPU and Rigs of Rods (ROR_BIN: the local reference harness,
# inspiration/ror-harness), 5 trials alternating which engine runs first, each run
# on a quiet machine; and per scenario the accuracy ensembles (ACCURACY-RULE.md).
# Agora S and L are RigsOfRods/content (GPLv3, not in this repository): CONTENT
# names their folder.
#
# Bounded pieces: one process per vehicle and scenario, written to
# <vehicle>/<scenario>/. Each run inside a piece lasts seconds to a few minutes;
# RUN_LIMIT (default 21600 s, quiet-machine waits included) kills a runaway piece; the
# waits dominate on a busy machine, each timed run itself lasts seconds to minutes.
# A finished piece (its summary.md) is skipped, so the script resumes where it
# stopped. ONLY=<vehicle> runs one vehicle.
# Run from engine/: sh docs/explicit-beams-acceptance/run.sh
set -e
D=docs/explicit-beams-acceptance
: "${ROR_BIN:?set ROR_BIN to the ror_ref harness binary}"
: "${CONTENT:?set CONTENT to the RigsOfRods/content folder}"
LIMIT=${RUN_LIMIT:-21600}
cargo build --release -p engine-webgpu --bin explicit_beams_scaling
# The BeamNG-resolution JBeam car first: the target class of vehicle.
for v in "jbeam_car:$D/vehicles/rubble_hires_jbeam.truck" \
         "agora_s:$CONTENT/agora/95bbUID-agoras.truck" \
         "agora_l:$CONTENT/agora/95bbUID-agoral.truck" \
         "sedan:$D/vehicles/rubble_sedan_ror.truck"; do
    name=${v%%:*}
    file=${v#*:}
    [ -n "$ONLY" ] && [ "$ONLY" != "$name" ] && continue
    for sc in idle drive impact; do
        out="$D/$name/$sc"
        [ -f "$out/summary.md" ] && continue
        mkdir -p "$out"
        # Wait for a quiet machine before starting the piece (the tool also waits
        # before every run inside it).
        while [ "$(sysctl -n vm.loadavg | awk '{print ($2 < 3)}')" != 1 ]; do sleep 30; done
        perl -e "alarm $LIMIT; exec @ARGV" ./target/release/explicit_beams_scaling "$file" \
            ror="$ROR_BIN" scenarios=$sc counts=1,2,4,8,16,32,64,128,256 rubble_counts= \
            fps=120 speed=27.78 drag=0 trials=5 nudges=8 quiet_wait=86400 out="$out" \
            > "$out/stdout.txt" 2> "$out/stderr.txt" || echo "$name $sc: stopped ($?)" >&2
    done
done
