#!/bin/bash
# The frame's choice of kernel (`--reach`) in the emulator against the reference, bit for
# bit: scenes that start out of reach, come into it and leave it; with and without surface
# contact; proxies at rest, falling and spinning; the world's fixed shapes. `--reach-eager`
# gives a frame to the contact-free kernel whenever the body starts it clear by the margin,
# so the frame changes kernel part way (the hand-over at many substeps). A minute.
cd "$(dirname "$0")/../.."
B=${B:-./target/fast/rigid}
F=${F:-./target/fast/floor}
run() { echo "== rigid $*"; perl -e 'alarm(1500);exec(@ARGV)' "$B" --emu "$@" 2>&1 | grep -E "emu:|frames by|first diff|rror|panick" | tail -3; }
frun() { echo "== floor $*"; perl -e 'alarm(1500);exec(@ARGV)' "$F" --v2 --emu "$@" 2>&1 | grep -E "emu:|frames by|first diff|rror|panick" | tail -3; }
run --scene post --surface --reach --reach-eager --gap 0.8 --frames 10
run --scene post --surface --reach --reach-eager --gap 0.8 --frames 10 --reverse
run --scene bricks --surface --reach --reach-eager --gap 0.9 --frames 12
run --scene rocks --surface --reach --reach-eager --gap 0.7 --frames 12
run --scene bricks --reach --reach-eager --gap 0.9 --frames 12
run --scene rocks --reach --gap 0.7 --frames 12
run --scene post --surface --reach --speed 0 --frames 6
run --scene post --surface --reach --reach-eager --speed 30 --gap 2.0 --frames 8
# (More hand-over substeps: the distance to the shapes varied.)
for g in 0.5 0.6 0.7 1.0 1.1 1.25; do run --scene bricks --surface --reach --reach-eager --gap $g --frames 9; done
for g in 0.55 0.75 1.05; do run --scene post --reach --reach-eager --gap $g --speed 17 --frames 6; done
# (The post struck and left behind, turning: the proxies' free flight with a spin in the
# contact-free kernel's frames after the contact.)
run --scene post --surface --reach --gap 0.4 --frames 45
run --scene post --reach --reach-eager --gap 0.4 --frames 45
# (The world's fixed shapes with surface contact, no proxies; a wall by node contact,
# which the contact-free kernel keeps.)
frun --pole 1.2 --speed 11 --surface --frames 14 --reach
frun --pole 1.2 --speed 11 --surface --frames 14 --reach --reach-eager
frun --pole 0.9 --speed 20 --surface --frames 10 --reach --reach-eager --reverse
frun --frames 8 --lift 0.05 --reach
frun --frames 10 --wall 1.0 --speed 16 --reach --reach-eager
echo "== done"
