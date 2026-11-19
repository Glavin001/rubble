#!/bin/bash
# The surface pass in the emulator against the reference: scenes, lane orders, by step.
cd "$(dirname "$0")/../.."
for a in "--scene post --surface --emu --frames 24" "--scene post --surface --emu --frames 12 --reverse" "--scene bricks --surface --emu --frames 14" "--scene rocks --surface --emu --frames 16" "--scene post --surface --emu --frames 2 --by-step" "--scene bricks --emu --frames 6" "--scene post --speed 25 --surface --emu --frames 14"; do
  echo "== rigid $a"
  ./target/fast/rigid $a 2>&1 | grep -v "^warning" | tail -4
done
