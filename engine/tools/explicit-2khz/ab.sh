#!/bin/sh
# Interleaved A/B of the explicit frame on one saved crash-town state, and its bit
# identity: explicit_frame_dump from two builds.
#
#   ab.sh <base dump binary> <branch dump binary> <cars> <steps before> <work dir> [rounds]
#
# Saves the state once (base build), runs 3 frames with each build and compares every
# state word (cmp), then times `rounds` (default 5) alternating runs of each
# (`explicit_frame_dump time`, 9 x 1 frame each) and prints each run's median.
# Env passes through (EV_ETK, EV_PAGE, EV_NO_CONTACT, EV_HZ, ...). One GPU run at a
# time; each bounded by bounded.sh.
set -e
base="$1"; branch="$2"; cars="$3"; before="$4"; dir="$5"; rounds="${6:-5}"
here="$(dirname "$0")"
mkdir -p "$dir"
state="$dir/state.bin"
[ -f "$state" ] || "$here/bounded.sh" 600 "$base" save "$cars" "$before" "$state" 2>&1 | grep -v warn
"$here/bounded.sh" 600 "$base" run "$cars" "$state" "$dir/base.txt" 3 2>&1 | grep -v warn
"$here/bounded.sh" 600 "$branch" run "$cars" "$state" "$dir/branch.txt" 3 2>&1 | grep -v warn
if cmp -s "$dir/base.txt" "$dir/branch.txt"; then echo "BITWISE: equal ($(wc -l < "$dir/base.txt") lines)"; else echo "BITWISE: DIFFER"; cmp "$dir/base.txt" "$dir/branch.txt" | head -3; fi
i=0
while [ "$i" -lt "$rounds" ]; do
  b=$(RUBBLE_GPU_TIMING=1 "$here/bounded.sh" 600 "$base" time "$cars" "$state" 9 2>&1 | grep 'frame ms' | awk '{print $4}')
  n=$(RUBBLE_GPU_TIMING=1 "$here/bounded.sh" 600 "$branch" time "$cars" "$state" 9 2>&1 | grep 'frame ms' | awk '{print $4}')
  echo "round $i base $b branch $n"
  i=$((i + 1))
done
