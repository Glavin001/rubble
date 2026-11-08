#!/bin/sh
# Interleaved A/B of explicit_vehicles_bench (the crash town, World steps): for each car
# count, `rounds` alternating runs of the base and branch builds, each printing its GPU
# median ms per World step (and the explicit frame's mean with EV_PHASES=1).
#   bench_ab.sh <base bench> <branch bench> "<counts>" [rounds]
# Env passes through (EV_PAGE, EV_ETK, EV_HZ, EV_SECS, EV_PHASES, ...). Each run under
# bounded.sh (the GPU lock and a kill timer).
base="$1"; branch="$2"; counts="$3"; rounds="${4:-3}"
here="$(dirname "$0")"
for n in $counts; do
  i=0
  while [ "$i" -lt "$rounds" ]; do
    for which in base branch; do
      bin=$base; [ "$which" = branch ] && bin=$branch
      line=$(EV_ALONE=0 "$here/bounded.sh" 900 "$bin" "$n" 2>&1 | grep -E "^$n vehicles" -A1)
      med=$(echo "$line" | head -1 | sed -n 's/.*GPU median \([0-9.]*\) ms.*/\1/p')
      frame=$(echo "$line" | sed -n 2p | sed "s/.*per step): //" | cut -c1-110)
      echo "n=$n round=$i $which step_median=$med | $frame"
    done
    i=$((i + 1))
  done
done
