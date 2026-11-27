#!/usr/bin/env bash
# The front load path (front-load-path.md) one design step at a time: the generated sedan with
# `load_path` and LOAD_PATH_STEPS set to the steps so far, each measured by front_ab.sh (cuts,
# slow push, S11, S12, elastic energy) into out/gen/sedan_<TAG>/. From engine/:
#
#   tools/sections/load_path_ab.sh            # every rung, in parallel (CPU)
#   tools/sections/load_path_ab.sh lp_c       # one rung
#   python3 -I tools/sections/front_ab.py off lp_c lp_cp lp_cpm lp_cpmu lp_cpmup lp --plot out/sections/load-path
#
# Rungs: lp_c cells (closed cells and braces); lp_cp + the bumper beam's plate; lp_cpm + the
# subframe's mounts (and rule 1); lp_cpmu + the upper rail; lp_cpmup + rules 4 and 5 (stops,
# panels); lp every step (+ the engine's stops). Each step of front_ab.sh is bounded.
set -u
steps() {
  case $1 in
    lp_c) echo cells ;;
    lp_cp) echo cells,plate ;;
    lp_cpm) echo cells,plate,mounts ;;
    lp_cpmu) echo cells,plate,mounts,upper ;;
    lp_cpmup) echo cells,plate,mounts,upper,panels ;;
    *) echo all ;;
  esac
}
tags=("$@")
[ ${#tags[@]} -eq 0 ] && tags=(lp_c lp_cp lp_cpm lp_cpmu lp_cpmup lp)
for t in "${tags[@]}"; do
  LOAD_PATH_STEPS=$(steps "$t") tools/sections/front_ab.sh "$t" load_path >"out/gen/ab_$t.log" 2>&1 &
done
wait
echo "done: ${tags[*]}"
