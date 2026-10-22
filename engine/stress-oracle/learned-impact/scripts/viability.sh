#!/bin/zsh
# Viability runs (planning session, 2026-10-22), local on MPS, standalone-oracle data only:
#   viability.sh <train calls> <eval calls> [steps]
# 1. W64/12 (the default), seeds 0,1,2; 2. W128/16, seed 0; 3. data scaling at 25/50/100% of
# the training groups (W64, seeds 0,1,2); gates-v1.1 reports on CPU behind each; 4. the report
# in findings/. Every GPU phase holds the machine-wide GPU lock (scripts/gpu_lock.sh).
# Pushforward stays off until its full-rebuild tests pass.
set -e
TRAIN=$1; EVAL=$2; STEPS=${3:-12000}; EXTRA=${4:-}  # e.g. --full-state for /5
HERE=${0:a:h}
PYD=$HERE/../python; R=$HERE/../runs/v5; mkdir -p $R
cd $PYD
PY=.venv/bin/python
LOCK=$HERE/gpu_lock.sh
train() {  # name, extra args
  local name=$1; shift
  [ -f $R/$name/done ] && return
  $LOCK "p2 viability $name" -- $PY -u -m learned_impact.outcome.train --dataset $TRAIN \
    --val-dataset $EVAL --out $R/$name --steps $STEPS --eval-every $((STEPS / 12)) --device mps ${=EXTRA} "$@"
  touch $R/$name/done
}
report() {  # run dir (CPU; one at a time in the background)
  [ -f $1/report-val.json ] || $PY -u -m learned_impact.outcome.report --checkpoint $1/best.pt \
    --dataset $EVAL --split val --out $1/report-val.json > $1/report-val.log 2>&1
}
RP=
for s in 0 1 2; do
  train w64-s$s --seeds $s
  [ -n "$RP" ] && wait $RP; (report $R/w64-s$s) & RP=$!
done
train w128-s0 --large --seeds 0
[ -n "$RP" ] && wait $RP; (report $R/w128-s0) & RP=$!
$LOCK "p2 viability scaling" -- $PY -u -m learned_impact.outcome.pilot --dataset $TRAIN \
  --val-dataset $EVAL --out $R/scaling --fractions 0.25,0.5,1 --seeds 0,1,2 --variants B \
  --steps-per-event 12 --min-steps 3000 --max-steps $STEPS --device mps ${=EXTRA}
wait
$PY -m learned_impact.outcome.viability --runs W64-s0=$R/w64-s0 W64-s1=$R/w64-s1 W64-s2=$R/w64-s2 \
  W128-s0=$R/w128-s0 --scaling $R/scaling --out $HERE/../findings/p2-viability-$(date +%Y-%m-%d).md
echo VIABILITY_DONE
