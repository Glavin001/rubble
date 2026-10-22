#!/bin/zsh
# Viability runs (planning session, 2026-10-22) on P1's full-state wall calls:
#   viability.sh <train calls> <eval calls> [steps]
# 1. W64/12 (default) and W128/16, variant B with hints; 2. data scaling at 25/50/100% of the
# training groups (W64, seeds 0,1); 3. gates-v1.1 reports; 4. the report in findings/.
# MPS jobs run one at a time; reports run on the CPU behind them.
set -e
TRAIN=$1; EVAL=$2; STEPS=${3:-12000}
PYD=${0:a:h}/../python; R=${0:a:h}/../runs/v5; mkdir -p $R
cd $PYD
PY=.venv/bin/python
train() {  # name, extra args
  local O=$R/$1; shift
  [ -f $O/best.pt ] && [ -f $O/done ] && return
  $PY -u -m learned_impact.outcome.train --dataset $TRAIN --val-dataset $EVAL --out $O \
    --steps $STEPS --eval-every $((STEPS / 12)) --pushforward 0.3 --device mps "$@"
  touch $O/done
}
report() {
  [ -f $R/$1/report-val.json ] || $PY -u -m learned_impact.outcome.report --checkpoint $R/$1/best.pt \
    --dataset $EVAL --split val --out $R/$1/report-val.json
}
train w64
(report w64 > $R/w64-report.log 2>&1) &
train w128 --large
wait
(report w128 > $R/w128-report.log 2>&1) &
$PY -u -m learned_impact.outcome.pilot --dataset $TRAIN --val-dataset $EVAL --out $R/scaling \
  --fractions 0.25,0.5,1 --seeds 0,1 --variants B --steps-per-event 12 --min-steps 2000 \
  --max-steps $STEPS --device mps --pushforward 0.3
wait
$PY -m learned_impact.outcome.viability --runs W64=$R/w64 W128=$R/w128 --scaling $R/scaling \
  --out ${0:a:h}/../findings/p2-viability-$(date +%Y-%m-%d).md
echo VIABILITY_DONE
