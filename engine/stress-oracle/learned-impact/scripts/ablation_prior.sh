#!/bin/zsh
# Matched ablation (planning session): variant {B, A} × Hertz QS prior {with, without},
# same data, seeds, steps and schedule; each scored with gates-v1.1 on validation.
#   ablation_prior.sh <calls-v4-train> <calls-v4-eval>
set -e
RPID=
TRAIN=$1; EVAL=$2
HERE=${0:a:h}/../runs/v4/ablation-prior; mkdir -p $HERE
cd ${0:a:h}/../python
PY=.venv/bin/python
DROP=bond_feat:log_qs_hertz_util
for seed in 0 1; do
  for cfg in B_prior B_noprior A_prior A_noprior; do
    O=$HERE/$cfg/seed$seed
    [ -f $O/report-val.json ] && continue
    v=${cfg%%_*}; d=""; [[ $cfg == *noprior ]] && d="--drop-features $DROP"
    if [ ! -f $O/best.pt ] || [ ! -f $O/done ]; then
      echo "== train $cfg seed $seed"
      $PY -u -m learned_impact.outcome.train --dataset $TRAIN --val-dataset $EVAL --out $O \
        --variant $v ${=d} --seeds $seed --steps 8000 --eval-every 2000 --pushforward 0.3 --device mps
      touch $O/done
    fi
    [ -n "$RPID" ] && wait $RPID  # one report at a time (each holds the eval set in memory)
    echo "== report $cfg seed $seed (background, CPU)"
    ($PY -u -m learned_impact.outcome.report --checkpoint $O/best.pt --dataset $EVAL --split val \
       --out $O/report-val.json > $O/report-val.log 2>&1 || echo "report failed: $cfg $seed") &
    RPID=$!
  done
done
wait
echo ABLATION_DONE
