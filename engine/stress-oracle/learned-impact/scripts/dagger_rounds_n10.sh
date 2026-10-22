#!/bin/zsh
# DAgger rounds on the 10 memorised events (/3 inputs), the overfit milestone's closed-loop
# criterion. Round k: fine-tune from round k-1 on the originals + every admitted relabel so
# far (to saturation), roll out in harness A with island-state dumps (P0's runner, CPU),
# relabel every rollout state with the oracle's island step (P0's rev-5 binary), repeat.
set -e
L=${0:a:h}/..
D=$L/runs/dagger-n10
P0D=/Users/glavin/Development/rubble/.claude/worktrees/distracted-heisenberg-a066f7/engine/stress-oracle/learned-impact
REL=$P0D/runs/bin/learned_impact_relabel.rev5
export LEARNED_IMPACT_ACCEPT_V3=1
prev=$D/r1/train/last.pt
# relabels of rollout 0 (P0's r0 run) are $D/r1/calls; rollout j>=1 is $D/rj/cl-eval
for k in 2 3 4; do
  P=$D/r$((k-1)); R=$D/r$k; mkdir -p $R
  # 1. relabel the previous round's rollout (the model's own states) with the oracle
  [ -d $P/rr/calls ] || (cd $L/python && .venv/bin/python -m learned_impact.outcome.dagger export-rollouts \
      --rollouts $P/cl-eval/states/a_model --recordings /Users/glavin/Development/rubble/.claude/worktrees/learned-impact-p1-data-aa822c/engine/stress-oracle/learned-impact/runs --dataset $L/runs/data/n10-v3 --out $P/rr && \
    .venv/bin/python -m learned_impact.outcome.dagger relabel --round $P/rr --bin $REL --threads 8 --calls v3)
  # 2. fine-tune on the originals + every admitted relabel so far
  EXTRA=$D/r1/calls
  for j in $(seq 1 $((k-1))); do EXTRA=$EXTRA,$D/r$j/rr/calls; done
  [ -f $R/train/memorisation.json ] || (cd $L/python && .venv/bin/python -u -m learned_impact.outcome.overfit_real \
      --dataset $L/runs/data/n10-v3 --events-json $L/runs/overfit-real/n10/events.json --n-events 10 \
      --extra-calls $EXTRA --init-checkpoint $prev --steps 10000 --lr 5e-4 --out $R/train --device mps)
  # 3. roll out in harness A with island-state dumps (P0's runner, CPU)
  [ -f $R/cl-eval/report.md ] || (cd $P0D/python && STRESS_PROVIDER_CALLS=v3 .venv/bin/python -u -m learned_impact.closed_loop_eval \
      --runs /Users/glavin/Development/rubble/.claude/worktrees/learned-impact-p1-data-aa822c/engine/stress-oracle/learned-impact/runs/p1-bulk-v1-free-t1 /Users/glavin/Development/rubble/.claude/worktrees/learned-impact-p1-data-aa822c/engine/stress-oracle/learned-impact/runs/p1-bulk-v1-anchored-t1 /Users/glavin/Development/rubble/.claude/worktrees/learned-impact-p1-data-aa822c/engine/stress-oracle/learned-impact/runs/p1-bulk-v1-free-t2 \
      --members ../runs/bin/n10-members.json --split val,test --candidates a_model --dump-island-states \
      --checkpoint $R/train/last.pt --python-dir $L/python --bin ../runs/bin/learned_p0.cl16-rev5-dump \
      --out $R/cl-eval --device cpu --jobs 8)
  prev=$R/train/last.pt
done
echo DAGGER_DONE
