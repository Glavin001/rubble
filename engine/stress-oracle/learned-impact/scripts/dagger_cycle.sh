#!/bin/zsh
# Routine DAgger, a standing step after each checkpoint (planning, 2026-10-22):
#   dagger_cycle.sh <checkpoint> <round name>
# 1. harness-A rollouts of training events with the checkpoint (P0's wrapper, $DAGGER_ROLLOUT:
#    `<wrapper> <checkpoint> <p2 python dir> <out dir>`, writing <out>/rollouts.json);
# 2. relabel every rollout state with the oracle's island step ($DAGGER_RELABEL, P0's binary),
#    in chunks, keeping only the calls admitted by energy balance (and the training loader's
#    energy-envelope admission);
# 3. append the round to runs/data/dagger-rounds.list: the next run's --extra-train is
#    `$(paste -sd, runs/data/dagger-rounds.list)` (each line <chunks dir>@<share>);
# 4. append its counts to findings/p2-dagger-rounds.md.
# Training events only: P0's runner refuses any other split, export-rollouts too.
set -e
CK=${1:?checkpoint}; NAME=${2:?round name}
L=${0:a:h}/..
ROLL=${DAGGER_ROLLOUT:?P0's rollout wrapper}
REL=${DAGGER_RELABEL:?P0's relabel binary}
COLS=${DAGGER_COLUMNS:-/Users/glavin/Development/rubble/.claude/worktrees/learned-impact-p1-data-aa822c/engine/stress-oracle/learned-impact/runs/calls-v5/p1_anchored_wall_t1_00006_s21001_e0/manifest.json}
SHARE=${DAGGER_SHARE:-0.15}
OUT=$L/runs/dagger-rollouts-$NAME
D=$L/runs/dagger-v5/$NAME
[ -f $OUT/rollouts.json ] || zsh $ROLL $CK $L/python $OUT
cd $L/python
[ -f $D/jobs.json ] || .venv/bin/python -m learned_impact.outcome.dagger export-rollouts --rollouts-json $OUT/rollouts.json --out $D
.venv/bin/python ../scripts/relabel_chunked.py --round $D --bin $REL --full-state-columns $COLS --threads 4
SUM=$(python3 -c "
import json,glob
t={'relabelled':0,'admitted':0,'rejected_jobs':0}
for f in glob.glob('$D/chunks/*/summary.json'):
    d=json.load(open(f)); [t.__setitem__(k,t[k]+d[k]) for k in t]
r=json.load(open('$OUT/rollouts.json'))['events']
print(f\"| $NAME | {len(r)} | {len(json.load(open('$D/jobs.json')))} | {t['rejected_jobs']} | {t['relabelled']} | {t['admitted']} | $(du -sh $D | cut -f1) |\")")
grep -qx "$D/chunks@$SHARE" $L/runs/data/dagger-rounds.list 2>/dev/null || echo "$D/chunks@$SHARE" >> $L/runs/data/dagger-rounds.list
F=$L/findings/p2-dagger-rounds.md
[ -f $F ] || printf "# DAgger rounds (routine, after each checkpoint)\n\nEnergy-envelope admission is applied again at load (the trainer's [data] line gives the final count).\n\n| round (checkpoint) | rollout events | states | jobs rejected | relabelled calls | admitted (energy) | disk |\n|---|---:|---:|---:|---:|---:|---:|\n" > $F
grep -q "^| $NAME |" $F || echo "$SUM" >> $F
echo "DAGGER_ROUND_DONE $NAME: next run --extra-train $(paste -sd, $L/runs/data/dagger-rounds.list)"
