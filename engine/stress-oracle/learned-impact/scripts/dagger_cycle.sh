#!/bin/zsh
# Routine DAgger, a standing step after each checkpoint (planning, 2026-10-22):
#   dagger_cycle.sh <checkpoint> <round name>
# 1. N training events (default 120, spread over the families round-robin), in parts of
#    PART events (default 20): harness-A rollouts of each part with the checkpoint (P0's
#    wrapper $DAGGER_ROLLOUT: `<wrapper> <checkpoint> <python dir> <out> <n> <events list>`);
# 2. the part's rollout states relabelled by the oracle's island step ($DAGGER_RELABEL),
#    keeping only the calls admitted by energy balance (the loader adds the energy envelope);
#    then the part's island states are deleted: a round's states take ~2.7 GB at once, which
#    the shared 30 GB data budget does not have;
# 3. the round appended to runs/data/dagger-rounds.list: the next run's --extra-train is
#    `$(paste -sd, runs/data/dagger-rounds.list)`;
# 4. its counts appended to findings/p2-dagger-rounds.md.
# Training events only (P0's runner and export-rollouts refuse any other split).
set -e
CK=${1:?checkpoint}; NAME=${2:?round name}
L=${0:a:h}/..
ROLL=${DAGGER_ROLLOUT:?P0 rollout wrapper}
REL=${DAGGER_RELABEL:?P0 relabel binary}
COLS=${DAGGER_COLUMNS:?a /5 calls manifest whose full-state columns the relabels use}
EVENTS=${DAGGER_EVENTS:-$L/runs/data/v6-train-events.list}
N=${DAGGER_N:-120}; PART=${DAGGER_PART:-20}; SHARE=${DAGGER_SHARE:-0.15}
PYD=${DAGGER_PYTHON:-$L/python}
D=$L/runs/dagger-v5/$NAME; W=$L/runs/dagger-work-$NAME
mkdir -p $D/chunks $W
python3 - $EVENTS $N $PART $W <<'PY'
import sys, collections
ev, n, part, w = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
by = collections.defaultdict(list)
for x in (l.strip() for l in open(ev) if l.strip()):
    by[x.rsplit("/", 3)[-3]].append(x)
groups = [sorted(v) for _, v in sorted(by.items())]
quota = [n // len(groups) + (1 if j < n % len(groups) else 0) for j in range(len(groups))]
pick = []  # evenly spaced within each family; families interleaved
cols = [[g[int(i * len(g) / q)] for i in range(min(q, len(g)))] for g, q in zip(groups, quota)]
for i in range(max(len(c) for c in cols)):
    pick += [c[i] for c in cols if i < len(c)]
for k in range(0, len(pick), part):
    open(f"{w}/part{k // part:02d}.list", "w").write("\n".join(pick[k:k + part]) + "\n")
print(f"{len(pick)} events in {(len(pick) + part - 1) // part} parts")
PY
cd $L/python
for P in $W/part*.list; do
  B=${P:t:r}; OUT=$W/$B
  [ -f $D/chunks/$B.done ] && continue
  [ -f $OUT/rollouts.json ] || $ROLL $CK $PYD $OUT $(wc -l < $P) $P
  [ -f $W/$B-relabel/jobs.json ] || .venv/bin/python -m learned_impact.outcome.dagger export-rollouts --rollouts-json $OUT/rollouts.json --out $W/$B-relabel
  .venv/bin/python ../scripts/relabel_chunked.py --round $W/$B-relabel --bin $REL --full-state-columns $COLS --threads 4
  python3 - $OUT/rollouts.json $W/$B-relabel $D/chunks/$B.json <<'PY'
import json, glob, sys
r = json.load(open(sys.argv[1]))["events"]
t = {"events": len(r), "states": len(json.load(open(f"{sys.argv[2]}/jobs.json"))), "relabelled": 0, "admitted": 0, "rejected_jobs": 0}
for f in glob.glob(f"{sys.argv[2]}/chunks/*/summary.json"):
    d = json.load(open(f)); [t.__setitem__(k, t[k] + d[k]) for k in ("relabelled", "admitted", "rejected_jobs")]
json.dump(t, open(sys.argv[3], "w"))
PY
  mv $W/$B-relabel/chunks $D/chunks/$B
  rm -rf "${OUT:?}/states"   # the part's island states: relabelled, no longer needed
  touch $D/chunks/$B.done
done
SUM=$(python3 - $D/chunks $NAME <<'PY'
import json, glob, sys, subprocess
t = {"events": 0, "states": 0, "relabelled": 0, "admitted": 0, "rejected_jobs": 0}
for f in glob.glob(f"{sys.argv[1]}/part*.json"):
    d = json.load(open(f)); [t.__setitem__(k, t[k] + d[k]) for k in t]
du = subprocess.run(["du", "-sh", sys.argv[1]], capture_output=True, text=True).stdout.split()[0]
print(f"| {sys.argv[2]} | {t['events']} | {t['states']} | {t['rejected_jobs']} | {t['relabelled']} | {t['admitted']} | {du} |")
PY
)
grep -qx "$D/chunks@$SHARE" $L/runs/data/dagger-rounds.list 2>/dev/null || echo "$D/chunks@$SHARE" >> $L/runs/data/dagger-rounds.list
F=$L/findings/p2-dagger-rounds.md
grep -q "^| $NAME |" $F || echo "$SUM" >> $F
echo "DAGGER_ROUND_DONE $NAME: next run --extra-train $(paste -sd, $L/runs/data/dagger-rounds.list)"
