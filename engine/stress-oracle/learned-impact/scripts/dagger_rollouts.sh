#!/bin/bash
# DAgger rollouts of a checkpoint in harness A (P0), for P2's standing DAgger step (the
# tracked copy of runs/bin/dagger_rollouts.sh; bins from runs/bin):
#   dagger_rollouts.sh CHECKPOINT P2_PYTHON_DIR OUT_DIR [N=120] [EVENTS_LIST]
# The harness-A settings of the closed-loop evaluation's a_model_bounds column: decoded state
# /2 with the model.json's full-state columns, the bodies' ids from the decode, bodies derived from the predicted hidden state
# where sent, the cumulative energy budget with the contacts' energy, non-penetration, debris
# routing (set by dagger_rollouts.py). Training events only (dagger_rollouts.py refuses any
# other split). Writes OUT_DIR/rollouts.json, states/, runs/ and the job's progress.log and
# status.json (learned-impact/runs/<basename OUT_DIR>/ when OUT_DIR is under runs/).
# Relabel the states with learned_impact_relabel.rev9 (builder revision 9: E_contact_start/end).
set -euo pipefail
here="$(cd "$(dirname "$0")/../runs/bin" && pwd)"
li="$(cd "$here/../.." && pwd)"
[ $# -ge 3 ] || { sed -n '2,3p' "$0"; exit 2; }
ckpt="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
pydir="$(cd "$2" && pwd)"
mkdir -p "$3"; out="$(cd "$3" && pwd)"
n="${4:-120}"
events="${5:-/Users/glavin/Development/rubble/.claude/worktrees/learned-impact-p2-model-7cf306/engine/stress-oracle/learned-impact/runs/data/v6-train-events.list}"
export STRESS_FULL_IDS=decode STRESS_FULL_BODIES=derive STRESS_FULL_ENERGY_BUDGET=1 STRESS_FULL_CONTACT_BOUNDS=1
cd "$li/python"
exec python3 ../scripts/job.py "$(basename "$out")" -- .venv/bin/python -u -m learned_impact.dagger_rollouts \
  --events "$events" --n "$n" --checkpoint "$ckpt" --python-dir "$pydir" \
  --bin "$here/learned_p0.cl31-rev9" --out "$out" --jobs 4
