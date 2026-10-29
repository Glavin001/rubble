#!/usr/bin/env bash
# Run every World destruction-reaction scenario (engine-world examples/reaction_replay.rs) and
# record each replay to MP4 plus a contact sheet (record.mjs). Re-run after merging new
# reaction wiring (e.g. blast->crush, support-loss->crush) to see what changed.
#
#   bash engine/tools/reaction-replay/record-all.sh [OUT_DIR] [scenario ...]
#
# Scenarios default to all of: building-blast-20 building-blast-60 slab-columns-on
# slab-columns-off block-columns-on block-columns-off crush-wall-conserve.
# One GPU job at a time: the simulations run first, then the recordings.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ENGINE="$(cd "$HERE/../.." && pwd)"
OUT="${1:-${TMPDIR:-/tmp}/rubble-reactions}"
shift || true
ALL=(building-blast-20 building-blast-60 slab-columns-on slab-columns-off block-columns-on block-columns-off crush-wall-conserve)
RUNS=("${@:-${ALL[@]}}")
mkdir -p "$OUT/json"
(cd "$ENGINE" && cargo build --release -p engine-world --example reaction_replay)
BIN="$ENGINE/target/release/examples/reaction_replay"
JSONS=()
for run in "${RUNS[@]}"; do
  case "$run" in
    building-blast-*) args=(building-blast --tnt "${run##*-}") ;;
    slab-columns-on) args=(slab-columns) ;;
    slab-columns-off) args=(slab-columns --support-loss off) ;;
    block-columns-on) args=(block-columns) ;;
    block-columns-off) args=(block-columns --support-loss off) ;;
    crush-wall-conserve) args=(crush-wall --crush conserve) ;;
    crush-wall-remove) args=(crush-wall --crush remove) ;;
    *) echo "unknown scenario $run" >&2; exit 1 ;;
  esac
  "$BIN" "${args[@]}" --out "$OUT/json/$run.json"
  JSONS+=("$OUT/json/$run.json")
done
if [[ -f "$ENGINE/../.nvmrc" && -s "${NVM_DIR:-$HOME/.nvm}/nvm.sh" ]]; then
  # shellcheck disable=SC1091
  source "${NVM_DIR:-$HOME/.nvm}/nvm.sh" >/dev/null
  pushd "$ENGINE/.." >/dev/null && nvm use >/dev/null && popd >/dev/null
fi
node "$HERE/record.mjs" "${JSONS[@]}" --out "$OUT"
