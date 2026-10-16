#!/usr/bin/env bash
# The stress-ref test suite without its long scenario runs, for iterating on the solver.
# Same tests, minus the ones that simulate a whole breaching panel, the hull scene packs
# and the showcases (minutes each). Run the full suite before committing:
#   cargo test --release -p stress-ref --no-fail-fast
# Method switches apply as usual (the proven methods are the default; the established
# ones for comparison): STRESS_METHODS=established scripts/test_fast.sh
set -euo pipefail
cd "$(dirname "$0")/.."
exec cargo test --release -p stress-ref --no-fail-fast "$@" -- \
  --skip same_outcome_at_different_frame_rates \
  --skip converges_as_the_timestep_shrinks \
  --skip same_panel_two_chunk_sizes_same_outcome \
  --skip solve_modes_agree \
  --skip pressure_panel_matches_opencourant \
  --skip hull_boxes_match_boxes \
  --skip scene_packs_import_and_stand \
  --skip cases::
