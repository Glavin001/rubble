# P1's recorder and call builder on the main-based branch (2026-10-21)

The planning session chose option (c): `main` is the base. P1's work (branch
`claude/learned-impact-p1-data-aa822c`, commit `2db44e6f`, built on the colleague's
`stress-web-demo` base `5a2f5e4f`) is ported here, together with only the stress-ref hooks it
needs. P1's running generation was not touched.

## What was ported

**stress-learned** (from `2db44e6f`, unchanged; `outcome_calls.rs` from `0c6dfca6`, calls schema /3):
- modules `outcome.rs` (the recorder), `outcome_calls.rs` (the call builder, `live_call`),
  `oracle_config.rs`, `procedural.rs` (the wall generator), `progress.rs`;
- binaries `learned_impact_outcome_record`, `learned_impact_outcome_generate` and
  `learned_impact_outcome_calls`;
- tests `tests/outcome.rs` and `tests/outcome_calls.rs`;
- dependencies `rayon` and `zstd`, and `serde_json` with `float_roundtrip`.

Main's M1 modules (`recorder.rs`, `constants.rs`, `catalog.rs`, the old bins) stay as they
were: the outcome code does not use the colleague's versions.

**stress-ref** (P1's own changes from `5a2f5e4f..2db44e6f`, applied to main):
- `keys.rs` (the one stable-key module).
- `world_branch.rs` (`World::adopt_model`, for the baseline branches).
- `world_coupling.rs`: `PredictedContact` / `PunchPlug` / `predict_contacts`, factored out of
  `predict_punch` with the same arithmetic; `predict_island_members`; routing usable on a
  standalone world. This merged cleanly with the debris-reach routing.
- `joint.rs`, `solver.rs`, `statics.rs`, `refine.rs`: diagnostics, which nothing on the
  established path reads:
  - `JointResponse::fail_ts` / `fail_c`, `kappa_fail`;
  - `RtBond::{damped, fail_ts, fail_c, peak_fail_margin}`, `fail_margin`, `observe_fail_margin`.
- `world.rs`:
  - the per-substep observer (`step_frame_watched`), as a refactor in which `step_frame_once`
    calls `step_frame_once_observed` with a no-op;
  - `chunk_contact_hits` (main already had `contact_hits`);
  - `break_bond` raising the failure margin;
  - the `branch` module and the `PredictedContact` / `PunchPlug` exports.
- `tests/learned_outcome_hooks.rs`.

**From the colleague's base**, only the substep observer. Its stress-provider path, the
V25–V39 modules (`causal_state`, `graph_*`, `runtime_*`, `joint_history`, …), the
`external_stress` hooks and the stress-gpu changes were not taken.

**learned-impact**:
- `FRAMES_SCHEMA.md`;
- P1's pilot findings and audit;
- `python/learned_impact/outcome_{audit,budget,sampling,calls_smoke}.py` and their tests.

**The closed-loop harness** now writes a real call record for every call: `live_call` on a
standalone copy of the frame-start state, written as `req/<frame>_<island>/call/`.
`World::apply_decoded` raises the failure margin of the bonds it breaks, as
`World::break_bond` does, so a margin ≥ 1 still means disconnected. That is the call builder's
check. The per-frame peak-utilisation reset happens after the frame-start snapshot, in the
recorder's order.

## Acceptance

1. **Defaults unchanged**: guard and suites, below.
2. **Recorder parity with P1** (P1's own `recipes.json`, recorder revision `p1-rev6`):

   | Event | End fingerprint (mine = P1's) | Frame arrays | Calls (`learned_impact_outcome_calls`) against P1's `calls-v2` |
   |---|---|---|---|
   | `m5_free_wall` | `9312819624c4b034` = P1 v1 and v2 | | 35.034 m/s, 6 fragments, 31 bonds |
   | `m5_wall_row` | `64931f9911847b7c` = P1 v1 and v2 | | 23.622 m/s, 133 fragments, 401 bonds |
   | `p1_free_wall_t1_00000_s101` | same | 46/46 byte-identical | `calls.json`, decompressed `arrays.bin` and manifest identical (36 calls) |
   | `p1_free_wall_t1_00001_s101` | same | 46/46 | identical (69 calls) |
   | `p1_free_wall_t1_00005_s101` | same | 46/46 | identical (153 calls) |
   | `p1_free_wall_t1_00010_s101` | same | 52/52 | identical |

   P1's own parity recipe, through the generator itself:
   `learned_impact_outcome_generate --family free_wall --tier 1 --n 80 --limit 8 --seed 101 --no-ensembles`
   (the same 8 recipes as P1's `p1-r6-free-t1-a`), then `learned_impact_outcome_calls`.
   For `p1_free_wall_t1_0000[0-7]_s101`, all 8 events match P1's `calls-v2`: end
   fingerprint, `calls.json`, `manifest.json` and even the compressed `arrays.bin.zst`
   are byte-identical (`cmp`).
   With the /3 builder (`0c6dfca6`), the same 8 events against P1's `calls-v3` are also
   byte-identical (`calls.json`, `manifest.json`, `arrays.bin.zst`).
3. **Closed loop with real call records**: `--provider files --echo-oracle` on the wall row
   (12 calls, each with `call/` written by `live_call` in `learned-impact-outcome/3`, QS features on) gives 23.622 / 133 /
   401, bit-identical to the in-process oracle provider.

So P1 can move future generation to this branch. Events recorded on either branch are
identical.

## Guard and suites

All flags at their defaults:

- **Guard**: 32/32 digests identical (`scripts/destruction_peak.py guard`).
- **Suites**: 229 tests pass, 0 fail:
  - stress-ref: 159 passed (`cargo test --release -p stress-ref --no-fail-fast`, including
    `learned_outcome_hooks`);
  - stress-learned: 16 passed (P1's `outcome` and `outcome_calls` tests with the /3 builder,
    and main's M1 tests);
  - stress-gpu: 28 passed (`STRESS_GPU_FIXED_HORIZON=1 STRESS_METHODS=established`,
    one thread);
  - stress-rubble: 26 passed (`--features gpu`, one thread; including `debris_routing` and
    `closed_loop_provider`).
- **Python**: P1's `test_outcome_audit.py` and `test_outcome_sampling.py`, 8 passed.

## Update: P1's storage commits (recorder revision 7)

Taken in full: `5d0a05b2` (the storage decision for the 20 GB budget) and `ef1eac02`
(zstd level 19 in both writers).
- `5d0a05b2` brings zstd per array file, thinned training events, the `traj_*` sidecar and
  `frame_bond_peak_util_prev`, `outcome::regenerate` with the record bin's `--regenerate`,
  the reader's row mapping, the generator's `--budget-gb` / `--min-free-gb` guard and
  `--no-thinning`, and the audit's `.zst` and thinned-event checks.
- Both touch only stress-learned and the Python audit. Every touched file is
  byte-identical to P1's `ef1eac02`, apart from `lib.rs` (main's module list, plus P1's
  `recursion_limit`) and `catalog.rs`.
- Tests: stress-learned 17 pass, including
  `a_thinned_training_event_regenerates_byte_identically_and_builds_the_same_calls`.
  P1's Python audit and sampling tests pass (8).
- Cross-branch check on P1's bulk recipe `p1_free_wall_t1_00000_s1001` (the first recipe
  of `--n 900 --seed 1001`, a thinned training event with 15 of 32 frames kept):
  - recorded here, the event directory is byte-identical to P1's `p1-bulk-v1-free-t1`
    event, apart from the wall-clock `cost` fields in `outcome.json`;
  - `--regenerate` gives byte-identical full events from my thinned event and from P1's;
  - the calls built from them are identical (`calls.json`, `manifest.json`,
    `arrays.bin.zst`);
  - the thinned event's 15 calls match the regenerated event's calls of the same frames.
