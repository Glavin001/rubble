# `learned-impact-frames/1`: the impact-outcome frame record

The P1 dataset of `docs/learned-stress-solver/learned-stress-solver-plan.md`. One event is
one scenario's **whole life cycle**, recorded frame by frame (1/60 s) by
`learned_impact_outcome_record` / `learned_impact_outcome_generate`
(`crates/stress-learned/src/outcome.rs`). Each frame holds the model's **causal inputs**
(taken before the step), the **oracle's outcome**, and the **baselines' outcomes on the same
frame-start state**, so one dataset serves the plan's variants A (impact-only), B (residual
over QS) and C (unified).

## Producing it

```bash
cd engine/stress-oracle
cargo build --release -p stress-learned --bins
# plan only (writes runs/<job>/recipes.json, which must match exactly on resume):
target/release/learned_impact_outcome_generate --job p1-free-t1 --family free_wall --tier 1 --n 500 --seed 1 --plan-only
# record (resumes; --pause-file stops new events while the file exists):
target/release/learned_impact_outcome_generate --job p1-free-t1 --family free_wall --tier 1 --n 500 --seed 1 --jobs 12 \
    --pause-file learned-impact/runs/PAUSE
# one event, and a replay check:
target/release/learned_impact_outcome_record --m5 m5_free_wall --out /tmp/ev
target/release/learned_impact_outcome_record --replay /tmp/ev/m5_free_wall
# the audit report:
cd learned-impact/python && .venv/bin/python -m learned_impact.outcome_audit ../runs/p1-free-t1 --out report.md --json report.json
```

`runs/<job>/progress.log` gets a line per finished event and a 30 s heartbeat;
`runs/<job>/status.json` is rewritten every 30 s (older than 5 min: hung). Failed events
are logged with recipe and seed in `failures.jsonl`; generation continues and fails at the
end if failures exceed `--max-failure-fraction` (default 5%).

## The oracle configuration: one explicit parameter

`--oracle NAME` (`oracle_config.rs`, default `a12_realtime`: P0's provisional pin,
`stress_rubble::scenes::realtime`). Every field it sets, every `Methods` switch included, is
written to `context.json` as `oracle_config` and re-applied from there on replay; the scene
JSON alone is not enough, because it omits methods equal to the built-in default, which
`STRESS_METHODS` can change. Regenerating under a new pin: add the configuration to
`OracleConfig::named` and rerun the same recipes into a new job.

## The stopping rule (the whole life cycle)

A recording stops when **all** hold (`StopRule`, written to `context.json`):

1. every impactor has stopped (< 0.05 m/s, absolutely or relative to the remnant, the largest
   piece: without gravity a pushed free wall never comes to absolute rest) or left the scene's bounds (the structures'
   initial box plus a margin of max(0.3 m, 25% of its largest extent), and not moving back
   in; under gravity only the horizontal extent counts);
2. no bond has broken for 30 frames (0.5 s);
3. the kinetic energy of the bodies that have not left is below 1% of the input energy, or
   every bonded cluster there sleeps (quasi-static) and every loose chunk is still.

The cap is 10 s simulated; an event that hits it is `truncated: true` and must be excluded
from outcome metrics. Anything nonfinite, a change of the chunk or bond set (the fixed-graph
recorder rejects refinement), or kinetic energy above 10x (input + gravity's largest
possible work) fails the event loudly.

## Files per event

| File | Content |
|---|---|
| `scene.json` | the full scene as run (oracle configuration applied) |
| `context.json` | schema, recorder revision, `oracle_config`, QS baseline and punch model, stop rule, family, tier, `split_group`, `layout_group`, generator parameters, scene hash, stable chunk keys `[structure, chunk]` and bond keys `[structure, source bond, a, b]`, column names of the packed arrays |
| `replay.json` | the recipe, frame count, input energy and end fingerprint (`--replay` checks it) |
| `frames.json` | per frame: time, phase, breaks, substeps, routed chunks and impactors, predicted contacts, punch bonds, QS-prior convergence, impactor contact, struck structures, the energy ledger (kinetic split, mechanical, dissipated and its parts, contact terms, elastic) |
| `outcome.json` | the whole run: stop reason, `truncated`, broken bonds, fragments, largest fraction, last break time, breaks in the last 0.5 s, input energy, each impactor's exit velocity and regime (perforate / arrest / rebound / deflect, geometric), frames per phase, wall-clock cost by part, oracle substeps, fingerprint |
| `catalog.json`, `arrays/*` | little-endian raw arrays (`catalog.rs`; dtypes `f32`, `i32`, `f64`), below |
| `snapshots/<boundary>.msgpack.zst` | recorder revision 8 on: exact restart points, the whole `World` (`World::save_state`, rmp-serde with named fields, one zstd frame) at boundary 0, every boundary whose frame routes an island, and every 10th; `World::restore_state(scene.json, bytes)` resumes the oracle bit for bit (contact cohorts, clock and energy ledger included). Listed in `context.json` `snapshots.boundaries` |

## Arrays

`F` frames, `N` chunks and `E` bonds over all structures (flat, structure order), `I`
impactors, `C` predicted contacts in all frames, `P` frames with a punch, `Q` near pairs in
all frames.

P2's per-call `learned-impact-outcome/1` records (its `OUTCOME_SCHEMA.md`) are derived from
this record by a Rust call builder; nothing here is a model feature yet, everything is raw
and in SI units.

**Static** (one row): `static_chunk_mass [1,N]`, `static_chunk_inertia_diag [1,N,3]`,
`static_chunk_half_extents`, `static_chunk_center_local`, `static_chunk_support`,
`static_chunk_structure`, `static_bond_endpoints [1,E,2]` (flat chunk indices),
`static_bond_geometry [1,E,13]`, `static_bond_stiffness [1,E,5]`,
`static_bond_strength [1,E,8]` (an infinite limit is -1), `static_bond_weibull`.

**State at each frame boundary** (`F + 1` rows; row `f` is frame `f`'s start, row `f + 1`
its end): `state_chunk_pos`, `state_chunk_vel`, `state_chunk_angvel` `[.,N,3]` (world,
hidden motion included), `state_chunk_cluster_quat [.,N,4]`, `state_chunk_q_world [.,N,4]` (the chunk's own world
orientation: cluster x hidden rotation x rest rotation, wxyz), `state_chunk_hidden [.,N,6]`
(hidden u, theta), `state_chunk_cluster`, `state_chunk_active`, `state_chunk_anchored`,
`state_bond_joint [.,E,4]` (damage, crush, fatigue, utilization), `state_bond_energy
[.,E,3]` (cumulative J: joint dissipation (fracture, friction, rebar), dashpot dissipation,
stored elastic energy; frame differences are the per-bond labels), `state_bond_mode [.,E]`
(0 none, 1 tension, 2 shear, 3 compression, 4 buckling), `state_bond_fail_margin [.,E]`
(recorder revision 6 on: the joint law's failure margin, a running maximum of
λ/κ_fail(r) per family; >= 1 exactly when the bond disconnected, checked on every row, and
never rounded up to 1 for a connected bond; the per-frame label is row f + 1),
`state_bond_connected`,
`state_impactor [.,I,13]` (position, orientation wxyz, velocity, angular velocity).

**Causal inputs of frame `f`** (known before the step):
- `frame_chunk_routed [F,N]`, `frame_impactor_routed [F,I]`: the routing's predicted impact
  islands (`World::predict_island_members`, the engine's own `select_islands`);
- `contact_i32 [C,5]` (frame, impactor, structure, entry chunk, cluster index) and
  `contact_f32 [C,17]`: every impactor contact the routing sweep predicts
  (`World::predict_contacts`): time within the frame, point, direction, closing speed,
  reduced mass, modulus, face radius, Hertz time, Hertz peak force, the punch's
  force/capacity ratio, plug mass, whether the punch breaks it, cluster size;
- `pair_i32 [Q,3]` (frame, chunk a, chunk b; a < b): chunk pairs of different clusters
  whose bounding spheres overlap at the frame start, the candidates for structure-contact
  and cracked-face edges;
- `frame_chunk_punch_plug [F,N]` (contact index + 1 of the plug holding the chunk, 0: none),
  `frame_bond_punch [F,E]` (the bonds the punch breaks);
- `frame_bond_qs_prior_util [F,E]`: **the QS prior**, a converged static solve of the
  oracle's own bond model under each predicted contact's Hertz peak force on its struck
  chunk (on a copy; one damage commit, no cascade): each connected bond's failure index;
  -1 for bonds disconnected at the frame start and for frames with no predicted contact.

**The oracle's outcome of frame `f`**: `frame_bond_break_bin [F,E]` (-1, or the bin 0..7 of
the frame's eight in which the bond broke), `frame_bond_peak_util [F,E]` (each bond's peak
failure index over the frame), `frame_bond_dissipated [F,E,2]` (J dissipated over the frame
by the bond's joint law (fracture, friction, rebar) and by its dashpots; recorder revision
5 on, earlier records difference `state_bond_energy`), `frame_impactor_contact [F,I,4]` (share of substeps with a
contact impulse on the impactor, first and last contact bin, bit mask of the structures it
touched), `frame_phase [F]` (index into `pre_contact, contact, secondary, separation,
settling`), and the frame-end state (row `f + 1`).

**Baselines on frame `f`'s start state** (the state folded to rigid bodies, momentum
exact, then one frame):
- `qs_*` (every frame): `blast_parity`'s model and switches (`Profile::BlastParity` applied
  through `World::adopt_model`: quasi-static solve, its budget, its features off, its bond
  model): `qs_bond_broken [F,E]`, `qs_bond_peak_util`, `qs_chunk_pos/vel/angvel [F,N,3]`,
  `qs_impactor [F,I,13]`;
- `punch_*` (`P` rows, frames listed in `punch_frame [P]`): the same with the punch's bonds
  broken first, as `STRESS_IMPACT_PUNCH` does before the engine steps. On other frames the
  punch's outcome is the QS one.

**Exact extras** (recorder revision 8 on; the dataset is the ground truth and only grows,
user decision 2026-10-22, so every frame is kept: `storage.select_training_frames` is off):
- `<name>_f64` beside `state_chunk_pos/vel/angvel`, `state_chunk_cluster_quat`,
  `state_chunk_q_world`, `state_chunk_hidden`, `state_bond_joint`, `state_bond_energy`,
  `state_bond_fail_margin` (unrounded) and `state_impactor`: the exact f64 values (the f32
  array is their rounding);
- `break_i32 [B,2]` (frame, bond) and `break_f64 [B]`: each break's exact solver time (s);
- `frame_impactor_impulse [F,I,3]` (f64, N s): the contact impulse on each impactor over
  the frame, its momentum change less gravity's, summed per substep;
  `frame_impactor_peak_force [F,I]` (f64, N): its largest substep-mean contact force.

These are CPU mirrors of the baselines' *stress models* under the oracle's own contact; the
GPU baselines additionally run rubble's rigid engine for contact (4 iterations, 1 step a
frame). See the findings note for how the two compare.

## Splits and ensembles

Every recipe has a `split` (train / val / test by a seeded hash of its layout group; the
`m5_reference` family is sealed) in `context.json`. Every val, test and sealed
configuration, and 15% of the training ones, is recorded as an ensemble of 8 members:
events `<id>_e<k>` sharing the parent's `split_group`, with `context.ensemble =
{id, member, size, perturbation: {impactors: [{offset_m, speed_scale}], weibull_seed}}`.
A member moves each impactor by up to ±1 mm per axis across its approach and scales its
speed by up to ±0.1% (P0's `kin` perturbation) and has its own Weibull seed (which matters
only for materials with a Weibull modulus). Members replace the unperturbed recipe.

## Energy labels

Under the pinned penalty contact the booked contact dissipation is the dashpot quadrature,
not its work, so the ledger does not close (+0.3% to +4.7% of E0, P0). Labels are kinetic
energies (exact, from the state) and the per-bond energies above; `contact_dissipated` in
`frames.json` is a diagnostic only (`context.energy_notes` says so).

## Phases

- `pre_contact`: no impactor contact and no break yet in the event;
- `contact`: an impactor touches the structure(s) of its first engagement;
- `secondary`: an impactor touches a structure after its first engagement ended or a new
  structure (the ram reaching the next wall), or bonds break with no impactor contact
  (debris strikes, cascades);
- `separation`: no contact or break, within 6 frames (0.1 s) of the last one;
- `settling`: later.

## Partitions and sampling (`python/learned_impact/outcome_sampling.py`)

Each recipe is its own `split_group` (descendants keep their parent's); `layout_group` is
the wall layout, held out whole. `partition` assigns layout groups to train / val / test
by a seeded hash and sends sealed families (`m5_reference`) to the sealed set.
`sample_frames` samples by phase quota (35% contact, 20% secondary, 25% separation and
settling, 20% quiet) and keeps every transition frame (a break or a phase change),
raising rather than dropping one over budget.
