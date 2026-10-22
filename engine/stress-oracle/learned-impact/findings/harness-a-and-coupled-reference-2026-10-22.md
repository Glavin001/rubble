# Harness A (the island step through a provider) and the parked coupled reference (2026-10-22)

The planning session and the user set the goal: prove viability against the standalone CPU
oracle, using its own rigid engine and its stress solve. The model is one function, the
oracle's island step: the island's full state in, and its full state out after the
frame's substeps.

Harness A is the primary acceptance harness. Harness B (rubble plus provider) is deferred,
but its first results are recorded below.

## Harness A: built, and bit-identical to the standalone oracle

`World::step_frame_island_provider` (`stress-ref/src/world_island.rs`) handles each frame
the routing selects:
- **Routing:** `predict_island_members`, the standalone oracle's own routing
  (`debris_reach_routing` on in the closed loop).
- **Provider:** gets the island's members and the frame-start world, and returns an
  `IslandState`, the island's full state at the frame's end.
- **Host:** steps the frame natively and imposes that state on the island.

`IslandState` holds:
- every per-chunk, per-bond and per-cluster field of the island, verbatim:
  - per chunk: `ChunkState`, `chunk_bonds`, the `f_int` and `turns` scratch, loads,
    `contact_hits`, contact dt, `exposure`;
  - per bond: `RtBond` with every `JointState` field, and the multirate entries;
  - per cluster: the `Cluster` records with their places in the cluster list, and the
    step-bound cache entries;
- `next_cluster_id`;
- the island's impactors;
- every contact-history entry involving its bodies (`sticks`, `pair_offsets`,
  `ground_offsets`).

The merge destructures `World` and `ReferenceSolver` exhaustively, with no `..`. So every
field has a written policy: the island's comes from the provider; the rest's, and the
globals (time, ledgers, event logs, counters, structure-level caches), come from the host.
A new field cannot be added without a policy, because the build fails.

**Why the host steps the whole world.** In the standalone oracle, the island and the rest
share the frame's substep schedule: the world's stable step, which the island usually
sets. So no host that leaves the island out can reproduce the rest bit for bit. The host
steps everything and discards its own island. For a model provider, the rest can see the
oracle's island only in two ways:
- through within-frame contact, which the routing rules out: the swept bounds are apart;
- through host-side caches; the merge refreshes `topology_version`, the `unchanged` set and
  `true_steps` whenever the provider's topology differs from the host's.

Ledgers, counters and event logs stay the host's. They are diagnostics.

**Tests** (`stress-rubble/tests/harness_a.rs`, no rubble involved):
1. **The oracle island step as provider is the standalone oracle, bit for bit, every frame
   of whole events** (fingerprint plus cluster counts):
   - wall_row: 2 walls, 40 m/s, 1 s, 60 frames, 36 of them routed, 8,648 chunk-frames in
     islands;
   - b5_wall_impact_v20: 1 s, 60 frames, all 60 routed, 46,497 chunk-frames.
2. **A different island changes the island alone.** The provider is the oracle's step from
   a start with the impactors 1% faster. Every chunk outside the island is bit-identical
   to the native frame, and every chunk is in its cluster. 440 and 1,243 island
   chunk-frames took the provider's outcome.
3. **Through the closed-loop harness** (`learned_p0 event --host a --provider oracle`) on
   P1's `p1_anchored_wall_t1_00012_s1002_e0`: the same state hash as the standalone oracle
   (74 fragments, 24.0003 m/s).

The /3 decoded boundary in harness A (`--provider oracle_decoded`) gives 84 fragments and
23.954 m/s on that event, its known loss.

Cost: about 2× the standalone oracle, since the host and the provider both step the frame
(45 s against 24 s on that event).

**Runner:** `closed_loop_eval` candidates:
- `a_ceiling`: the sanity column, identical to the oracle;
- `a_ceiling_decoded` and `a_ceiling_def`: the /3 boundary's own ceiling, reduced and with
  deformation;
- `a_model` and `a_model_def`: a /3 checkpoint, whose decoded state is applied to the
  island on a copy of the frame-start world, with the island's other state carried as it
  was.

Full-contract models will return `IslandState` directly, in P1's full-state format once
it's fixed. First run: `runs/cl-eval-a`, on the 16 bulk validation ensembles with P2's
bulk-w32 checkpoint.

## Harness A ceilings on the bulk validation ensembles

`runs/cl-eval-a-ceilings/report.md`: 16 ensembles, 128 members, gates-v1 / v1.1.

| Candidate | anchored v1 / v1.1 | free v1 / v1.1 |
|---|---|---|
| `a_ceiling`: the oracle's island step through the provider | yes / yes | yes / yes |
| `a_ceiling_decoded`: a perfect /3 model, the oracle's outcome through the decoded state | no / no (all 7 metrics) | no / no (ram, motion) |
| `a_ceiling_def`: /3 plus the carried deformation | no / no (motion_v_0.1 91% vs 95%, motion_v_0.5 94% vs 97%) | yes / no (ram only, at the bar) |

- **Even a perfect /3 model fails harness A.** The carried deformation closes most of the
  free-wall gap but not the anchored one.
- **So the full-state contract (/5) is necessary.**

## Harness A's per-frame peak utilisation (fixed, a2681fa2)

- **The bug:** harness A never reset the bonds' per-frame peak utilisation. A model call's
  `log1p_peak_util_prev` then read a running maximum over the event, not the previous frame's
  peak as recorded.
- **Scope:** physics never reads it, so the oracle and every `a_ceiling*` column are
  unaffected. The `a_model*` runs in `runs/cl-eval-a` were biased, and that run was stopped at
  475/640.
- **The fix:** `step_frame_island_provider` and `oracle_island_step` reset it after the
  provider reads the frame start, which is the recorder's order. `harness_a.rs` checks it
  equals the recorder-order native run.
- **What islands read now:** an island under a decoded state reads 0 at its next call, because
  the decoded state carries no peak. This matches the coupled closed loop.
- **Rerun:** `runs/cl-eval-a2`, with a pinned revision-5 build (`bin/learned_p0.cl14-rev5`):
  /3 checkpoints need /3 inputs, and revision 6 changed /3-era features.

## DAgger relabel (P2)

`stress_learned::relabel` and `learned_impact_relabel` take a job (event, frame, and one of
the island's own state, an exact `IslandState`, or a model's decoded state).

**What it returns:**
- the oracle's frame-end `IslandState`;
- the call (`learned-impact-outcome/4`): inputs from that start, targets from the oracle's
  frame, captured by the recorder's own code (`outcome::oracle_frame`, now shared);
- `dagger` provenance in the manifest.

**Tests and checks:**
- **Round trip:** a relabel of the oracle's own state is bit-identical.
- **Against the recorder:** an own-state relabel is the recorder's call bit for bit, inputs and
  targets. This holds on a fresh recording (test) and on P1's calls-v4-eval. The exceptions:
  - `y_impactor_regime` is -1: the event-level regime, unknown to one frame;
  - there is no punch baseline.
- P2's validator accepts the records.

**Cost:** about 1 s per relabel for 40–150-chunk islands, plus one replay of the parent event,
which an event's jobs share.

## Comparison videos

`runs/videos-2026-10-22/` (`learned_impact.compare_videos`; stress-viz engines `a_learned` and
`qs`):
- **What each shows:** oracle | the model in harness A | QS, with a broken-bonds chart, for the
  M5 free wall, the M5 wall row, an anchored wall under gravity, a corner hit, four rams, a
  ballistic-limit pair, and a bulk validation event.
- **Check:** each oracle panel reproduces P1's record (pieces and exit speeds, 8/8).
- **First set:** P2's bulk-w32 (/3), read in that folder's `index.md`.

## Parked: the coupled reference (rubble plus the oracle's island step), for later

- **Interface:** `World::step_frame_with_frame_provider` puts the coupled frame's island
  step (`coupled_substeps`) behind a provider. With the oracle it equals `step_frame`:
  - whole runs, untiered: bit for bit;
  - per frame, both routings: from one prepared world, bit for bit
    (`tests/coupled_reference_ceiling.rs`).
- **Production rubble policy:** `STRESS_P0_ENGINE_POLICY=production`, which is 4 iterations
  and one step per frame. Every coupled event records its engine context: policy, routing,
  adapter and OS build.
- **Rubble determinism, an item for rubble's owner:**
  - Untiered: GPU rubble is bit-reproducible run to run on one machine.
  - Tiered: it is **not**. Rubble integrates the structure, and its GPU contact solve isn't
    order-stable. Three runs of an anchored event gave three state hashes, with the same
    fragments and exit speed to 5 digits.
  - So a tiered coupled reference is reproducible only statistically. The fix is
    deterministic contact accumulation in engine-webgpu.
- **Chaos floor:** 0.15 s impact windows, production policy, 16 kin members per scene.
  - Untiered equals the standalone oracle's statistics on the six free wall scenes.
  - Tiered is within ±0.03 m/s and ±2 fragments.
  - b5_v20, the scene with gravity and ground, shifts slightly under both routings.
  - Cost per event: 1–2× standalone. Data: `runs/p0-coupled-chaos/summary.json`.
- **External goldens through rubble** (`stress_ref_coupled check scenes golden`, the
  oracle CLI with a rubble-coupled factory):
  - Untiered: 4 failures, all b5 against OpenCourant (speed_lost, failure_mode,
    hole_area). This looks like the documented gap the standalone oracle shares (README),
    not yet checked row by row.
  - Tiered: 12 failures. Besides the b5 rows, it misses the b6 spall (no spall at all) and
    b7 masonry's ball speed against three oracles.
  - So under rubble, untiered routing matches the external references and tiered does not.
  - Logs: `runs/coupled-golden-tiered{0,1}/progress.log`.
- **The routing flag:** `STRESS_ROUTING_TIERED=1` (`World::routing_tiered`) applies the
  deployed tiered rule under any solve mode. P3's GPU routing is an exact port, 540/540
  frames agree (P3's `learned_routing.rs`).
