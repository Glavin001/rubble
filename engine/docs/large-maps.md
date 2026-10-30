# Large maps: a 5 km² destructible city

Status: **proposed** (2026-10-30). Nothing here is built yet.

## Goal

A map of about 5 km² (≈ 2.2 km × 2.2 km), filled with many varied destructible
buildings, where:

1. **An idle map costs almost nothing.** Physics cost does not grow with how many
   buildings exist.
2. **Cost tracks destruction.** It grows with what is breaking, moving or has
   been broken nearby, not with the size of the map.
3. **Physics quality is unchanged.** The rules in `AGENTS.md` still hold: no lossy
   shortcuts by default, explicit limits that fail loudly, and forward-only
   fracture.

### What "5 km² of city" means in numbers

The Vibe Town pack (`engine/scenes/town`, branch `feat/unreal-client`) covers about
296 m × 150 m (0.044 km²). It holds 70 buildings, 51,921 chunks and 132,529 bonds.
At the same density, 5 km² is about 113 times that:

| | Vibe Town | 5 km² at town density |
|---|---|---|
| Buildings | 70 | ~8,000 |
| Chunks | 52k | ~5.9M |
| Bonds | 133k | ~15M |

None of today's paths can hold that resident. The `scale()` preset tops out at
524k bodies and 1M colliders, and threshold placement reserves one body slot per
chunk. Even if it fit, every per-collider and per-chunk pass would run over all of
it on every step.

## Where cost comes from today

All of these were surveyed on `origin/main` and `feat/unreal-client`.

**Per-step passes over every body or collider** (`engine-exec/src/rigid_plan.rs`,
used by the World API, native, Metal and CUDA):

- `build_aabbs` rebuilds statics every step.
- `broadphase_clear`, `broadphase_insert` and `broadphase_query` run over all
  colliders.
- `build_awake_list`, the CSR clear and prefix, the colour-tile passes,
  `predict_bodies`, `extract_velocity_sleep`, `update_refs` and `pose_pack` run
  over all bodies.
- The fracture lab's plan (`engine-fracture/src/plan.rs`) already has awake-only
  and moved-only versions of most of these. They only switch on when fewer than
  a quarter of bodies are awake, and `rigid_plan` has none of them.

**Resting work:**

- Rest-carry re-emits every resting manifold each step.
- `solve_dual` touches every contact and every joint on every iteration.

**Threshold destruction** (`engine-world/src/threshold.rs`):

- An intact building is a static or dynamic cluster body with **one collider per
  chunk**.
- `fr_loads` runs over every chunk and `fr_breaks` over every bond, every step.
  The seven strike kernels also run whenever any projectile is registered.
- A split dispatches over the whole destructible. The town is a single
  destructible, so one chip launches threads for 52k chunks.
- The readback copies about 2.5 MB per step for the town, and `apply_fracture`
  scans every bond and chunk on the host.
- The suggested threshold solver sets `sleep_lin = -1`, so clusters never sleep
  and the all-asleep skip never fires.

**Measured idle cost grows with building count:**

- town16, town64 and town256 idle at 2.9, 5.7 and 9.6 ms
  (`stress-fracture-benchmark.md`).
- 155k all-asleep bodies cost 2.6 ms GPU.

**Breaks at 2+ km:**

- No world partition or streaming.
- f32 positions with no `shift_origin`. At ±1.1 km one ulp is about 0.12 mm,
  which comes within 2× of the lab's sleep threshold at 120 Hz.
- One heightfield is capped at 4,096 samples per side.
- Large-static caps: 1,024 on the uniform grid, 10 oversized on the hierarchical
  grid.
- Renderers do no culling and no LOD.

## Design

**The core idea:** a building nobody is near does not exist in the physics world.
It exists only as an *instance record*: an archetype id, a transform and a state.
It becomes physical before anything can touch it. Everything that is physical
costs in proportion to what is awake.

### 1. Three tiers per structure

```
dormant ──(predicted contact)──▶ resident-intact ──(first break)──▶ resident-damaged
   ▲                                   │
   └──(nothing near; state unchanged)──┘
```

- **Dormant.** An instance record lives in a host-side spatial index. It has no
  bodies, colliders, fracture tables or GPU memory.
  - Cost is about 64 B of host memory per building and zero per step.
  - 8,000 buildings come to 512 KB.
- **Resident-intact.** The building is materialised exactly as threshold
  placement does it today: anchored clusters are static, loose clusters are
  dynamic, and every chunk has its collider. Everything starts asleep. Its
  fracture tables are loaded.
- **Resident-damaged.** It has broken at least once. It stays resident, so cost
  grows with how much has been destroyed, which is the goal.
- **Back to dormant.** A resident-intact building with no awake neighbours goes
  back to dormant. This is exact: nothing about it has changed, so its state is
  still the archetype's.

**Why dormant is lossless.** A dormant building behaves exactly like a resident
building whose bodies are all asleep.

- A sleeping body changes only through a contact, a joint, a host command or a
  ray or projectile.
- Activation happens strictly before any of those can reach it (§2).
- To make that hold, every archetype is **settled and verified at bake time**.
  It has to reach sleep with no breaks under gravity, or the bake fails with an
  error naming the structure.
- Its settled poses are what get materialised.

### 2. Predictive activation (forward-only)

At the end of frame *N*, after host commands for frame *N+1* are known, the host
tests every awake body's swept bounds over the next frame against the dormant
index. That frame includes gravity and any queued impulse or explosion radius.

- Any hit is materialised before frame *N+1* steps. This is the same check as the
  stress charter's routing check (M5), applied one tier up.
- Rays, projectiles and explosions query the dormant index the same way.
  `body_set_projectile`, raycasts and `explode` activate what they would reach.

**Look-ahead.** A second, wider test over a horizon (default 0.5 s) pre-stages
uploads asynchronously so that materialising rarely costs a frame spike. The
one-frame test is the correctness gate, and it materialises synchronously if
staging has not finished.

**Budgets are explicit.** `LargeWorldDef { max_resident_structures,
max_resident_chunks, … }`. Exceeding one fails with a named error. It never
silently skips activation.

### 3. Archetypes and instancing

- Buildings come from a library of **archetypes**: chunk shapes, bonds, materials
  and settled poses, baked once.
- Instances share an archetype's GPU-ready tables, such as the hull arena and
  fracture bond tables. Materialising copies only per-instance state.
- **Variety comes from two sources:**
  - many archetypes, generated (§6) and imported;
  - per-instance render palettes, which change colours and textures only.
- A material swap that changes strength produces a new archetype, never a
  per-instance tweak to bonds.
- Splitting the town pack into one destructible per building is part of this. It
  also stops a single chip from dispatching over all 52k chunks.

### 4. Make resident cost follow awake work (lossless)

Materialised structures are asleep. They must cost about nothing while asleep.
Each item below gets an A/B test against the old kernels on the same state
(see the memory note on native wgpu reproducibility).

- **a. Port the fracture plan's awake-list and moved-only passes into
  `rigid_plan`.** This covers awake-list predict, extract and commit, moved-only
  bounds, the persistent resting broadphase table and awake-collider
  insert/query. Statics stop being rebuilt every step.
- **b. Compact the remaining per-body passes** (CSR clear and prefix, colour
  tiles, `build_awake_list`) to the awake set. This is D7 in
  `production-plan.md` and `ENGINE_AUDIT` item 197.
- **c. Skip rest-carry and the dual solve for pairs where both bodies sleep.**
  The sleep stasis already makes their update a no-op; this skips the dispatch
  as well.
- **d. Threshold fracture over awake clusters only.**
  - `fr_loads` and `fr_breaks` skip chunks and bonds whose clusters are asleep.
    Frozen contacts mean their loads are unchanged and already below capacity.
  - Strike kernels run only over clusters inside a registered projectile's
    swept bounds.
  - Readback and `apply_fracture` touch dirty clusters only.
- **e. Threshold sleep.** Clusters are allowed to sleep, with the same thresholds
  the rest of the engine uses.
  - This changes the threshold solver's suggested policy from its current
    "never asleep".
  - It is validated by break-set equality on the town bombard and the stress
    oracles with sleep on and off.
  - If it does not match, it stays a flag and goes to you for a decision.

### 5. Ground, precision and big statics

- **Tiled heightfields.** Terrain is split into tiles, for example 256 m with
  4,096 or fewer samples each. Each tile sits in the hierarchical grid at its own
  level instead of the oversized list. Terrain is always resident and static, so
  with 4a it is free when nothing touches it.
- **`World::shift_origin`.** This is already planned in `public-api.md`. It
  re-centres the world around the activity focus. Maps are authored centred
  (±1.1 km), and shifting keeps active play near the origin.
  - Precision gets strictly better with a shift; no input is degraded.
  - For a server with activity spread across the whole map, ±1.1 km is the
    documented worst case.
- **Graveyard and sink.** The y = −10 km graveyard and the 10 km patch-sink
  sentinels become configurable, so deep maps don't collide with them.

### 6. Content: varied buildings at city scale

**A procedural structure generator** generalises `cuboid.rs` and `glass_tower.rs`
into one parametric description:

- Footprint polygon, storeys, storey height.
- **Structural system:**
  - load-bearing masonry, using the running-bond bricks from `cuboid.rs`;
  - reinforced-concrete frame with slabs;
  - steel frame with curtain wall and Voronoi panes, from `glass_tower.rs`;
  - timber frame.
- **Envelope:** openings with lintels, roof type (gable, hip, flat, parapet),
  balconies, chimneys.
- **Materials:** drawn from the town's 121-material library.
- **Output:** a chunk/bond archetype, settled and verified at bake.

**Imported archetypes:**

- the town's 70 buildings and its props, split per building;
- the vibe-city presets;
- Shardwright GLBs for hero buildings. Shardwright re-export stays the fix for
  bad source data, as charter item R4 requires.

**A city layout generator** builds the map from a seed:

- a road network with districts (downtown towers, mid-rise, residential,
  industrial, suburbs);
- lots filled from district archetype pools;
- props, and terrain tiles.

Output is a map file: terrain tiles plus instance records. Loading the map loads
records, not bodies.

### 7. Rendering

Physics no longer touches intact buildings, so the renderers must not either.

- **Web.**
  - Intact buildings are drawn as one instanced mesh per archetype, with frustum
    and distance culling.
  - Only resident-damaged structures draw per chunk.
  - Far-away buildings use impostors.
- **Unreal.**
  - Per-archetype Nanite meshes with HLOD.
  - Chunk instances are only created on activation (`unreal/PLAN.md:186`, "a
    city of destructibles needs chunk meshes").

## Gates and benchmark

**A new `city_scale` benchmark** runs native and in the web lab. It sweeps the
building count K ∈ {70, 700, 8,000} on the same 2.2 km map shape, across four
scenarios:

| Scenario | What happens | Target |
|---|---|---|
| S0 still | Nothing moves | Physics GPU about 0 ms (the step is skipped) |
| S1 drive | One vehicle drives a 2 km route through streets | Per-step ms flat in K, within noise; ≤ 0.5 ms physics on an M3 Max |
| S2 demolish | One tower is levelled | Cost set by that tower, flat in K |
| S3 war | Strikes spread across 20 buildings | Cost grows with the 20, not with K |

**Lossless gates.**

- With the dormant tier on vs everything resident (small K, where both fit):
  identical break sets and matching trajectories for the same events.
- For 4a–4e: A/B against the old kernels on the same state.
- The stress-oracle bombard still meets the charter targets.

## Milestones

| | What | Depends on |
|---|---|---|
| L0 | `city_scale` bench plus today's baseline. Expect linear-in-K numbers. | — |
| L1 | Per-building destructibles: split the town. Threshold fracture over awake clusters only (4d). Threshold sleep (4e). | L0 |
| L2 | Lossless resident cost: 4a, 4b, 4c. Dirty-only readback. | L0 |
| L3 | Dormant tier: archetype library, instance index, predictive activation for bodies, rays, projectiles and explosions, resident budgets, return to dormant. | L1, L2 |
| L4 | Structure generator, archetype bake and verify, city layout generator. | L3 (format) |
| L5 | Tiled heightfields and `shift_origin`. | — (parallel) |
| L6 | 5 km² showcase map: web lab with culling and impostors, plus Unreal with HLOD. | L3, L4, L5 |

L1, L2 and L5 are independent and can run in parallel.

## Open decisions (yours)

1. **Returning damaged areas to dormant.** Settled ruins could be snapshotted
   back to dormant (poses, bond bits, cluster topology) to free resident budget.
   - On re-activation they would lose their contact warm-starts, so the first
     steps after waking differ slightly from staying resident.
   - **Default:** damaged structures stay resident (cost grows with destruction
     done). The snapshot is an opt-in flag, measured and reported.
2. **Threshold sleep.** If 4e changes break outcomes on the bombard, it stays a
   flag and you decide.
