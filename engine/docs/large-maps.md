# Large maps: a 5 km² destructible city

Status: **proposed, revision 2** (2026-10-30). Nothing here is built yet.

Revision 2 follows your review. The first draft worked around engine costs with a
separate "dormant" tier that kept far-away buildings out of the world. This
revision fixes those costs at their source instead:

- sleeping and static bodies cost nothing per step;
- readbacks carry only what changed;
- fracture patterns are shared prototypes, not stored per wall.

## Goal

A map of about 5 km² (≈ 2.2 km × 2.2 km), filled with many varied destructible
buildings, where:

1. **Idle costs nothing.** Physics cost per step is set by awake bodies and
   active destruction only. The number of buildings, chunks or sleeping bodies
   in the map does not change it.
2. **Cost tracks destruction.** It follows what is breaking and moving.
3. **Physics quality is unchanged.** The rules in `AGENTS.md` still hold: no
   lossy shortcuts by default, explicit limits that fail loudly, and
   forward-only fracture.

## Why idle cost grows today

Sleep and the broadphase already remove almost all of the *work* for a sleeping
or far-away body:

- no narrowphase pairs;
- no primal solve;
- the broadphase query returns at once.

What remains is the *launch*. Most passes are dispatched with one GPU thread per
body or collider in the world. Each thread reads a flag and returns, but it still
runs, and the step pays for all of them.

**In `rigid_plan`**, which the World API, native, Metal and CUDA use, these run
every step over all bodies or colliders:

- `build_aabbs` (statics are rebuilt every step);
- `broadphase_clear` over the whole table, then `broadphase_insert` and
  `broadphase_query` over every collider (sleepers and statics are re-inserted
  each step);
- `build_awake_list`, the CSR clear and prefix, `count_color_tiles` and
  `scatter_color_tiles`;
- `predict_bodies`, `extract_velocity_sleep`, `update_refs` and `pose_pack`.

**Rest-carry** re-emits every resting manifold each step. `solve_dual` touches
every contact and joint on every iteration, including those between two
sleepers.

**The fracture lab's plan** (`engine-fracture/src/plan.rs`) already has
awake-list and moved-only versions of most of these passes. They switch on only
when fewer than a quarter of bodies are awake (`plan.rs:861`), and `rigid_plan`
has none of them.

**Measured cost.** About 4 ns per body per step in the lab's best case, and
2.6 ms GPU for 155k bodies all asleep. At city scale that floor alone would be
tens of milliseconds.

**Threshold destruction** (`engine-world/src/threshold.rs`) adds its own costs:

- `fr_loads` runs over every chunk and `fr_breaks` over every bond, every step.
- The strike kernels run over everything whenever any projectile is registered.
- A split dispatches over the whole destructible.
- **Readback** (`Fracture::extras`, `threshold.rs:971`) copies every chunk owner
  (4n bytes), every bond flag (4e bytes) and every body frame (32 bytes per slot)
  on every step. That is about 2.5 MB for the town, whether or not anything
  changed. `apply_fracture` then scans all of it on the host.
- The World pose readback writes only moved bodies, but copies and maps a buffer
  sized to every body.
- The suggested threshold solver sets `sleep_lin = -1`, so clusters never sleep.

The result is the measured idle cost of town16, town64 and town256: 2.9, 5.7 and
9.6 ms. That is a defect to fix, not a property of the design.

## Design

### 1. Sleeping and static bodies cost nothing

**The rule:** every per-step pass is dispatched over an awake set or a changed
set, never over N.

- **a. Awake-indexed dispatch everywhere.**
  - Predict, extract, commit, `update_refs`, `pose_pack`, the colour tiles and
    CSR all use indirect dispatch sized by the awake count. CSR indexes the
    awake list, not body ids.
  - The quarter-awake switch goes away. The awake path is the only path, and it
    is A/B tested against the old kernels on the same state.
- **b. Persistent broadphase for statics and sleepers.**
  - Statics and sleeping bodies live in a resident structure. Each is inserted
    once and changed incrementally when a body sleeps, wakes, is created or is
    removed.
  - Only awake colliders are inserted per step, and only awake colliders query.
  - The fracture plan rebuilds the whole resting table after any sleep event.
    That becomes incremental insert and remove, so settling debris does not
    cost O(map).
- **c. No resting work between two sleepers.**
  - Rest-carry, CSR, packing and the dual solve skip any pair whose bodies are
    both asleep, or asleep and static.
  - Their manifolds stay stored and come back on wake, exactly as rest-carry
    keeps them today. Only the per-step re-emission stops.
- **d. Fully-asleep frames** keep today's skip of the whole step.

### 2. Readback carries changes only

- **Poses.** The GPU writes a compacted list of moved, slept and woken bodies plus
  a count. `pose_pack` already does this. The copy and map become sized by the
  count: the host reads the count, then copies just the records (or uses a
  bounded ring).
- **Fracture.** The GPU appends events to a list with a count:
  - bonds broken;
  - chunk ownership changes;
  - new bodies with their frames.

  The host applies those events and never scans all bonds or chunks.
- **Quiet frames** read back only the counters.

### 3. Shared fracture patterns (prototypes)

The same element of the same shape appears hundreds of times in a city: a 4 m
brick wall with a window, a slab bay, a column, a curtain-wall panel. Store its
fracture once.

- **Element prototype.**
  - An element shape, such as a wall panel, plus a small set of fracture
    patterns for it (for example 10).
  - Each pattern holds the chunk hulls, the internal bonds, the material
    capacities and the chunk→face map.
  - Geometry and bond tables are GPU-resident once per pattern.
- **Element instance.** A transform, a prototype id, a pattern id (chosen by
  seed), and the bonds that join it to its neighbours. That is tens of bytes.
- **Building.** An assembly of element instances plus the bonds between
  elements, such as wall to slab and slab to column.

**While intact**

- **Bodies.** The building is one cluster body: anchored, static, asleep. Today
  the threshold model already makes each bonded cluster one body.
- **Colliders.** Each element instance uses its pattern's chunk hulls *by
  reference*: an instance transform plus pattern id. The hulls are not copied.
  Contacts and loads are still per chunk, exactly as today.
- **State.** The bonds' intact bits are implicit ("all intact") until the first
  break touches that element.

**On fracture**

When bonds break and a piece splits off, it gets its own body. That body is
"racked" from the pattern:

- its chunk shapes are the pattern's, referenced and never copied;
- its pose comes from the instance transform;
- its bond state comes from that point on.

So memory and work scale with *broken* elements, not with all elements.

**Rough sizes at 5 km²**

- About 8,000 buildings at a few hundred elements each gives roughly 2M element
  instances × ~32 B ≈ 64 MB. That is resident and costs nothing while asleep
  (§1).
- Patterns: a few hundred prototypes × 10 patterns × ~100 chunks ≈ 300k unique
  hulls. That is the same order as one town today.
- Variety comes from many prototypes and buildings, seeded pattern choice and
  render palettes. A strength-changing material is a different prototype.

**One open question for you:** whether an intact element collides with its
pattern's chunk hulls (proposed, identical to today) or with its single original
shape. The single shape is the exact union of the chunks, so it would also remove
the seam edges. It does change contact results, so it is not proposed unless you
choose it.

### 4. Threshold destruction tracks destruction

- **Threshold sleep.** Clusters may sleep, with the engine's normal thresholds.
  It is validated by break-set equality on the town bombard and the stress
  oracles with sleep on and off. If it does not match, it stays a flag and goes
  to you for a decision.
- **Loads and breaks.** `fr_loads` and `fr_breaks` run over chunks and bonds of
  awake clusters only. A sleeping cluster's contacts are frozen, so its loads
  are unchanged and already below capacity.
- **Strikes.** The strike kernels run only over clusters inside a registered
  projectile's swept bounds. This is the routing check that already exists.
- **Splits.** A split dispatches over the dirty clusters' chunks, not the whole
  destructible.
- **The town** is split into one destructible per building, so one chip is local
  by construction.

### 5. Map scale

- **Tiled heightfields.** Tiles of 256 m or so, each with at most 4,096 samples
  per side. Each tile sits in the hierarchical grid at its own level. They are
  static, so §1b makes them free.
- **`World::shift_origin`.** Already planned in `public-api.md`. Maps are
  authored centred (±1.1 km), and shifting keeps play near the origin.
- **Configurable sentinels.** The graveyard and terrain-sink offsets (−10 km)
  become configurable.

### 6. Content

- **Structure generator.**
  - It generalises `cuboid.rs` and `glass_tower.rs`. A building is assembled
    from element prototypes: masonry walls, RC frame and slabs, steel frame with
    curtain wall, timber frame, roofs, openings.
  - Materials come from the town's 121-material library.
  - Each prototype pattern is settled and verified at bake time. The bake fails
    with an error naming the element if it does not hold.
- **Imported buildings.** The town's 70 buildings, split per building; the
  vibe-city presets; Shardwright hero buildings.
- **City layout generator.** A road network, districts and lots filled from
  district pools, built from a seed.

### 7. Rendering

Rendering follows the same rule: intact buildings must not cost per chunk.

- Intact elements draw as one instanced mesh per prototype pattern, with culling
  and LOD or impostors at distance.
- Racked pieces draw per body.
- Unreal uses Nanite and HLOD per prototype.

## Gates and benchmark

**A new `city_scale` benchmark** runs native and in the web lab. It sweeps the
building count K ∈ {70, 700, 8,000} on the same 2.2 km map, across four
scenarios:

| Scenario | What happens | Target |
|---|---|---|
| S0 still | Nothing moves | The step is skipped: about 0 ms |
| S1 drive | One vehicle drives 2 km of streets | Flat in K, within noise |
| S2 demolish | One tower is levelled | Cost set by that tower, flat in K |
| S3 war | Strikes spread across 20 buildings | Grows with those 20, flat in K |

**Lossless gates.**

- Each change in §1, §2 and §4 is A/B tested against the old kernels on the same
  state.
- The stress-oracle bombard still meets the charter targets.

## Milestones

| | What |
|---|---|
| L0 | `city_scale` benchmark and today's baseline. This shows the linear-in-K cost. |
| L1 | §1: sleeping and static bodies are free (awake-indexed dispatch, persistent broadphase, no sleeper–sleeper resting work). |
| L2 | §2: readback carries changes only. §4: threshold sleep and awake-only fracture, with the town split per building. |
| L3 | §3: element prototypes, shared patterns and racking on fracture. |
| L4 | §6: structure and city generators. |
| L5 | §5: tiled heightfields and `shift_origin`. |
| L6 | 5 km² showcase map (§7). |

L1, L2 and L5 are independent.

## Not needed now

A separate dormant or streaming tier, which kept buildings out of the world
entirely, was in revision 1. With §1–§3 a sleeping building costs memory only, at
about 64 MB for the whole map. Streaming comes back only if a map's resident
memory exceeds a budget, which would be an explicit limit with an error.
