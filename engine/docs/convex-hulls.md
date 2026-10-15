# Convex hulls in the core Slang AVBD library

Convex hulls (up to 64 vertices, 128 faces, 192 edges) and compound bodies made
of many hulls are first-class collider shapes on every host: WebGPU (Rust
`WgpuSim` and the browser host in `crates/engine-webgpu/fracture/simulation.mjs`),
Metal (`MetalWorld`), and CUDA (compiled only). The opt-in browser-only
`fracture_step.slang` path is retired.

Gauss-map pruning and incident-face selection are adapted from three-avbd
(MIT, Steven Bobyn; see `slang/THIRD_PARTY_LICENSES.txt`).

## Hull data: packed ABI v2

One `uint4` arena (Slang `gHulls`, binding 18). Floats are stored as raw bits, so
indices never pass through f32 and offsets are exact. The layout is documented in
`crates/engine-schema/src/convex.rs`. In short, each hull at record base `H` has:

- **Header:** counts and section offsets.
- **Local bounds:** AABB centre and half extents, bounding radius, inner radius,
  and an interior point.
- **Vertices:** each with its incident-edge adjacency, used for hill-climbing
  supports.
- **Face planes:** outward normal and offset.
- **Face loops:** `u8` vertex indices, counter-clockwise about the outward normal.
- **Edges:** `a | b<<8 | faceL<<16 | faceR<<24`.

`HULL_FLAG_CLOSED` certifies reverse-edge adjacency, which enables Gauss-map
pruning and hill climbing. Uncertified face soups fall back to exhaustive
projections. They are never pruned using unproven adjacency.

Collider fields:

- `params = (asfloat(base | 0x40000000), uniform scale, 0, bound radius)`. The tag
  keeps the base a normal float, never a subnormal.
- **Boxes** reuse the unit cube that every arena holds at record 0, with
  `params.xyz` as a per-axis scale. Box/hull pairs therefore run through the same
  SAT. Box/box keeps its specialised path.

Cookers that produce identical bytes:

- **Rust:** `ConvexHull::from_points` (incremental hull, reduced to 64 vertices
  by farthest-point sampling) and `ConvexHull::from_faces` (Shardwright face soups:
  weld, merge coplanar faces, certify adjacency, uncertified fallback).
- **Bodies:** `cook_body` builds a body from one hull or a compound. It computes
  mass properties and principal axes, and recentres each child on its own bounds
  so `local_pos` carries the offset.
- **Arena:** `HullArena` deduplicates hulls.
- **JS:** `fracture/geometry.mjs`, with the same algorithm.

`fracture/hull-golden.json` pins the cross-language bytes. Regenerate it with:

```sh
UPDATE_HULL_GOLDEN=1 cargo test -p engine-schema golden
```

## Pipeline

| Stage | Kernel | Notes |
|---|---|---|
| Bounds | `build_aabbs` | Exact per-vertex AABB plus local-AABB OBB cache; sleeping bodies' bounds are built on step 0 |
| Broadphase | `broadphase_insert/query` | Hierarchical size-level grid when `HOST_S(18)` = levels > 0 |
| Narrowphase | `narrowphase` | Reuse first; hull pairs deferred via `candidate._pad = CONVEX_DEFER \| hint` |
| Classify | `convex_classify` (slot 64) | Plane, sphere and capsule pairs emit contacts; hull/hull and hull/box pairs are compacted into the small (front) or large (back) SAT bucket |
| Args | `convex_write_args` | Slots 70, 71 (balanced 2-D rows) and 72 |
| SAT | `convex_sat_small` (slot 70) | 16 lanes per pair, 4 pairs per workgroup; V ≤ 16, F ≤ 32, E ≤ 48 |
| SAT | `convex_sat_large` (slot 71) | 64 lanes per pair |
| Manifold | `convex_manifold` (slot 72) | One thread per SAT pair; clip, reduce, emit |

**Broadphase.** The hierarchical grid is selected whenever the scene has convex
colliders; set `ENGINE_GRID_LEVELS=on|off` to force it either way.

- **Insertion.** Each collider is inserted once, at the level whose cell
  (`base·2^level`, where base is the median diameter) covers its bound.
- **Queries.** A collider queries its own level and every coarser level's 27
  cells, skipping empty levels via an occupied-level mask in scratch slot 51.
- **Sleeping and fixed colliders.** Fine sleeping or fixed colliders still query
  coarse movers.
- **Oversized statics** go to a 10-entry list (scratch 53..63), tested by every mover.
- **Rejection.** Exact hash-cell checks stop aliasing. Box and hull pairs pass
  the 6-axis OBB test first. Planes are bounds-tested.

**SAT.** Each pair's lane segment caches A in B's local frame (one transform per
vertex or plane) plus B's data in shared memory.

- **Face axes:** strided across the segment's lanes, then tree-reduced.
- **Edge axes:** A edges are strided across lanes and pruned with the Gauss map.
- **Face bias:** an edge must exceed `0.95·face + 0.01`.
- **Hint from the previous frame:** the previous feature, found by the reuse
  probe, adds hysteresis (`CONVEX_HYSTERESIS`), which keeps features and warm
  starts stable.
- **Fast path (`CONVEX_FAST_FACE`):** if last frame's reference face still wins,
  the O(Ea·Eb) edge query is skipped.
- **Output:** the work item stores the chosen axis and feature plus the best edge
  pair.

**Manifold.** For face contacts:

1. Take the incident face that touches the deepest support vertex, found by hill
   climbing.
2. Clip the incident edges against the reference side planes (Cyrus–Beck).
3. Add reference vertices that fall inside the incident face. They are projected
   along the reference normal, and the projected point is what gets tested.
4. Stream every candidate into a reducer that keeps the deepest point and the
   extremes in 4 (or 8) directions.
5. Pick at most `MAX_CONVEX_CONTACTS` (default 4) points: deepest, farthest, then
   the largest triangle on each side.

If a biased face's patch clips empty, the manifold falls back to the SAT's best
edge pair. On the fast path it runs a scalar Gauss-map edge query instead. A
projected support point is the last resort.

**Feature IDs:**

- Face contacts: `type<<30 | ref<<23 | inc<<16 | kind<<14 | a<<7 | b`.
- Edge contacts: `3<<30 | ea<<8 | eb`.

## Beyond three-avbd: reuse, filters, runtime hulls

**Whole-step manifold reuse** (three-avbd `reuseManifolds`). Call
`WgpuSim::step_reuse_manifolds()`, `MetalWorld::step_reuse_manifolds()` or
`sim.step(n, {reuseManifolds: true})`. The step skips bounds, broadphase,
narrowphase and convex work. `reuse_all_manifolds` turns every previous contact
run into this step's manifold, using one thread per previous contact (slot 73),
and persistence re-evaluates C0 from the anchors.

Hosts track validity the way three-avbd's `manifoldsValid` does. Any change to
bodies, colliders, filters, hulls or commands, an overflow, or a reset forces a
full step instead.

**Collision filtering**, per collider. three-avbd's filters are per body; per
collider means compound children can differ. Set it with
`CollisionFilter { categories, mask, group }.apply(&mut collider)`, or with
`filter: {categories, mask, group}` on a JS collider.
- **Categories and mask:** 16 bits each, stored in `Collider.generation`. 0
  means "collides with everything", so unfiltered scenes are unchanged.
- **Group:** a signed Box2D-style group in the upper 16 bits of
  `Collider.material`. A shared positive group always collides and a shared
  negative group never does, overriding categories.
- **Where it runs:** the test is in `emitCandidate`, so every broadphase path
  filters identically.
- **Cost:** none when unused, because hosts set `HOST_S(23)` only when some
  collider carries a filter.

**Runtime hull management.** `HullArena` provides:
- deduplication with reference counts (`insert`/`release`/`refs`);
- a first-fit free list that merges adjacent blocks, and a free tail shrinks the
  arena;
- dirty record ranges (`take_dirty`), so hosts upload only what changed.

Hosts grow the GPU hull buffer on demand: power-of-two size, with contents copied.
WebGPU rebuilds every bind group that references binding 18. The calls are
`WgpuSim::sync_hulls`, `MetalWorld::insert_hull`/`release_hull`/`set_hulls` and
`sim.syncHulls(arena)`.

`set_collider` / `setCollider` swaps a collider's hull or filter at runtime, for
example to replace a fractured chunk. The JS `HullArena` mirrors the Rust one.

## Bindings, slots and scratch

| Resource | Owner |
|---|---|
| Binding 29 | `gConvexWork` (`maxCandidates` × 32 B). Declared last, so Metal assigns buffer 27; the Metal host rebinds 27 around the convex dispatches. |
| Indirect slots 70..72 | Convex args (buffer sized ≥ 73 slots). |
| `GpuCounters` bytes 44 and 60 | `convexSmall` / `convexLarge` (`StepCounters::convex_sat_small/large`; byte 56 is the two-pass narrowphase deferred count). |
| `HOST_S(18)` | Grid levels. |
| `HOST_S(22)` | Degree-aware primal lanes (needs `gColorCounts` ≥ 128 words). |
| Scratch 51 | Occupied grid levels. |
| Scratch 53..63 | Oversized-static list for the hierarchical grid (count + 10 IDs). |
| `HOST_S(23)` | Collision filters present. |
| Indirect slot 73 | Previous contacts (whole-step manifold reuse). |

This removes the old collisions: binding 27 as fracture hits versus solver
points, and slot 69 as SAT args versus recolour. Every entry stays within
WebGPU's 10 storage buffers per stage; `every_entry_fits_the_webgpu_storage_buffer_budget`
enforces it.

## Solver hookups (no new algorithms)

- **Browser host.** It now uses the Rust host's faster existing paths:
  - persistent hot/cold contacts;
  - the 3-dispatch block CSR scan;
  - stable colour tiles;
  - `write_sleep_args`;
  - the dual's last-iteration flag.
- **Degree-aware primal lanes:**
  - `record_color_degrees` records each colour's largest CSR degree.
  - `primalLanesForDegree` widens the lanes to 16, 32 or 64 for high-degree
    compounds, when the colour is small enough.
- **Profiling.** `simulation.profileStep()` labels per-phase passes, including
  the convex classify, SAT and manifold phases. `profileStep({singlePass:true})`
  times the whole step in one pass, for an honest total.

## Verification without a GPU

```sh
cargo test -p engine-schema                                  # cooker, ABI decode, oracle, golden bytes
cargo test -p engine-oracles --test convex_slang_cpu         # production Slang kernels on the CPU
cargo test -p engine-webgpu --lib -- --skip gpu::            # naga validation, storage budget
nvm use && node --test crates/engine-webgpu/fracture/*.test.mjs crates/engine-webgpu/tests/*.test.mjs  # Node per .nvmrc
```

`convex_slang_cpu` compiles the real kernels with
`slangc -target cpp -DENGINE_CPU_TEST=1`:

- atomics become sequential operations;
- the SAT entries emulate their 64 lanes phase by phase, which preserves barrier
  ordering;
- the run covers `build_aabbs` → hierarchical broadphase → `narrowphase` →
  `convex_*`.

It checks the results against exhaustive SAT and distance oracles:

- contact existence;
- face/edge axis optimality;
- exact depth for resting contacts;
- points on both surfaces;
- broadphase completeness and uniqueness;
- exact bounds;
- determinism;
- hint stability;
- compounds, uncertified hulls, 64-vertex hulls, and box, sphere, capsule and
  plane pairs.

## GPU validation and benchmarks (run when the GPU is free)

```sh
# WebGPU (Rust) GPU tests, including the new convex pile/manifold tests
cargo test -p engine-webgpu --lib gpu::tests::convex -- --test-threads=1
cargo test -p engine-webgpu --lib -- --test-threads=1
# Metal (includes avbd_slice with a real convex hull)
cargo test -p engine-metal -- --test-threads=1
# Native convex scenes (Convex Pile 10k, Convex Compounds 10k, Convex Rubble 30k)
cargo run -p engine-scenes --release --bin scene_fps -- Convex
```

Browser (`cargo run -p engine-webgpu --bin wgsl_export`, then serve
`crates/engine-webgpu` and open `/fracture/`):

```js
await (await import('./convex-regression.mjs')).runConvexGpuTests({scale: 10000, boundary: true})
await (await import('./regression.mjs')).runFractureRegression()   // Masonry arch, level 0
```

Record:

- chip, browser, adapter, commit and source hashes;
- active bodies, candidates, contacts, manifolds and colours;
- `convexSmall` / `convexLarge`;
- the per-phase and single-pass timestamps.

The baseline to beat is the retired path's dense 10k × 64-vertex prisms run:
about 282 ms/tick, with roughly 96 ms of narrowphase and 106 ms of solve
(`fracture/convex-validation.json`).

## Known limitations and follow-ups

- **Pre-existing Metal overlap.** On Metal, the legacy uniform grid's large-body
  list (scratch 26..41) overlaps `HOST_S(26)` (incremental colouring). The
  hierarchical grid avoids it by using scratch 53..63.

- **Reuse is absolute, not relative.** Contact reuse still uses the absolute
  pose generation (`update_refs`). Reuse based on the relative pose (Jolt style)
  needs body-local normals in persistence and is not implemented. Moving piles
  rely on the SAT hint and the cached-face fast path instead.
- **No separating-axis cache for separated pairs.** Only pairs that had contacts
  carry a hint.
- **Face bias on sliver faces.** On sliver-faced hulls, a kept face can report up
  to a few millimetres less depth than the true penetration. The CPU suite caps
  this at 1 cm, and the worst observed case is 2.8 mm.
- **No compound manifold merging per body pair.** Evaluate it once GPU profiles
  exist.
- **CUDA:** the kernels compile but there is no CUDA host runtime.
- **Grid sizing cache.** It is keyed on the collider count. Changing a body
  between static and dynamic without adding colliders keeps the previous levels.
