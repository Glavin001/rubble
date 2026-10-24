# RubbleDemo: plan and status

The design and its reasoning are in [DESIGN.md](DESIGN.md). Setup and the C ABI gaps are in
[README.md](README.md). This file covers what is done, what is blocked, and the order of the
remaining work.

Last updated: 2026-10-24, branch `feat/unreal-client`, PR #71.

## Goal

The goal is Rubble's web demos in Unreal Engine 5.8 with much better graphics:

- rigid scenes;
- vehicles;
- deformable soft-body cars;
- the fracture lab.

It has to work at Rubble's target of 100k+ rigid bodies and dozens to hundreds of soft cars, at
60–120 FPS. Nanite and Lumen stay on, and it has to run on every Unreal platform.

## Done

| What | Where | Verified by |
|---|---|---|
| Client on the C ABI: every family, with a mock only for scenes the ABI can't build yet | `Plugins/Rubble/Source/RubbleRuntime` | `scripts/unreal/test.sh` (`Rubble.Clock`, `.Mock`, `.Abi`, `.Glb`, `.Render`) |
| The design: GPU Scene instances, exact affine bones for soft cars, the render-graph replayer | `DESIGN.md` | review |
| Phase 1a in Rust: a world on a Metal device and queue supplied by the host, with poses exported into buffers the host owns | `engine-webgpu` `metal_interop.rs`; `rbCreateWorldOnMetal`, `rbWorld_ExportPoses`, `rbBody_GetPoseIndex` | `engine-ffi` `tests/metal_export.rs` |
| GPU Scene PoC: a compute pass writes instance transforms from a pose buffer in Rubble's layout | `URubbleInstancesComponent`, `RubbleGPUScene.usf`, `ARubbleGpuScenePoc` | `Rubble.Poc.GpuScene`, `Rubble.Poc.Scale` |
| 100k boxes through the PoC: 12.2 ms a frame, game thread flat at 1.2 ms | DESIGN.md section 2 | `RUBBLE_POC_SIDE=317 … --only Rubble.Poc.Scale` |
| **Phase 2:** the world on Unreal's Metal device (a queue of its own), poses exported into a ring of four Unreal-owned slots (`FRubbleGpuPoses`), small dynamic bodies drawn as GPU Scene instances; static and large bodies (`rubble.GpuScene.MaxExtentCm`, 150 cm) stay CPU-placed for Lumen | `RubbleGpuPoses`, `ARubblePrimitives` | `Rubble.Render` (20 tests) on the GPU path; screenshots equal the readback path's at the same step |
| **Phase 3:** small fracture chunks (per-chunk bodies) as GPU Scene instances, shown on shatter | `ARubbleFracture` | bookkeeping (every chunk drawn once), house and arch screenshots |
| Moves stay per event: GPU-drawn bodies report only sleep and wake | `rbBody_SetMoveReports` | `engine-ffi` `tests/abi.rs` |
| Bodies whose centre of mass is off their origin draw right | `rbBody_GetPoseOffset` | `tests/metal_export.rs` |
| The page's rings and gables (its builders, masses, friction, iterations, sleep, `reuseContacts`, capacity tiers) | `MockRigidScene.cpp` `WebRing`/`WebGables`, `RbWorldDef.reuse_contacts` | `Rubble.Render.Destruction` |

`Rubble.Render.Destruction` on the M3 Max (`METAL_SM5`, 1280×720), GPU path:

| Scene | Bodies | Frame | Rubble per step | Renderer CPU |
|---|---|---|---|---|
| `rigid:ring110k` | 110,453 | 70.8 ms | 17.3 ms (the page: ~10 ms) | 0.04 ms |
| `rigid:ring` | 9,143 | 10.4 ms | 0.8 ms | 0.02 ms |
| `rigid:gables` | 6,734 | 11.8 ms | 0.8 ms | 0.02 ms |
| `rigid:pile` | 32,012 | 11.8 ms | 0.1 ms | 0.01 ms |
| `rigid:wall` | 2,012 | 9.5 ms | 0.6 ms | 0.01 ms |
| `fracture:house` | 131 chunks | 20.5 ms (p95 73 ms at the shatter) | 6.2 ms | 0.01 ms |
| `fracture:arch` | 23 chunks | 12.1 ms | 0.6 ms | 0 ms |

`ring110k` on the readback path: 189 ms. The rest of its frame is Rubble's step: the integration
is no longer the bottleneck at this scale.

Not yet verified by the PoC:

- **Nanite:** the proxy path is built and tested where the platform supports Nanite, but it has
  not run yet.
- **Virtual shadow maps.**

Lumen lighting of GPU-only instances and ordinary shadows are confirmed.

## The town demo (`fracture:town`, in progress)

vibe-land's Vibe Town ScenePack: 70 buildings (houses, shops, a cinema, a library, five
ten-storey towers), streets and props, 51,921 chunks and 132,529 bonds, one destructible.

- **Setup:** `scripts/unreal/fetch-town.sh` (the pack) and `scripts/unreal/fetch-city-textures.sh`
  (vibe-land's 2K Poly Haven arrays, CC0). Both copy from a vibe-land checkout; neither is
  committed.
- **Drawing:** every chunk from the pack's own collider, with 27k boxes and 2,627 shared hulls in
  2,633 instanced batches. `M_RubbleTownChunk` projects the texture arrays in chunk space plus the
  chunk's place in the town.
  - **The hybrid:** resting chunks are Lumen-visible CPU instances. A chunk that wakes moves to a
    GPU Scene instance drawn from Rubble's poses, and back when it sleeps. Moves are sleep and wake
    only, so a frame carries at most about 1.8k moves while 46k chunks move.
- **Controls:** F / click fires a 60 cm iron ball at 45 m/s; M drops a 1.2 m meteor (about 22 t)
  at 50 m/s.
- **Tests:** `Rubble.Town.Shots` (fixed cameras) and `Rubble.Town.Destroy` (cannon, meteors,
  timeline).
- **Rubble fixes made for it:**
  - v2 packs' per-material bond limits (before this, every bond was unbreakable);
  - `rbDestructible_GetChunkPose`;
  - the pack node per chunk;
  - creation poses before a body's first frame.
- **Look** (`M_RubbleTownChunk`, `M_RubbleTownGlass`, `M_RubbleTownGround`):
  - each material's texture tinted to its colour, an up-facing layer (slab tops, slate on the
    pitched roofs), a palette per building (the six siding colours), weathering and ground grime;
  - glass over interior-mapped, randomly lit rooms; the towers (modelled without openings) get
    framed windows on their outer faces, in each chunk's own frame so they break with it;
  - grass to the fogged horizon, a warm late-afternoon sun, volumetric clouds;
  - collapse dust tinted by what broke, sized by the bonds broken, lingering and drifting.
- **Fixed: library shapes.** The pack loader read no `nodeColliders` of kind `shape`
  (`scenario.shapeLibrary`): 24,561 of the town's chunks (every tower wall among them), and the
  lab's `house-1story`, `house-2story`, `villa-savoye` and `comp-wall-bay` nodes of that kind,
  collided as their `nodeSizes` boxes. They are their library hulls now; an unknown collider
  kind is an error. Hosts draw pack nodes at `rbDestructible_GetChunkNodeFrame` (a hull's
  pose is its centre of mass on its principal axes). Town physics load: 2.0 s to 3.5 s.

| | |
|---|---|
| Load | 3.5 s |
| At rest | 16–18 ms a frame (render thread 15 ms: 2,633 batches without Nanite) |
| All 52k chunks awake | 34–75 ms per Rubble step |

**Fixed: the town's explosions were the solver, not its bonds.** Under World's default solver
(60 Hz, no lab flags) a struck town gained energy: chunks still bonded vibrated at 30–90 m/s,
then the town came apart and threw chunks up to 4.8 km in 10 s, at random (1–3 runs in 3, in
metres or centimetres, hull or box colliders alike). The earlier "progressive collapse under
self-weight" reading was wrong. Fixes:
- the town runs the solver its pack is tuned with (`rbWorld_UseDestructibleAssetSolver`: 120 Hz,
  8 iterations, the lab's contact, inertia, sleep and post-stabilization flags);
- World's step plan now runs post-stabilization (`FLAG_POST_STAB`): without
  `record_stab_velocity` a body's velocity was never updated under the lab's flags (no gravity,
  no response to contacts), which also affected the house and arch scenes;
- World's bond joints take their angular penalties in the D6 row order (bending and twist
  were swapped);
- cannonballs and meteors are CCD bullets (20 cm roofs and slabs are thinner than their step);
- hits from 1 m/s (the threshold is in cm/s here).

Now (`town_probe`, 3 runs each, and `Rubble.Town.Destroy`): stable, repeatable, the town settles
back to sleep, at most 4,290 chunks awake.

**The town runs the threshold strength model** (by decision, 2026-10-24; `?bonds=1` keeps the
bond joints). It is World's D1b (`StrengthModel::Threshold`, PR #79, merged into this branch):
the fracture lab's threshold mode on the world's own device.
- The world uses canonical contacts, at most 2,097,120 (the contact ingest's dispatch; placing
  fails above that). The asset's threshold solver steps at 60 Hz. The lab's ground (its grade,
  50 cm up) is placed on Unreal's.
- Cannonballs and meteors are projectiles (`rbBody_SetProjectile`): strike prediction loads the
  town as the lab's shots do.
- A cluster's chunks share one body: 281 bodies at rest for 51,921 chunks. Split events move
  their instances to the child bodies (`ARubblePrimitives::Rebind`); every piece's origin is the
  town's frame, so nothing jumps.
- `Rubble.Town.Destroy`: 1.4 ms of physics a frame (44 ms under bond joints), frames 14 ms,
  64 break events. Cannonballs hole the house walls; the meteors break pieces off the tower roofs
  but rest on them.

**Support loss (by decision, 2026-10-24; `?supportloss=0` turns it off).** The threshold model
has no static solve, so a tower stood while one tie to its footings held: 16 balls of 22 t at
60 m/s still left 8 columns and 12 wall pieces. `DestructibleDef::support_loss` (engine-world
`support_loss.rs`, opt-in):
- after a frame's breaks, each touched anchored piece's weight flows down its intact bonds to its
  supports, and a bond carrying more than its compressive capacity breaks the next frame;
- tipping: a piece whose centre of mass is outside the footprint of its loaded support bonds
  loses them all and falls.
In compression alone a tower holds on 2–3 of its 16 columns (13 MN against about 6.4 MN a column),
so it is the tipping that brings one down. `Rubble.Video.Town` shoots tower A's column rows from
the west, a volley a second (eight balls at 1.0 and 2.6 m), and it topples at about 3 s.

Under bond joints, strikes broke nothing: under the pack's limits a 22 t meteor at 50 m/s broke
no bond, in World and in the lab's own runner (`engine-viewer/tests/town_lab_meteor.rs`).

## Blocked: Nanite and virtual shadow maps need macOS 15

UE 5.8 enables `METAL_SM6`, which Nanite, virtual shadow maps and hardware ray tracing need, only
on macOS 15+ with an M2 or newer GPU. The development Mac runs macOS 14.4, so Unreal falls back to
`METAL_SM5`. The whole client runs without Nanite and VSM there today, not just the PoC.

**Once the Mac is on macOS 15**, run these first, before any more Rubble rework:

1. `scripts/unreal/test.sh --no-build --only Rubble.Poc.GpuScene`
   - The Nanite assertion runs instead of the warning.
   - `poc/nanite_triangles.png` shows the boxes coloured by cluster.
   - Shadows come from virtual shadow maps.
2. `RUBBLE_POC_SIDE=317 scripts/unreal/test.sh --no-build --only Rubble.Poc.Scale`
   - 100k boxes on Nanite. Record the numbers in DESIGN.md next to the `METAL_SM5` ones.
   - Watch the cost of VSM page invalidation with every box moving.
3. Run the whole suite and re-shoot the family screenshots on SM6. The static scenery and
   materials have only ever been seen without Nanite and VSM.

If anything fails there, fix the GPU Scene path before going further.

## Next, in order

Steps marked *(SM5 ok)* can be built and checked on the current Mac. Their Nanite checks come
later, with the runs above.

1. **PoC 2: Nanite skinning driven from the GPU (the soft-car route, DESIGN.md section 4).**
   - A skeletal mesh with one bone per binding frame and one weight-1 influence per vertex.
   - Bone matrices come from a compute pass through `UTransformProviderData` /
     `FTransformProviderRenderProxy`.
   - Gate: deformed positions match the `soft_skin` formula, and Nanite renders the mesh (needs
     SM6).
   - Before then, check whether the transform provider also drives non-Nanite GPU skinning on
     SM5. If it doesn't, build the Deformer Graph fallback first.
2. **Clean-ups to the GPU Scene write path** *(SM5 ok)*:
   - Move from the deprecated `GPUWriteParams` to `GPUSceneWriterUB`.
   - Keep an eye on `UpdatePrimitiveInstancesFromCompute`, which is experimental in 5.8.
   - One primitive per GPU chunk: fine for hundreds; a city of destructibles needs chunk meshes
     merged per asset (DESIGN.md section 3).
3. ~~Phase 2~~ and 4. ~~Phase 3~~: done (above).
5. **Phase 4: soft cars**, after PoC 2. It needs:
   - the `soft_frames` kernel;
   - the editor commandlet that builds the vehicle skeletal meshes;
   - UV-space strain and tear textures;
   - from Rubble: the skin bindings and the flexbody render mesh on the ABI (README gaps 1 and
     5).
6. **Phase 1b: the render-graph replayer**, for Windows, Linux, the consoles and mobile:
   - from Rubble: an HLSL target for the kernels (P8) and a host-plan C ABI;
   - gate: bit-equal poses against the wgpu host.
7. **Phase 5:** async-compute overlap and frame-budget profiling, with one GPU workload at a
   time.

## Rubble engine work this depends on (not in this branch)

- **Simulation cost at the target** is the largest gap. `ring110k` takes 17.3 ms per step
  through `rubble::World` here, against ~10 ms on the page's own host; 20 soft cars take about
  60 ms a frame (DESIGN.md, "What this needs from Rubble"). Only lossless optimisations are
  allowed (AGENTS.md).
- **The page's scene builders on the ABI.** The rings and gables are ported from
  `scene-builders.mjs` into C++; `engine-scenes` already has them in Rust, so exposing those
  (with each scene's solver settings) would make one source for the web, Unreal and the tests.
- **The README's C ABI gaps 1–11**, and the HLSL target and host-plan ABI for Phase 1b.

## Rules kept while doing this

- **No per-body or per-vertex CPU work** in the at-scale path. The readback client stays as the
  reference and the fallback.
- **No custom vertex factory.** Everything goes through Unreal's own primitives, so Nanite and
  Lumen apply.
- **Commit and push along the way.** Benchmark one GPU workload at a time.
