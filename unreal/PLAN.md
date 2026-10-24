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

| | |
|---|---|
| Load | 3.5 s |
| At rest | 16–18 ms a frame (render thread 15 ms: 2,633 batches without Nanite) |
| All 52k chunks awake | 34–75 ms per Rubble step |

**Open: the town's bond strengths under World's bond model (needs a decision).** The town was
tuned for vibe-land's stress solver. Under `rubble::World` (bonds as breakable joints), the
`town_probe` runs (a 7 t ball at 45 m/s into a house, then a 22 t meteor onto the next) show:

| `strength_scale` | Result |
|---|---|
| 1 (the pack's own) | A progressive collapse: about 250 bonds broken after 3 s and about 9,500 a second by 8 s, the whole town awake by 9 s. Bond preload does not settle in 600 steps and does not stop it. |
| 1.1 | The same, starting about 3 s later |
| 1.25, 1.5, 2, 4 | Nothing breaks, not even under the meteor; about 3k bodies wake |

So impacts barely break bonds. What fails is self-weight in bonds that are right at their limit.
Also, the street surfacing is one dynamic group that touches every building, so the town is a
single sleep island: once it moves, nothing re-sleeps. Making the streets static exceeds the
1,024 large statics (`Overflow(512)`, README gap 8).

The options are engine fracture calibration for packs, a re-export of the pack, or the lab's stress
model. The scene's `?strength=` parameter (default 1, the pack's own) stays as is until decided.

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
