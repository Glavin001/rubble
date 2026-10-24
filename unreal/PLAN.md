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

Not yet verified by the PoC:

- **Nanite:** the proxy path is built and tested where the platform supports Nanite, but it has
  not run yet.
- **Virtual shadow maps.**

Lumen lighting of GPU-only instances and ordinary shadows are confirmed.

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
3. **Phase 2: the GPU Scene rigid renderer on real Rubble poses** *(SM5 ok)*:
   - Unreal side of 1a:
     - create the world with `rbCreateWorldOnMetal` on Unreal's `MTL::Device` and queue
       (`IMetalDynamicRHI`);
     - allocate the current and previous pose buffers on the render thread;
     - pass their `MTLBuffer`s and offsets (`FMetalRHIBuffer::GetCurrentBuffer()`) to
       `rbWorld_ExportPoses`.
   - One `URubbleInstancesComponent` per bucket (box, sphere, capsule, each hull). The instance →
     pose slot table comes from `rbBody_GetPoseIndex`.
   - The Lumen hybrid: large bodies stay on CPU-visible instances fed by the compact readback.
   - Gate: 100k bodies with under 0.2 ms of CPU, matching the readback client's picture.
     `Rubble.Render` becomes the parity check.
4. **Phase 3: destructibles as GPU Scene instances** *(SM5 ok)*:
   - Needs chunk → body and chunk-local buffers on the GPU, and `rbDestructible_GetChunkLocal`
     (README gap 2).
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

- **Simulation cost at the target** is the largest gap: about 110 ms a frame for 100k awake bodies,
  and about 60 ms for 20 soft cars (DESIGN.md, "What this needs from Rubble"). Only lossless
  optimisations are allowed (AGENTS.md).
- **The README's C ABI gaps 1–11**, and the HLSL target and host-plan ABI for Phase 1b.

## Rules kept while doing this

- **No per-body or per-vertex CPU work** in the at-scale path. The readback client stays as the
  reference and the fallback.
- **No custom vertex factory.** Everything goes through Unreal's own primitives, so Nanite and
  Lumen apply.
- **Commit and push along the way.** Benchmark one GPU workload at a time.
