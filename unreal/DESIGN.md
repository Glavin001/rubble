# Rubble in Unreal at scale: a GPU-resident design

## Target

| | |
|---|---|
| Rigid bodies | 100,000+ (any number awake: a building coming down wakes most of it) |
| Soft-body vehicles | dozens to hundreds, deforming, with detailed render meshes |
| Destructibles | the rigid budget above, as chunks of many structures |
| Frame rate | 60–120 FPS: **16.7–8.3 ms per frame**, shared by physics and rendering on one GPU |

## Hard requirements

- **Nanite and Lumen stay on.** The bodies feed Unreal through its own primitive paths (GPU
  Scene instances, skeletal meshes), never a custom vertex factory.
- **Every Unreal platform.** Mac (Metal), Windows (D3D12, Vulkan), Linux (Vulkan), consoles and
  mobile. Anything API-specific is a stepping stone, not the design.

**Platform floor for Nanite.** In UE 5.8, Nanite, virtual shadow maps and hardware ray tracing need
the SM6 shader platform: D3D12 or Vulkan SM6 on Windows and Linux, and `METAL_SM6` on Mac.
`METAL_SM6` needs an M2 or later GPU **and macOS 15 or later**. On older systems `MetalRHI.cpp`
falls back to `METAL_SM5` ("SM6 is enabled but is not supported on this system"), which has no
Nanite. Lumen (software) still runs on `METAL_SM5`. Our design adds no limit beyond Unreal's
own.

## The rule this sets

**Per-body and per-vertex data never touches the CPU.** At 100k bodies, 60 ns of CPU work
each is 6 ms, most of a 120 FPS frame, before anything is drawn. The CPU handles work that
grows with events: breaks, hits, the player's vehicle, ray results, the camera target. It never
handles work that grows with bodies or vertices.

The client today does the opposite, measured on an M3 Max (`scripts/unreal/test.sh`,
`Rubble.Render.Perf`):

| Path today | Cost per frame | Grows with |
|---|---|---|
| Pose readback, then `FTransform` copy, merge and lerp, then ISM instance updates | 6 ms CPU at 100k moving bodies | bodies |
| A mesh component per fracture chunk, each moved separately | render-thread cost and one draw per chunk | chunks |
| Soft skin readback, then CPU strain and tear loops, then a full procedural-mesh rebuild | about 1–2 ms for 20 cars × 2.5k vertices; tens of ms at 200 cars × 20k | vertices |

It is a good stepping stone, but it cannot reach the target.

## The design: Rubble runs inside Unreal's frame on Unreal's GPU

```
Game thread (CPU): input, events, gameplay           -- O(events)
    | commands (impulses, vehicle input, spawns)   ^ compact readback: events, player telemetry, rays
    v                                               |
Render graph (RDG), one GPU, one device:
  [async compute]  Rubble step x k (replayed engine_exec plan)
                     -> pose_pack (current + previous pose, 32 B/body)
                     -> fracture views (chunk -> body)    -> soft_skin (vertices, normals, strain, torn)
  [graphics]       cull + bucket (compute, GPU)  -> indirect draws with the Rubble vertex factory
                   base pass, VSM shadows, velocity (motion vectors from the previous pose)
```

### 1. Rubble's buffers live on Unreal's GPU device

The renderers in sections 2–4 are compute passes inside Unreal's render graph, so Rubble's pose,
chunk and skin buffers must be on Unreal's device. **The host that reaches every platform is the
render-graph replayer** (below): Rubble's Slang compiled to HLSL goes through Unreal's own shader
pipeline, which cross-compiles to Metal, D3D12, Vulkan, the consoles and mobile.

**The Metal bridge is a development stepping stone only.** Rubble's wgpu runs on Unreal's
`MTL::Device` and command queue (wgpu-hal `device_from_raw`, `queue_from_raw`,
`buffer_from_raw`), and Rubble copies its outputs into buffers Unreal owns. It lets the renderers
be built and verified on a Mac before the replayer exists. It does not generalise:

| Platform | Interop through wgpu | Why it isn't the design |
|---|---|---|
| Mac (Metal) | works (`engine-webgpu` `metal_interop`) | Mac only |
| Windows (D3D12) | wgpu-hal `from_raw` exists | Rubble has never run on wgpu's D3D12 backend; needs cross-queue `ID3D12Fence` sync |
| Windows and Linux (Vulkan) | wgpu-hal `device_from_raw` exists | `VkQueue` is not thread-safe: submit on Unreal's RHI thread or add timeline semaphores |
| Consoles, mobile | none | wgpu does not run there |

#### The render-graph replayer (the cross-platform host)

Rubble already writes each step as a list of commands that no host owns:
`engine_exec::rigid_plan::encode_step` and the fracture `Planner`. Every host replays that
list: the browser in JavaScript, the Metal viewer, `WgpuSim`. An Unreal host is one more
replayer.

- **Rust (through the C ABI)** builds the plan: buffer sizes, the words the host uploads, and
  the ordered dispatches and copies. This is single-source logic, as with the browser.
- **C++** replays the plan into RDG passes. The kernels are Rubble's Slang, compiled to HLSL and
  loaded as Unreal global shaders. Unreal then cross-compiles them to Metal, D3D12 or Vulkan.
  This needs the HLSL target that `public-api.md` plans for P8.
- **Every buffer is an RDG buffer on Unreal's device.** Renderers read the pose, chunk and skin
  buffers in place, so nothing is copied, nothing is read back, and nothing synchronises across
  devices.
- **Physics goes on async compute** where the platform supports it: step N+1 overlaps the
  rendering of frame N, and the renderer draws the last finished step, interpolated by alpha.

Two alternatives to the replayer:

- **Readback (today):** grows with bodies on the CPU.
- **Rubble on its own wgpu device and queue, sharing buffers:** cross-queue fences every frame
  and two schedulers on one GPU. This is replaced by the interim host above, which shares
  Unreal's device and queue.

### 2. Rigid bodies: written into Unreal's GPU Scene, rendered as ordinary instances

Rubble bodies stay ordinary instanced static meshes as far as Unreal is concerned, so
**Nanite, Nanite virtual-shadow-map caching, Lumen (distance fields and the surface cache follow
GPU Scene transforms), motion vectors, culling and HLOD all apply unchanged.** Only the place
the transforms come from changes:

- **One instanced primitive per bucket** (mesh and material): the unit box, sphere, capsule,
  each hull mesh. Each is created with GPU-only instance data (`FInstanceSceneDataBuffers`,
  GPU-only), sized for the bucket's bodies.
- **Each frame, a compute pass writes the instances' transforms** straight into GPU Scene through
  `FSceneInterface::UpdatePrimitiveInstancesFromCompute` with an `FGPUSceneWriteDelegate`. This
  API is public in 5.8, marked experimental since 5.6. Unreal's own NiagaraNanite and PCG GPU
  instancing use it to drive Nanite instances from the GPU. The pass reads Rubble's pose buffer
  (previous and current, 32 B per body), interpolates by the clock's alpha, and writes the
  instance's local-to-world and previous local-to-world, which gives motion vectors.
- **Nothing per body touches the CPU.** At 100k bodies the pass reads 3.2 MB and writes about
  6 MB, on the order of 0.1–0.2 ms.

A custom vertex factory was the first draft. It was rejected because it loses Nanite and Lumen's
full participation, and every future Unreal feature would need porting to it.

**Lumen and GPU-only instances: a hybrid.** In 5.8, Lumen's distance-field scene and mesh cards
exclude primitives with GPU-only instance data (`DistanceFieldObjectManagement.cpp`,
`LumenScene.cpp` and `LumenMeshCards.cpp` check `!IsInstanceDataGPUOnly()`). Such bodies are
still lit by Lumen (GI, reflections, screen traces) and keep Nanite and VSM, but they don't
occlude or bounce light in Lumen's world-space tracing. Lumen culls small objects there anyway,
and 100k moving distance fields would not fit in a frame. So:

- **Large bodies** (vehicles, wall and roof sections, anything above a size threshold) use
  ordinary, CPU-visible instance data and take part in Lumen fully. They are few, so the CPU
  cost is small: their poses come from the compact readback.
- **Small debris**, the bulk of the 100k, use GPU-only instances written by the compute pass.

**Proof of concept (`ARubbleGpuScenePoc`, test `Rubble.Poc.GpuScene`, run with
`-RubblePoc=gpuscene`).** 10k boxes with GPU-only instance data. A compute pass
(`RubbleGPUScene.usf`, through `UpdatePrimitiveInstancesFromCompute`) writes their transforms every
frame from a pose buffer in Rubble's layout and frame. Checked on an M3 Max with macOS 14.4
(`METAL_SM5`):

- the instances draw and move from the GPU-written transforms;
- they cast shadows;
- Lumen lights them: with GI off, their shadowed faces lose the ground's bounce.

Nanite could not be checked on this machine, because it needs macOS 15+ (above). The Nanite
proxy path (`FRubbleNaniteProxy`) is built, and the test checks it wherever the platform supports
Nanite.

### 3. Destructibles: chunks as GPU Scene instances

- **Each chunk is an instance of its own chunk mesh,** written by the same pass as chunk
  world = body pose[chunk → body] × chunk local. The chunk → body and chunk-local buffers are
  GPU-resident and written by Rubble's split kernels. This is how Unreal renders Chaos geometry
  collections with Nanite: one instance per piece, in GPU Scene.
- **A destructible's chunks share one primitive.** Unique chunk meshes become one Nanite mesh
  with one instance per chunk, or a mesh per chunk under one primitive per asset, so draw count
  stays per asset, not per chunk. Nanite's cost grows with visible triangles, not instances.
- **A split never touches the CPU;** chunks just read another body's pose. Split, bond and hit
  events still reach the CPU, compacted, for dust, audio and gameplay.

### 4. Soft-body vehicles: Nanite skeletal meshes with exact affine bones

**Rubble's skin is exactly bone skinning.** `soft_skin` places each render vertex in a frame of
three nodes (centre c, nodes x and y), with coefficients fixed at bind time:
`v = c + a(x−c) + b(y−c) + d((x−c)×(y−c))`. With F = [x−c, y−c, (x−c)×(y−c)] now and F₀ at rest,
that is `v = M v₀ + t`, where `M = F F₀⁻¹` and `t = c − M c₀`: an affine bone transform applied to
the rest vertex with weight 1. So:

- **Each car is a Nanite skeletal mesh.** It has one bone per distinct binding frame (a few
  hundred to a couple of thousand per car) and one weight-1 influence per vertex. Nanite's bone
  skinning reproduces Rubble's deformed positions exactly.
- **The bone matrices are computed on the GPU** by a small Rubble Slang kernel (`soft_frames`,
  the same formula as `soft_skin`), from node positions in place. They reach Nanite through
  Unreal's GPU-only transform provider: `UTransformProviderData` and
  `FTransformProviderRenderProxy`, public in 5.8, the hook instanced and Nanite skinning use.
  Nothing per car goes through the CPU.
- **The vehicle assets are built in the editor,** because Nanite data has no runtime build. A
  commandlet builds each vehicle type's skeletal mesh from Rubble's flexbody render mesh and its
  bindings: bones, rigid weights, Nanite on. The bindings are deterministic from the descriptor
  and the mesh.
- **Damage and tearing go through UV-space textures,** since Nanite skinned meshes take no
  per-vertex dynamic attributes. Each frame a compute pass splats Rubble's per-vertex strain and
  torn flags into a small texture in the car's UV layout:
  - the paint material samples strain for dents and scratches;
  - a masked Nanite material (programmable raster, cars only) clips torn regions.
- **Normals are the only approximation.** Bone skinning carries rest normals through each bone's
  matrix, while Rubble smooths normals over the deformed neighbours. They agree until panels
  shear. If the difference shows, the UV texture carries a normal correction. Measure first.
- **Lumen** treats the cars as skinned meshes: screen traces, and hardware ray tracing through
  its deformed fallback when enabled.
- **Expected cost:** 200 cars × about 1,000 bones × 48 B is under 10 MB of bone matrices a frame.
  The splat pass grows with vertices, but only for visible cars.

**Fallback: Deformer Graph.** A `UMeshDeformer` kernel running `soft_skin` gives the exact skin
through the GPU skin cache, without Nanite. It is used where Nanite isn't available (mobile) and
as the reference that validates the bone path.

**Needs from Rubble:** the skin bindings (node triple and rest frame per vertex) and the
procedural car's flexbody mesh (`engine-softbody/src/visual.rs`) on the ABI.

### 5. What the CPU still reads back

All of it is asynchronous, 1–2 frames late, and grows with events rather than bodies:

- the event ring, already GPU-compacted: joint, bond, mount and beam breaks, splits, pieces,
  and P4 hits;
- player vehicle telemetry and the chase-camera target;
- ray batches for gameplay;
- counters and overflow flags.

### 6. Rendering features at this scale

- **Nanite:** not for simulated bodies (custom vertex factory). Static scenery stays Nanite.
- **Virtual shadow maps:** moving instances invalidate their pages every frame, which is a
  known cost at 100k awake. Distant moving geometry can use a coarser shadow LOD, which changes
  shadows, not physics.
- **Lumen:** software tracing doesn't see custom vertex factories, so dynamic bodies are lit by
  screen traces and the static world's surface cache. Hardware ray tracing would mean rebuilding
  bottom-level structures for 100k movers every frame, which is too costly.
- **Motion blur and TSR** come from the per-instance and per-vertex motion vectors.

## Frame budget (120 FPS, 8.3 ms)

| GPU work | Budget | Notes |
|---|---|---|
| Rubble rigid step(s) | ≤ 3 ms | async compute, overlapping rendering |
| Rubble soft steps (240 Hz: 2 per frame at 120 FPS) | ≤ 2 ms | |
| Cull, bucket, skin | ≤ 1.5 ms | |
| Base pass, shadows, Lumen, post | ≤ 4–5 ms | scene-dependent |

CPU: under 2 ms of Rubble work, all of it per event.

## What this needs from Rubble

1. **An HLSL target in `engine-slang-build`** (P8), validated by the kernels compiling as Unreal
   global shaders on Metal and D3D12. Risk areas: atomics, wave operations, indirect dispatch,
   and up to 31 bound buffers.
2. **A host-plan C ABI.** The plan words, buffer sizes and upload words go through FFI, as the
   wasm `Plan` and `RigidStep` already do, so the C++ replayer stays thin and holds no physics.
3. **Stable GPU layouts for the outputs** (pose record, chunk views, skin output), and their
   binding indices in the plan.
4. **GPU-compacted events and telemetry** for the readback.
5. **Simulation cost at the target, which is the larger problem.** Measured today:

   | Workload | Cost |
   |---|---|
   | 100k awake bodies | about 110 ms a frame |
   | 10k awake bodies | about 1–2.5 ms a step |
   | One soft car | 3.5–5 ms a step at 240 Hz (`docs/softbody.md`) |
   | 20 soft cars | about 60 ms a frame |

   The integration can be brought to near zero cost, but the simulation itself is 10–50× over
   budget at the target when most bodies are awake. That is Rubble engine work, and it is
   lossless only (AGENTS.md). Batching soft vehicles shares colour dispatches across cars, so
   the cost should grow less than linearly with car count, but this needs measuring.

## Phases

1. **Rubble on Unreal's device.**
   - **1a, the Metal bridge (development):** wgpu over Unreal's `MTL::Device` and queue, with
     output buffers owned by Unreal. Gate: exported poses equal Rubble's own.
   - **1b, the render-graph replayer (every platform):** Slang → HLSL, Unreal global shaders,
     and the plan replayed in RDG behind a backend trait in `rubble::World`. Gate: bit-equal
     poses against the wgpu host on the same GPU.
2. **The GPU Scene rigid renderer:** GPU-only instanced primitives per bucket, with poses
   written by `UpdatePrimitiveInstancesFromCompute`. Gate: 100k bodies drawn with less than
   0.2 ms of CPU, with Nanite and Lumen still working.
3. **The destructible renderer:** chunks as GPU Scene instances, with GPU chunk views.
4. **Soft cars as Nanite skeletal meshes with exact affine bones:**
   - the `soft_frames` kernel and a GPU transform provider;
   - an editor commandlet that builds the vehicle assets;
   - UV-space strain and tear textures.

   Gate: 200 cars drawn with less than 0.5 ms of CPU, with positions equal to `soft_skin`'s.
   Deformer Graph is the fallback.
5. **Async-compute overlap and budget profiling.** Run one GPU workload at a time while
   measuring.

The readback client keeps working throughout, as the reference picture and the fallback. Its
tests (`Rubble.Abi`, `Rubble.Render`) become the parity checks for each phase.
