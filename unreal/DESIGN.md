# Rubble in Unreal at scale: a GPU-resident design

## Target

| | |
|---|---|
| Rigid bodies | 100,000+ (any number awake: a building coming down wakes most of it) |
| Soft-body vehicles | dozens to hundreds, deforming, with detailed render meshes |
| Destructibles | the rigid budget above, as chunks of many structures |
| Frame rate | 60–120 FPS: **16.7–8.3 ms per frame**, shared by physics and rendering on one GPU |

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
chunk and skin buffers must be on Unreal's device. There are two hosts, and the renderers don't
care which one filled the buffers.

- **Interim, and first: Rubble's wgpu on Unreal's Metal device.**
  - The Metal RHI exposes `RHIGetDevice()` and `RHIRunOnQueue()`. wgpu-hal can wrap an existing
    `MTL::Device`, queue and buffer.
  - Unreal allocates the output buffers (RDG external buffers). Rubble writes into them through
    imported wgpu buffers.
  - Submissions go on Unreal's queue, so Metal's queue order replaces cross-device fences.
  - The whole `rubble::World` API keeps working unchanged.
- **End state: Rubble's plan replayed in the render graph,** described below. This needs
  `World`'s uploads routed through a backend trait instead of wgpu, a large Rust change.

#### The render-graph replayer (end state)

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

### 4. Soft-body vehicles: Unreal's Deformer Graph, with Rubble's skin kernel

- **Each car is a skeletal mesh** carrying the detailed flexbody render mesh, with a
  `UMeshDeformer` (Deformer Graph / Optimus). Its compute kernel is Rubble's `soft_skin`,
  compiled to HLSL. The kernel reads the car's node positions from Rubble's buffer and writes
  deformed positions and normals into the GPU skin cache.
- **The car gets everything a skinned mesh gets:** ray-traced dynamic geometry for Lumen,
  virtual shadow maps, motion vectors, the material system. Strain goes to a vertex-colour
  output that drives the dent and scratch mask. Torn triangles collapse in the kernel, so index
  buffers are never rebuilt.
- **The kernel runs once per rendered frame,** not once per physics substep.
- **Render LOD only:** near cars use the full flexbody mesh, far cars the collision skin. The
  simulation is identical at every distance.
- **Expected cost:** 200 cars × 20k vertices = 4M vertices at three nodes each, about 1–2 ms a
  frame.
- **Open question:** Nanite for deformed meshes depends on 5.8's Nanite-skinning support for
  deformers. Without it, cars are non-Nanite skinned meshes, which still get the features above.

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

1. **Rubble on Unreal's device.** wgpu over Unreal's `MTL::Device` and queue, with the output
   buffers allocated by Unreal and imported into wgpu. Gate: the same poses as Rubble on its own
   device, and Unreal reading them in a render-graph pass with no CPU copy. The render-graph
   replayer (Slang → HLSL, plan replay) follows separately; the HLSL target is needed for
   phase 4 anyway.
2. **The GPU Scene rigid renderer:** GPU-only instanced primitives per bucket, with poses
   written by `UpdatePrimitiveInstancesFromCompute`. Gate: 100k bodies drawn with less than
   0.2 ms of CPU, with Nanite and Lumen still working.
3. **The destructible renderer:** chunks as GPU Scene instances, with GPU chunk views.
4. **Soft skinning through Deformer Graph,** running Rubble's `soft_skin` in HLSL, with render
   LOD. Gate: 200 cars drawn with less than 0.5 ms of CPU.
5. **Async-compute overlap and budget profiling.** Run one GPU workload at a time while
   measuring.

The readback client keeps working throughout, as the reference picture and the fallback. Its
tests (`Rubble.Abi`, `Rubble.Render`) become the parity checks for each phase.
