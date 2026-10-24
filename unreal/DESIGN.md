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

### 1. Rubble's kernels run in Unreal's render graph

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

Two alternatives, both rejected as the end state:

- **Readback (today):** grows with bodies on the CPU.
- **Rubble on its own wgpu device, sharing buffers with Unreal through wgpu-hal:** zero copy is
  possible, but it costs cross-queue fences (`MTLSharedEvent` or D3D12 shared fences) every
  frame, two schedulers fighting over one GPU, and reaching into Unreal's RHI internals. It is
  kept only as a fallback if the HLSL path stalls. The renderers below read "a GPU buffer of
  poses" and don't care which host filled it.

### 2. Rigid bodies: GPU-driven instancing

- **One custom primitive with a Rubble vertex factory.** The vertex shader fetches the body's
  pose by instance id from Rubble's pose buffer. It interpolates previous → current by the
  clock's alpha and outputs motion vectors from the previous pose. Nothing about the pose is
  written on the CPU.
- **Buckets by mesh, LOD and material.** Unit box, sphere, capsule and each hull mesh live in one
  vertex and index atlas.
- **A cull-and-bucket compute pass:** frustum and HZB occlusion, distance LOD, then compaction
  into per-bucket indirect draw arguments. Sleeping bodies cost a 32-byte read. At 100k that is
  3.2 MB read and about 0.1 ms.
- **Expected cost at 100k bodies:** a few dozen draws and 1–10M triangles, depending on hull
  detail. Shadows draw them again through the same factory.

### 3. Destructibles: one draw per asset, not per chunk

- **Each asset's chunk meshes go into one vertex buffer,** each vertex tagged with its chunk.
  The vertex shader computes chunk world = body pose[chunk → body] × chunk local, which is the
  render contract.
- **Chunk → body and chunk-local transforms are GPU buffers,** written by Rubble's split kernels
  (the `fr_view_*` views). A split never touches the CPU and never rebuilds a mesh; chunks just
  read another body's pose.
- **Placed instances of one asset share the buffer.** Instance id gives the destructible's
  offset, so 100k chunks across many buildings come to a handful of draws.
- **The CPU still gets split, bond and hit events** for dust, debris, audio and gameplay. These
  are compacted on the GPU and read back as a few KB.

### 4. Soft-body vehicles: skinned on the GPU, at render rate

- **Rubble's own `soft_skin` kernel** (Slang, the same source as the browser) runs once per
  rendered frame, not once per physics substep. It reads node positions in place and writes
  position, normal, strain and torn flags into a skin buffer, much like Unreal's GPU skin cache.
- **A skin vertex factory** reads that buffer by vertex id:
  - torn triangles collapse in the vertex shader, so index buffers are never rebuilt;
  - strain drives the dent and scratch mask;
  - motion vectors come from the previous frame's skin buffer.
- **Render LOD only:** near cars get the full flexbody mesh (`engine-softbody` `visual.rs`); far
  cars get the collision skin (nodes plus triangles). The simulation is identical at every
  distance, so this is a visual LOD, not a physics trade-off.
- **Expected cost:** 200 cars × 20k vertices = 4M vertices skinned at three nodes each, about
  1–2 ms a frame on an M3 Max. The current path would need tens of ms on the CPU plus the
  uploads.

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

1. **Proof of the host: one rigid scene in RDG.** Slang → HLSL → global shaders, with a C++ plan
   replayer. Gate: poses bit-equal to the wgpu host on the same GPU after N steps (the way
   `pipelined_gate` tests hosts today).
2. **The GPU-driven rigid renderer:** the vertex factory, cull and bucket pass, and indirect
   draws. Gate: 100k bodies drawn with less than 0.2 ms of CPU.
3. **The destructible renderer:** one draw per asset, with GPU chunk views.
4. **Soft skinning in RDG, with the skin vertex factory and render LOD.** Gate: 200 cars drawn
   with less than 0.5 ms of CPU.
5. **Async-compute overlap and budget profiling.** Run one GPU workload at a time while
   measuring.

The readback client keeps working throughout, as the reference picture and the fallback. Its
tests (`Rubble.Abi`, `Rubble.Render`) become the parity checks for each phase.
