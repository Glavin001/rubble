# Physics API review: what is best, and what is best for Rubble

This review covers public APIs in three groups:

- **Rigid-body engines:** Box3D (v0.1.0, 2026-06-30), Box2D v3, Jolt, PhysX 5
  (including its Direct GPU API), Rapier, Bepu v2, Bullet and Unity DOTS.
- **GPU-native simulators:** NVIDIA Newton, MuJoCo/MJX/MJWarp and Genesis, with
  Warp for interop.
- **Destruction systems:** NVIDIA Blast, Unreal Chaos Geometry Collection, PhysX
  APEX Destruction and Houdini RBD.

Each engine is judged by the same question: does the design make sense for a
**GPU-resident, asynchronous, destruction-first** engine that hosts drive
through Rust and, later, a C ABI?

The concrete API that follows from this review is in
[`public-api.md`](public-api.md).

## What Rubble is, for API purposes

Six facts shape the API.

1. **The state lives on the GPU.** Any getter that reads it is a readback, and
   any setter that writes it is an upload. Per-pair host callbacks during a step
   cannot exist.
2. **Results arrive late.** A host learns what a step did one or more frames
   later, through a ring of staging buffers.
3. **It scales on throughput.** It targets 10k+ active bodies, and fracture
   splits of 100k chunks take about 2 ms. Batches and indirect dispatch help;
   per-object calls hurt.
4. **Destruction is its differentiator.**
   - Shardwright assets carry a chunk hierarchy, world-anchor bonds, bond patch
     geometry and a material library.
   - It offers three strength models:
     - bonds as AVBD joints;
     - the fully GPU-resident threshold splitter (`GpuFracture`);
     - the Blast-parity stress solver.
   - Its fracture is forward-only, and its splits are deterministic thanks to
     fixed-point sums.
   - Soft bodies and vehicles also break.
5. **It is lossless and loud.** Defaults never change results. Every trade-off
   is an explicit setting, and limits fail with clear errors.
6. **It is deterministic** for the same build on the same GPU.

## Engine by engine

### Box3D: the base vocabulary

**Best features**

- Opaque IDs, `{index1, world0, generation}`. They are null when zeroed and
  packable into a `uint64`.
- `b3Default*Def()` structs, copied on create.
- Shapes are first-class and attached after their body is created.
- The body origin and the centre of mass are separate.
- Events are flat arrays read after the step:
  - body moves, with `fellAsleep`;
  - contact begin, end and hit (with `approachSpeed` and material ids);
  - sensor begin and end;
  - joint threshold events.
- Events are opted into per shape.
- `b3SurfaceMaterial` carries a `userMaterialId`.
- Joint definitions share a common base and carry force and torque thresholds.
- Recording and replay are validated by state hashes.
- A mover (capsule cast plus collide) supports character controllers.
- A `b3DebugDraw` callback struct, plus profile and counters.

**For Rubble:** adopt nearly all of it as the object model and naming. It is the
newest design in this lineage, it is 3D, and it was built to replace Chaos in an
Unreal game, which is Rubble's own goal. Its API is also the lingua franca of
C physics libraries.

**Do not copy:**

- synchronous getters, which would be hidden readbacks;
- pre-solve and custom-filter callbacks, which cannot run on the GPU;
- the task-system hooks, because the GPU replaces them;
- `b3Pos` double precision for now; offer origin shifting instead.

### Box2D v3

Box3D inherits everything good about Box2D v3. The extra lesson is its move from
v2.4: pointers became IDs and listener callbacks became event arrays. That is the
same move Rubble needs, so it confirms the direction.

### PhysX 5 Direct GPU API: the shape of GPU access

**Best features**

- One batched form for every get and set:
  `(data, gpuIndices, Type, n, waitEvent, signalEvent)`.
- Data is AoS per attribute.
- Each object has an opaque GPU index, separate from its CPU object.
- Contacts can be copied out on the GPU.
- Lifetime high-water statistics, so capacities can be tuned.

**Weaknesses**

- Turning it on turns off sleeping, scene queries, CCD, characters and debug
  drawing.
- Contacts that overflow are dropped with only a warning.

**For Rubble:** Rubble needs this kind of API for engines that render on the
same device, so they can update 10k transforms without a per-object call.

**What Rubble should do differently:** keep every feature working, and use the
staging ring as the ordinary path, with direct access as an addition. Overflow
must be loud: a per-step flag plus an error, never a silent drop.

### Newton and MuJoCo/MJX/MJWarp: the lifecycle and the capacity model

**Best features**

- **Model/State/Control separation** in both Newton and MuJoCo:
  - a builder, then `finalize(device)` into an immutable device model;
  - user-owned, double-buffered states;
  - `notify_model_changed(flags)` for edits to the model.
- **Observables** (Newton): outputs that cost something, such as contact forces
  or accelerations, are allocated and computed only when asked for.
- **Explicit capacities** (MJWarp): `naconmax` and `njmax`, plus a per-world
  overflow bitmask.
- **Static and dynamic fields** (MJX): changing a static field recompiles.
- **Multiple worlds** as an axis, with per-world parameter batching.

**For Rubble:**

- Adopt the **asset/instance split for destruction**. A Shardwright asset is an
  immutable model; a placed destructible is its state.
- Adopt **observables** for contact forces and bond loads. Rubble computes them
  anyway, but reading them back is the expensive part, so it must be opt-in.
- The overflow-bitmask idea is already in Rubble (`overflow_flags`). Add the
  high-water statistics.
- Multiple worlds are useful later, for servers and RL. They are not core now.

**Do not copy:** the Python-first builder, or behaviour after overflow that is
left undefined.

### Jolt

**Best features**

- Batch adds (`AddBodiesPrepare`/`Finalize`), with "always batch" as the rule.
- `BodyID` sequence numbers.
- The ObjectLayer to BroadPhaseLayer table.
- `StateRecorder` with a filter.
- `CharacterVirtual`.
- Its determinism contract is written down.

**For Rubble:**

- Adopt **batch creation**. Uploading a whole scene at once matters at 10k+
  bodies.
- Adopt a **written determinism contract**.
- Adopt a **filtered state snapshot**, later.

**Do not copy:** the threaded listeners, which call back during the step in a
nondeterministic order.

### Rapier and Bevy integration

**Best features**

- Generational arenas.
- Per-collider opt-in for active events.
- A pipeline that holds no state, so every set is passed in explicitly.
- serde snapshots, and `enhanced-determinism`.
- `bevy_rapier`'s cycle of sync components, step, write back. This is the
  template for engine bindings.

**For Rubble:**

- Adopt the **binding pattern** for Bevy, Godot and Unreal adapters.
- Adopt **serde for assets and definitions**.

### Bepu v2

**Best feature:** callbacks as generic structs, so customization costs nothing
at runtime.

**For Rubble:** the GPU equivalent is **Slang interfaces linked when the
pipelines are built**, for custom materials, integrators and damage programs.
Treat that as a future extension point. The default is data-driven tables.

### Unity DOTS Physics

**Best features**

- Components as data, where a missing component means a default.
- Event streams consumed by jobs.
- A world that can be rebuilt every step.

**For Rubble:** this informs a **flat SoA view in the C ABI** (arrays of handles
and transforms) for ECS-style hosts. Do not adopt the stateless rebuild:
Rubble's persistent contacts and sleeping are where its performance comes from.

### Genesis and Warp

- **Genesis:** a `build` step and `envs_idx` subsets. Exceeding a capacity
  *halts* the simulation, which is loud but crude.
- **Warp:** zero-copy interop through DLPack and the CUDA array interface. For
  Rubble, the matching future work is wgpu-hal native-handle export, the plan's
  P8.

### NVIDIA Blast: the destruction model

**Best features**

- **The asset and the instance are separate.** The asset holds the chunk tree,
  support chunks, bonds (normal, area, centroid) and world bonds. A family is a
  pre-sized memory block holding every actor an instance can produce.
- **Damage is data.** Built-in damage descriptors cover radial, capsule, shear,
  triangle and impact-spread damage. Damage programs come as graph and subgraph
  shaders, and the material holds health and thresholds.
- **Damage runs in three stages**, each independently testable and
  interceptable:
  - **generate** is pure and produces fracture buffers;
  - **apply** changes health and produces fracture events;
  - **split** produces split events.
- **Fracture commands are the replicable unit.** They are `{bond nodes,
  health}` and `{chunk, health}` records.
- **Split events carry the parent by value,** as `TkActorData`, because the
  parent may no longer exist when the event is read.
- **Stress solver settings** cover elastic and fatal limits for compression,
  tension and shear. Its inputs are force, gravity and centrifugal terms. It
  reports the number of overstressed bonds, convergence error and excess forces.
  It produces the same fracture commands as the damage path.
- Assets are validated loudly: invalid input returns NULL rather than being
  silently fixed.

**For Rubble:** this is the core of Rubble's destruction API, because Rubble
already *is* this model on the GPU:

- `FractureTables` and pre-allocated body slots play the role of the family;
- `fr_breaks` → `fr_uf_*` → `fr_assign`/`fr_commit` is apply plus split;
- the threshold predictor and the stress profile are the stress path.

What Blast lacks, and Rubble supplies:

- everything runs on the GPU and is batched;
- forward-only splitting inside the step;
- splits that are deterministic thanks to fixed-point sums.

### Unreal Chaos Geometry Collection

**Best features**

- Damage thresholds per cluster level, or derived from material strength and
  connectivity.
- Damage propagation factors for breaks and for shocks.
- Anchors: an anchor field at construction time, plus runtime
  `set_anchored_by_box` and `remove_all_anchors`.
- Removal policy: remove on sleep, remove on break, removal duration.
- One-way debris interaction, by level.
- Per-type event opt-ins (break, crumble, removal, collision, trailing), each
  with a rich event struct.
- The root proxy renders until the collection breaks.

**For Rubble:**

- Adopt **runtime anchoring**.
- Adopt **per-type destruction event opt-ins**.
- Adopt the **root proxy**, so an intact building is one draw.
- Adopt **debris policy as explicit configuration**. Removal is lossy, so it
  must be opt-in and visible in events, never a default.

**Do not copy:** "fields" as a scripting system. Damage descriptors cover the
same use with less machinery.

### PhysX APEX Destruction

**Best features**

- Global budgets: maximum chunk islands, maximum chunks, maximum actors created
  per frame, maximum fractures per frame, and sorting by benefit.
- A report bitmask with a maximum depth for detailed events, while the totals
  stay exact.
- Network sync that sends damage, fracture and chunk motion through separate
  handlers.

**For Rubble:**

- Adopt **budgets with an explicit over-budget policy**, defaulting to fail
  loudly, in line with AGENTS.md.
- Adopt **depth-limited detail with exact totals**.
- Adopt **replicated fracture commands** for vibe-land's netcode: send commands,
  not transforms.

### Houdini RBD, Teardown and Rayfire (briefly)

- **Houdini:** a broken constraint can become a different constraint type
  (`next_constraint_name`). This maps onto Rubble's yield-then-break bonds.
- **Teardown:** connectivity comes from occupancy, and flood fill finds what is
  still anchored. This confirms that anchoring belongs in connectivity
  analysis, which `fr_uf` already does.

## The best overall design

A single winner does not exist. The best design for Rubble is a composition,
with each part taken from the system that does it best.

| Layer | Taken from | Why it is best for Rubble |
|---|---|---|
| Object model, naming, definitions, IDs, shapes, joints, materials, event taxonomy | **Box3D** | The modern standard, 3D, and familiar to every integrator. The C ABI can map one to one. |
| Asynchronous results and frame-latent getters | **Rubble** (no engine does this fully) | The state is on the GPU, so the API must never hide a readback. |
| Batched device access, both indexed and zero-copy | **PhysX Direct GPU API** | The only proven design for updating 10k objects per frame on a shared device. |
| Asset/instance lifecycle, observables, capacity model | **Newton / MJWarp / Blast** | Immutable assets, pre-sized instances, outputs computed only on request, overflow bitmasks. |
| Batch creation | **Jolt** | One upload per scene. |
| Destruction model: damage as data, generate → apply → split, fracture commands, split events by value, stress settings | **Blast** | Rubble's GPU pipeline already has this shape. Blast's API makes it usable and replicable. |
| Anchors, level thresholds, debris policy, root proxy, event opt-ins | **Chaos** | Game-production needs that Blast leaves to the integrator. |
| Budgets with policy, depth-limited events, network sync | **APEX** | Large scenes and multiplayer, without silent degradation. |
| Record and replay with state hashes | **Box3D** (with Jolt's filter) | Determinism is a Rubble guarantee, and this proves it in CI. |
| Engine bindings | **bevy_rapier** pattern | One adapter shape for Bevy, Godot and Unreal. |
| Customization | Data tables now; Slang interfaces later (**Bepu**-style) | No callbacks are possible on the GPU. |

## Refinements: what changed after each review pass

1. **First pass (Box2D-like):** IDs, definitions, frames and events. These were
   correct but incomplete.
2. **Box3D pass:**
   - shapes become first-class;
   - the body origin is separated from the centre of mass;
   - contact, hit and sensor events follow Box3D's taxonomy;
   - per-type joint definitions with thresholds;
   - record and replay.
3. **GPU-native pass:**
   - no synchronous getters;
   - batched indexed access and observables;
   - high-water statistics alongside the overflow flags;
   - batch creation;
   - the C ABI also offers a flat SoA view.
4. **Destruction pass:** destruction becomes a first-class pillar of the API,
   not a late phase. It gets:
   - an asset/instance split;
   - damage descriptors;
   - Blast's three stages mapped onto Rubble's GPU kernels;
   - replicable fracture commands;
   - split events carrying the parent by value;
   - runtime anchors;
   - budgets that fail loudly by default;
   - explicit debris policy;
   - root-proxy rendering;
   - per-type destruction events.

## Open questions to settle by measuring

- **Contact begin/end detection on the GPU.** It costs a "matched last step" bit
  on persistent pairs. Measure it with events off, which must stay free, and
  with events on, before committing to the design.
- **Which strength model is the default for a destructible?** The threshold
  splitter is fully resident and fast. `blast_parity` is the charter baseline
  but is host-coupled. The plan offers both behind one API, chosen per
  destructible, with `Threshold` as the resident default.
- **Fracture-command replication across GPUs.** Commands make the *topology*
  deterministic, because the same bonds break and the same pieces form.
  Positions still diverge across different GPUs, so clients need correction
  snapshots for body poses.
