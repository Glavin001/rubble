# Shardwright → GLB → Slang AVBD / WebGPU

A local, dependency-free browser demo importing **one self-contained GLB** with
`SHARDWRIGHT_fracture` v1. Shardwright does the slow fracture/hull cooking offline.
The lab reads baked meshes, convex hulls, full mass/inertia tensors, bonds,
hierarchy, and the resolved physical material library. Reading, scene building
and the per-tick host plan run in Rust compiled to wasm (no fracture bake runs
in the browser); the same Rust runs every scene natively on Metal
([`fracture_viewer`](../../../README.md#native-fracture-viewer-macos--metal)).
GPU shader compilation is a separate startup cost.

## Run

The `assets/*.glb` files are stored in Git LFS. If they are ~130-byte pointer files,
run `git lfs install && git lfs pull` first.

From the Rubble repository root, with the Slang toolchain described in
[`engine/TOOLCHAIN.md`](../../../TOOLCHAIN.md):

```sh
export RUSTUP_TOOLCHAIN=stable
export WGPU_BACKEND=vulkan
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
cargo run --manifest-path engine/Cargo.toml -p engine-webgpu --bin wgsl_export
wasm-pack build engine/crates/engine-fracture-wasm --release --target web \
  --out-dir ../engine-webgpu/generated/fracture-wasm --no-typescript
python3 -m http.server 8777 --bind 127.0.0.1 --directory engine/crates/engine-webgpu
```

Open <http://127.0.0.1:8777/fracture/> in a WebGPU-enabled browser. Generated WGSL
and wasm are ignored by git, so rerun the export after checkout or shader changes,
and the `wasm-pack` build after changes to `engine-fracture`. `wasm-pack` comes
from `web/`'s devDependencies (`npm --prefix web ci`) or `cargo install wasm-pack`.

## Shared architecture

Each piece of logic has one source, used by both hosts:

| Piece | Source | Browser (WebGPU) | Native (Metal) |
|---|---|---|---|
| GLB / FRAC reader, metadata | `engine-fracture` `asset.rs` | wasm `Asset` | direct |
| Scene building: levels, compound hulls, bonds, rows/grids/suburbs, roads, cuboid house, Scene menu | `engine-fracture` `scene.rs`, `cuboid.rs`, `bonds.rs`, `samples.rs` | wasm `Scene` | direct |
| Hull cooking and the hull arena (ABI v2) | `engine-schema` `convex.rs` | wasm `HullArena` | direct |
| Per-tick host plan: buffer sizes, host words, step state, ordered dispatches | `engine-fracture` `plan.rs` | `simulation.mjs` replays it | `engine-viewer` `sim.rs` replays it |
| Physics kernels | `slang/kernels/*.slang` (`ENGINE_WGSL` layout) | WGSL (`wgsl_export`) | MSL (`engine-viewer` `build.rs`, from the plan's entry list) |
| Camera, shot rays, floor, indexed mesh | `engine-fracture` `view.rs` | wasm `Orbit` | direct |
| Draw shader | `slang/render/fracture_draw.slang` | WGSL (`wgsl_export`) | MSL |

`engine-fracture-wasm` is the wasm-bindgen surface; `wasm.mjs` loads it and adapts
types. `Planner` emits each tick as flat command words: a copy (from the tick's host
upload or between buffers), a direct dispatch (kernel, workgroups, pass-uniform
slot), an indirect dispatch (kernel, argument slot, pass slot) or a phase marker
(profiling and frame-slicing boundaries). The hosts only create buffers and
pipelines, replay commands, read counters back into the plan, and draw. The JS
left is GPU I/O, the page and input. On a recording mock device the
replayed commands are byte-identical to the former JS host's, uploads included.

**Deterministic parity shot.** Both hosts can run the same scene, deterministic
budget, shot and tick count:

```sh
cargo run -p engine-viewer --release --example fracture_compare -- arch 0 240
```
```js
(await import('./parity.mjs')).parityShot(__FRACTURE_DEMO__.device,{scene:'arch',level:0,ticks:240})
```

Measured on an M3 Max after 240 ticks:

| Scene | Metal standing | Browser standing | Metal contacts @240 | Browser contacts @240 |
|---|---|---|---|---|
| arch, level 0 | 5 of 23 | 5 of 23 | 85 | 95 |
| house, level 0 | 70 of 131 | 70 of 131 | 969 | 780 |
| cuboid 7×7 grid | 31,907 of 35,035 | 31,927 of 35,035 | 46,760 | 47,286 |

The arch's counters are identical through tick 120. Each host repeats its own
run exactly; between hosts, f32 differences in the generated shader code (Slang →
MSL directly vs Slang → WGSL → Dawn) grow through contact-rich, chaotic motion.
The default **House** scene contains 131 building parts, 163 baked collision
hulls and 454 bonds. The Scene selector also retains the 23-piece masonry arch.
Both start at rest on the floor. Click to shoot an orange sphere; drag to orbit
and scroll to zoom. Impacts wake sleeping pieces. Reset reloads 16 shots.
Sphere trajectories compensate for gravity to aim near the clicked scene depth.

Live counters update approximately once per second: **TPS** counts completed
physics ticks (target 120), **FPS** counts submitted render frames, and **Sim
speed** is TPS / 120 (1× means real time). Pausing physics leaves rendering active,
so TPS drops to zero while FPS continues. Measurements exclude loading and hidden
tabs. Physics ms/tick includes CPU encoding/submission and the GPU/readback wait;
these are wall-clock measurements, not GPU timestamps. Frame ms is the interval
between rendered frames, including physics blocking, not isolated rendering cost.

The house includes four levels: **131 / 567 / 3,158 / 5,905 fragments**. The arch
has **23 / 165 / 171 / 392**. Only one level is active at a time. The house defaults
to level 0 because the current browser host and convex path are still experimental;
finer levels increase collision work substantially. Assets load only on selection.
Levels 2 and 3 bake 35 and 36 flat (zero-volume) sheets into compound fragments;
the importer skips those hulls (every fragment keeps real ones) and says so.

**20 houses · Level 3 in a row** places twenty level-3 houses along +z (131,840
bodies, 133,280 hulls) from `assets/veneer-house-hull16.glb`. Copies share one
hull arena and draw instanced. “Fire the wrecking ball” rolls a 1.6 m, 200 t ball
at 40 m/s along the ground through the whole row (`__FRACTURE_DEMO__.fireRow()`;
`benchmarkRow()` records per-phase GPU timings). The scene budgets 8 candidates
and 8 contacts per collider and 64 colours and colouring rounds, and runs with a
16-vertex hull limit: a larger hull is a load error, never reduced.

**49 houses · Level 2 in a 7×7 grid** uses the bundled `veneer-house.glb` (64-vertex
hull limit; 154,742 bodies, 245,882 hulls) and fires the ball down the middle
column. On an M3 Max an all-asleep step costs 2.6 ms of GPU time and the peak after
the shot ~38 ms (~114k awake, ~880k contacts). Unbonded pieces knock over
neighbouring houses, so most of the grid ends up awake. The 16-vertex re-bake
measured the same peak with 15% more pieces, so the grid keeps the original asset.

**Suburb · 48 houses on lots with roads** (`build_suburb` in `engine-fracture`) spaces the same houses
out: 4 × 3 blocks of 2 × 2 lots with 5-6 m yards, separated by 8 m roads with
sidewalks and dashed centre lines (drawn on the ground; the physics ground is
still the plane). The ball rolls down one column of lots and hits 6 houses. On an
M3 Max 24 of 48 houses collapse (debris still topples unbonded lot neighbours;
43 of 49 on the grid), ~55k pieces stay awake, and the step settles at ~15 ms
(~66 ticks/s) by 30 s. 10 m yards still lose 22 houses.

**Cuboid suburb · 96 box-built houses** uses the same kind of layout, twice the
size (6 × 4 blocks of 2 × 2 lots, about 69k pieces), with a procedural
house made only of box colliders (`engine-fracture` `cuboid.rs`, ~715 pieces): running-bond
brick walls with interlocking corners, door and windows with lintels, sills and
glass, stepped brick gables, a ridge beam, rafters and roof boards, an exterior
chimney, a porch on posts with a step, and a picket fence with a mailbox. Every
piece is a loose body held by gravity and friction (μ = 0.5); `engine-fracture` `tests/cuboid.rs`
checks that no two pieces start interpenetrating.
- The roof pitch is 12°: woken whole, a house stands and sleeps again in 2.6 s.
  At 18° the settling jolt starts the rafters sliding and the roof falls in.
- The scene uses zero-gap contacts (`FLAG_CONTACT_GAP`). With the default margin
  each course rests 2 mm deep, so the walls and the taller gables sink unevenly.
- On an M3 Max the shot collapses 12 of 48 houses, 6-12k pieces stay awake, and
  the GPU step is 3.5-4.2 ms (box-box contacts, no hull SAT), about 4x faster than
  the hull suburb. The remaining cost is mostly the primal's per-colour dispatches.

**49 box-built houses in a 7×7 grid** puts the same cuboid house on the dense
grid layout. Measured live in the browser on an M3 Max after the row shot
(wall-clock ticks including counter readback): 145-166 ticks/s for physics alone
and 79-86 ticks/s with a frame rendered every tick; 19 of 49 houses collapse by
20 s and up to ~20k pieces are awake. The hull grid of the same layout runs at
~33 ticks/s.

**Overlap physics ticks** (checkbox, opt-in) stops waiting for each tick's
counter readback: up to three ticks are queued ahead of the GPU and counters
are read in the background. The GPU runs the same ticks in the same order;
only host-side uses of the counters lag a tick or two (HUD, overflow report,
colour-slot estimate, few-awake solver choice, the all-asleep skip, which now
applies only when the counters describe the latest tick). On the cuboid grid
after the shot, 1 tick per call goes from 100 ticks/s waiting to 274
overlapped; the hull grid, GPU-bound at ~28 ms per tick, goes from 28.9 to
30.2. In lockstep, overlapped and waiting runs stay bit-identical until the
engine's own run-to-run nondeterminism appears (two plain runs of the cuboid
grid diverge at tick ~296).

### Bond modes

The **Bonds** menu picks how the baked bonds hold the fragments, on the same asset,
level, camera and shot rule, so the three can be compared:

| Mode | What holds the fragments | Where it runs |
|---|---|---|
| **None** (default) | Nothing: pieces start asleep and wake when hit | rubble (this page's WGSL host) |
| **Breakable D6 joints** | Each bond a breakable AVBD weld (below) | rubble (this page's WGSL host) |
| **Stress solver** | The GPU stress solver's bonds (`engine/stress-oracle`, profile `blast_parity`): rubble steps the bodies, its contacts load the bond graph, and a quasi-static stress cascade breaks bonds and splits fragments on the GPU | Rust on wasm with its own WebGPU device and canvas (`stress-oracle/crates/stress-web`) |

**Stress solver mode.**
- It runs the same Rust as the native reference (`stress-rubble` example
  `fracture_glb`): the level's Shardwright stress export, stress-ref's import, the
  resident GPU frame, and `stress_rubble::render`.
- It uses the lab's orbit camera and shot rule (`engine-fracture` `view.rs`,
  `plan::shot_velocity`).
- It runs one copy of the structure; the samples' layouts of copies are the other modes'.
- **JSPI.** The solver reads the GPU synchronously, so in the browser each read
  suspends the wasm stack to the event loop through JavaScript Promise Integration
  (`engine-gpu-wait`, `gpu-wait.mjs`, the page's import map). It needs Chrome 137 or later.
- **Build:**
  ```sh
  CARGO_PROFILE_RELEASE_DEBUG=0 wasm-pack build engine/stress-oracle/crates/stress-web --release --target web \
    --out-dir ../../../crates/engine-webgpu/generated/stress-web --no-typescript
  ```
  `engine/scripts/build-web.sh` includes this step.

**Native reference** (house, level 0, one cannonball, 120 frames, M3 Max): 15 fragments,
151 of 453 bonds broken, median frame 4.2 ms.

**Status in Chrome (M3 Max):**
- It loads, draws, aims and fires like the other modes, and it holds real time (60 physics
  frames/s, 2.3 ms each) until a ball strikes.
- **Nothing breaks yet.** Chrome allows 10 storage buffers per shader stage, and the
  stress kernels for contact ingest and GPU splits bind up to 26 (`layout`), so WebGPU
  rejects those pipelines. Packing them to 10 or fewer is the remaining work.
- The same WGSL kernels run natively (`STRESS_GPU_SHADERS=wgsl`, where Metal allows 31)
  with the Metal library's result: a lab shot (×64 ball) breaks 210 bonds into 17
  fragments on both.
- **WGSL rules the Slang source now meets:**
  - no constant infinity or NaN (`common.slang` `float_bits`);
  - barriers only under workgroup-uniform control flow (`group_uniform`,
    `group_uniform4`, `group_uniform_island`, and uniform reductions in `group_sum2`);
  - storage access read from the shader's declarations (`gpu.rs` `wgsl_access`).
  Metal's values are unchanged (guard).
- **View options.** "Inspect bond graph" and "Color by sleep state" are this page's
  renderer's, so they are disabled in this mode.

### Breakable D6 joints (experimental)

“Breakable D6 joints” turns each baked bond of the selected level into
a D6 bond joint (`JointDesc::bond`, `JOINT_FLAG_BOND`; rows in
`slang/physics/joints.slang`, see `engine/docs/JOINTS.md`):

- **Rows.** Six AVBD rows in the bond frame (normal, u, v; bending about u and v;
  torsion) about the baked rest pose. Each row has a fixed penalty and an
  accumulated multiplier.
- **Penalties.** They are Shardwright's elastic springs (`kn = A/ce`, `ks = A/cg`,
  `ktu,v = Iuu,vv/ce`, `kt = J/cg`; `bonds.rs bond_springs`) times one
  scene-wide factor, so redundant bonds share load in the bake's proportions.
  Rows stiffer than 5× their lighter body's inertial term are capped:
  stiffer rows lock bonded clusters in Gauss-Seidel.
- **Breaking.** A bond breaks when `max(Fn⁺/T + M/Mcap, Fn⁻/C, V/S) > 1` on the
  step's final iteration. The capacities come from Shardwright's reference
  stress solver: material and interface strengths × area × strength scale
  (`bond_capacities`). Broken bonds leave the exact joint-pair hash, so their
  pieces collide again.
- **Sleep and wake.** Bonds join sleep islands and wake their partners.
- **Preloading.** The scene first settles one house with breaking off
  (`Scene::preload_variant` / `settled`, driven by each host) and starts every copy from those poses.

`bond-regression.mjs` checks the welds on the GPU:

| Case | Result |
|---|---|
| Strong hang | holds within 0.03 mm |
| Weak hang | breaks |
| Resting in compression | holds |
| Cantilever | 0.1° tilt |
| Weak bending | breaks |
| Bonded pair released from rest | falls and lands intact |

**Measured on house scenes (M3 Max), not yet good enough to enable by default.**
With 8 iterations a whole house of per-fragment welds never fully settles:
- long spans creep (roof pieces still sag after 6 s);
- 20 g slivers jitter;
- a woken house therefore never sleeps, and about 1% of its bonds break under
  self-weight.

| | 49-house grid | 20-house row |
|---|---|---|
| Peak GPU per step | ~30 ms (38 ms unbonded) | ~36 ms |
| Bonds broken | 11% | 47% (Level 3) |
| Idle (all asleep) per step | 6.4 ms | 5.7 ms |

The likely fix is structural: simulate each intact bonded island as one compound
rigid body (exactly rigid, sleeps, one body per standing house), evaluate bonds
with a stress solver, and split islands where bonds break, as Shardwright's own
reference simulator does.

`veneer-house-hull16.glb` is a fresh Shardwright bake of the same source
(`benchmarks/assets/veneer_house.glb`, seed 1337) with `assets/veneer-house-hull16.toml`
(building_lite plus `max_hull_vertices = 16`, enforced by its `collision_shapes`
gate), exported with `export-glb --uncompressed`. Current Shardwright fractures
this source into 142 / 614 / 3,619 / 6,592 fragments, more than the bundled
house's bake, independent of the cap. Level 0 has one 7 m slab hull whose stored
faces are 16 µm non-planar, beyond the f32 allowance of the convexity check, so
that level is rejected; the row uses level 3.

Drop another Shardwright fracture GLB onto the page or use the file picker.
“Download extracted physics metadata” exposes the parsed physical asset and
materials for another consumer, including a future stress solver.

## Export your already-baked asset

In the Shardwright checkout (the CLI binary is named `prefracture`):

```sh
export RUSTUP_TOOLCHAIN=stable
export WGPU_BACKEND=vulkan
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
cargo run -p frac-cli --release -- export-glb \
  --input out/my-model.glb \
  --physics out/my-model.fracphys \
  --out out/my-model.fracture.glb \
  --uncompressed
```

`--physics` defaults to the input path with `.fracphys` suffix. Supply
`--materials path/to/materials.toml` when the bake used a custom material library.
`--uncompressed` is required by this minimal demo's accessor reader; the exporter
also supports preserving meshopt for other consumers. Packaging never regenerates
fractures. The resulting file is the only asset the page needs.

Bundled asset provenance: Shardwright's `out/structs/masonry_arch/masonry_arch`
render/physics pair, seed 1337, built-in material library 1.0.0. Exported using the
command above with `--uncompressed`; all four levels are included (751 fragments,
2,821 bonds total). The asset's bake settings hash remains in its physics payload.

The house is Shardwright's `veneer_house` asset, not the separate
`two_storey_building` benchmark. Source: `out/structs/building_lite/veneer_house/`
render/physics pair, seed 1337, built-in library 1.0.0, exported with
`prefracture export-glb --uncompressed`. Its 54.1 MB GLB includes all four levels;
no source sidecars or runtime bake are required.

## Contract and implementation

- `engine-fracture/src/asset.rs`: bounds-checked GLB / schema-1 FlatBuffers reader;
  reusable independently of any GPU. Exposes full tensors, bond frames/areas/moments,
  composition, strength scales, reinforcement, boundary loops and child mappings.
  Interface material IDs share the material ID space:
  `library.interface_material[id - library.material.length]`; -1 means unspecified.
- `engine-fracture/src/scene.rs`: selects a hierarchy level, converts full inertia
  to a principal basis, transforms both render and collision geometry to that
  basis, and cooks baked convex topology into the **packed hull ABI v2** with
  `engine-schema`'s cooker (`hull-golden.json` pins its bytes). Each compound child
  is recentred on its own bounds and placed by the collider's local position. CPU
  work is decoding/conversion, not hull generation.
- `simulation.mjs`: replays the plan (`engine-fracture/src/plan.rs`), which
  schedules Rubble's **core** Slang AVBD kernels with
  persistent hot/cold contacts, the hierarchical size-level broadphase, the convex
  narrowphase (`convex_classify` / `convex_sat_small|large` / `convex_manifold`),
  the block CSR scan, stable colour tiles, degree-aware primal lanes and sized
  sleep work — the same schedule as `engine-webgpu/src/gpu.rs`. Rendering reads
  GPU body poses directly. Batches contain 1–4 ticks with one counter readback per
  batch; rendering is submitted before awaiting that readback. Slow ticks reduce
  the batch size, and fully sleeping scenes skip GPU physics.
- Convex collision lives in the core library (`slang/physics/convex.slang`,
  `slang/kernels/convex_step.slang`); see
  [`engine/docs/convex-hulls.md`](../../../docs/convex-hulls.md). The former
  `fracture_step.slang` kernels and `fracture_export` were retired.

The vendor extension stores the original FRAC bytes in a standard GLB bufferView
and embeds the full material library in root JSON. It is optional for rendering
and is **not a Khronos standard**. See Shardwright's `docs/GLB_FRACTURE.md` and
`schemas/frac.fbs` for the versioned contract. Generic optimizers must preserve or
remap the custom bufferView and embedded glTF node references.

## Scope

This demonstrates **import and rigid-body collision**, not bond failure or a stress
solver. Bonds are inspection lines. Fragments start sleeping at their baked poses on the
floor and wake through contact with a projectile or moving neighbor. This initial
rest state is not a solved bonded equilibrium; the world anchors are visualized
but are not fixed constraints. Physical strength properties are available
to consumers but are not converted into AVBD breakable constraints. Contact
friction is currently a demo-wide 0.5; surface shading uses glTF base colors.
Textures, arbitrary edited node transforms, mixed hierarchy frontiers and runtime
fracture refinement are not implemented. Input must follow Shardwright's baked
asset/COM coordinate contract (meters, Y-up, right-handed).

Convex limits are 64 vertices, 128 faces and 192 edges per hull. Hull pairs are
compacted into a small bucket (≤16 vertices: 16 lanes per pair, 4 pairs per
workgroup) and a large bucket (64 lanes per pair). SAT works in one hull's local
frame from shared memory and prunes edge axes with adjacent-face Gauss maps; the
previous frame's feature gives hysteresis and a cached-face fast path that skips
the edge query. Clipping streams points into a deepest + extremes reducer (no
polygon buffers) and keeps at most four points (`MAX_CONVEX_CONTACTS`).
Generation-gated contact reuse skips SAT/clipping for unchanged contact patches.
Compound mass and inertia remain on the body; child hulls are never replaced by
spheres.

At import, exact duplicate vertex records are welded for adjacency, and coplanar
face patches are normalized. Original vertex coordinates/order remain available.
Uncertifiable edge adjacency selects exhaustive vertex-projection SAT instead of
unsafe Gauss-map pruning. Degenerate zero-volume hulls are rejected with their
fragment/hull ID. The bundled house's levels 2 and 3 contain such source hulls and
need recooking; levels 0 and 1 remain usable. A failed level selection preserves
the previous scene. No GLB extension change or runtime fracture is involved.

The hierarchy inserts each collider into one power-of-two size level. Queries
visit their own and coarser levels; fixed/sleeping fine colliders also probe coarse
movers. Exact cell coordinates disambiguate hash collisions. Same-body pairs are
excluded and pair ownership prevents duplicates. Cell widths conservatively cover
proxy diameters, including f32 padding. There is no fixed-capacity large-body list
or pair-truncating chain shortcut. Storage capacities still report overflow.

The target remains **10,000+ active rigid bodies**. Correctness at that count does
not imply a 120 TPS performance guarantee. Use the scale regression below and
record hardware, active counts, candidates, contacts and timings. The optional
`simulation.profileStep()` reports GPU timestamp phases when supported; ordinary
HUD timings include queue/readback waits. Projectiles still use discrete collision,
not CCD. Bonds remain inspection metadata rather than stress constraints.

The Gauss-map/contact-selection port is based on three-avbd at `2bb1a318`;
its MIT copyright/license is preserved in `engine/slang/THIRD_PARTY_LICENSES.txt`.

## Headless physics benchmark

Physics only: no canvas and no renderer, with WebGPU on the real GPU in
headless Chrome. It prints JSON with ms/tick (ticks queued back to back),
the counters, and per-kernel GPU times.

```sh
nvm use
node engine/crates/engine-webgpu/fracture/headless.mjs grid warm=420 measure=120
node engine/crates/engine-webgpu/fracture/headless.mjs cuboidGrid budget='{"deterministic":true}' trace=60
```

- Scenes: `grid`, `suburb`, `house`, `cuboidGrid`, `cuboids`.
- `shot=none` skips the row shot. `profile=0` skips the kernel timings.
- `budget` is merged into the scene capacity. Use
  `{"deterministic":true}` for run-to-run identical results, so two shader
  builds can be compared tick for tick.
- `CHROME=` picks the browser (default: Google Chrome).
- `bonds=1` uses the lab's breakable bonds. `columns=`/`rows=` change the
  grid layout.
- `staticUnhit=1` makes the houses outside the shot's column static. This
  changes behaviour and is only for estimates.
- `CHROME_ARGS` passes extra browser flags, for example Dawn toggles.
- No npm dependencies; see `headless-page.mjs` for the protocol.

## Verification

```sh
nvm use   # .nvmrc: Node 22.22 (node:module stripTypeScriptTypes needs >= 22.13)
node --test engine/crates/engine-webgpu/fracture/*.test.mjs
cargo test --manifest-path engine/Cargo.toml -p engine-oracles --test convex_slang_cpu
```

`simulation.test.mjs` drives the browser host with a mock WebGPU device against
the exported WGSL (no GPU). The node tests run the lab's wasm path (build it
first, see *Run*); the scene, bond, hull and cuboid-house logic is tested in Rust:

```sh
cargo test --manifest-path engine/Cargo.toml -p engine-fracture --release
cargo test --manifest-path engine/Cargo.toml -p engine-viewer --release --test fracture_regression
``` `convex_slang_cpu` runs the production Slang convex
and broadphase kernels on the CPU (`slangc -target cpp`) against exhaustive
oracles; see `engine/docs/convex-hulls.md`.

Checks the real bundled asset's hierarchy/materials, asymmetric inertia conversion,
render/collision frame agreement and malformed containers. In the browser console,
after selecting **Masonry arch**, level 0:

```js
await (await import('./regression.mjs')).runFractureRegression()
```

Checks 120 GPU steps without pre-impact drift, then shoots the crown and runs
720 more steps. Checks wake-up, fragment motion, contacts, finite poses, unit
orientations, no overflow, hull/sphere floor support, and exact reset of all body
fields and ammunition. Shardwright's `cargo test -p frac-io --test
fracture_glb` checks byte-preserving physics round trips, compressed/uncompressed
render compatibility, determinism and invalid pairings.


For GPU convex conformance and a 10,000-active-body, 64-vertex-hull workload:

```js
await (await import('./convex-regression.mjs')).runConvexGpuTests({scale: 10000})
// add boundary:true for work-list row-limit cases (32,768+ SAT pairs)
```

The conformance suite compares actual GPU contacts with exhaustive CPU SAT,
checks exact bounds, arbitrary rotations, mixed-size/static/sleeping/compound
pair ownership and duplicates, and covers 64-vertex prisms plus 124-face/186-edge
bipyramids. It also checks an open compound frame (a sphere in the hole must not
collide), a sphere touching its jamb, and batched contact-reuse pose parity.
It also covers hull/box, hull/sphere, hull/capsule and hull/plane pairs, dense
compounds, SAT work-list sizes, and reuse/hint on versus off. Scale results
(dense 64-vertex prisms with reuse off, 4-child compounds, small rubble) include
active count, contacts, candidates, overflow, wall time and per-phase plus
single-pass GPU timestamps; no asleep benchmark shortcut. Shader changes require
rerunning `wgsl_export`. The packed hull ABI is v2; do not mix older host code or
generated shaders with the new importer.

The measurements below predate the core convex path (they were taken with the
retired `fracture_step` kernels) and are kept as the baseline to beat.

The recorded [local validation](./convex-validation.json) on 2026-10-14 used
Chrome WebGPU with an Apple / metal-3 adapter. The dense scale workload kept
10,000 bodies active, with about 35,900 candidates and 112,100 contacts, and no
overflow. Five ticks averaged **282 ms/tick**; a separate timestamp sample spent
about **96 ms in narrowphase and 106 ms in AVBD solving**. These are local,
non-isolated measurements, not a real-time result. Further optimization must
address both collision work and the shared solver/dispatch costs.

For the impacted house level 1, five timestamp samples had roughly 7,600–7,700
candidates and 7,700 contacts; narrowphase took 0.8–15 ms and solving 86–136 ms.
The timings varied substantially. This confirms that tighter bounds/contact
reuse do not by themselves eliminate the solver bottleneck. The core convex path
also runs on the native Metal and WebGPU (Rust) hosts; CUDA only compiles it.
