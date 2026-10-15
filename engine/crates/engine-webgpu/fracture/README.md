# Shardwright → GLB → Slang AVBD / WebGPU

A local, dependency-free browser demo importing **one self-contained GLB** with
`SHARDWRIGHT_fracture` v1. Shardwright does the slow fracture/hull cooking offline.
The browser reads baked meshes, convex hulls, full mass/inertia tensors, bonds,
hierarchy, and the resolved physical material library. No Rust/Wasm bake runs in
the browser. GPU shader compilation is a separate startup cost.

## Run

From the Rubble repository root, with the Slang toolchain described in
[`engine/TOOLCHAIN.md`](../../../TOOLCHAIN.md):

```sh
export RUSTUP_TOOLCHAIN=stable
export WGPU_BACKEND=vulkan
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
cargo run --manifest-path engine/Cargo.toml -p engine-webgpu --bin wgsl_export
python3 -m http.server 8777 --bind 127.0.0.1 --directory engine/crates/engine-webgpu
```

Open <http://127.0.0.1:8777/fracture/> in a WebGPU-enabled browser. Generated WGSL
is ignored by git, so rerun the export after checkout or shader changes.
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

### Breakable bonds (experimental, opt-in)

“Breakable bonds (experimental)” turns each baked bond of the selected level into
a `JOINT_BOND` weld (`world_step.slang`):

- **Rows.** Six AVBD rows in the bond frame (normal, u, v; bending about u and v;
  torsion) about the baked rest pose. Each row has a fixed penalty and an
  accumulated multiplier.
- **Penalties.** They are Shardwright's elastic springs (`kn = A/ce`, `ks = A/cg`,
  `ktu,v = Iuu,vv/ce`, `kt = J/cg`; `geometry.mjs bondSprings`) times one
  scene-wide factor, so redundant bonds share load in the bake's proportions.
  Rows stiffer than 5× their lighter body's inertial term are capped:
  stiffer rows lock bonded clusters in Gauss-Seidel.
- **Breaking.** A bond breaks when `max(Fn⁺/T + M/Mcap, Fn⁻/C, V/S) > 1` on the
  step's final iteration. The capacities come from Shardwright's reference
  stress solver: material and interface strengths × area × strength scale
  (`bondCapacities`). Broken bonds leave the exact joint-pair hash, so their
  pieces collide again.
- **Sleep and wake.** Bonds join sleep islands and wake their partners.
- **Preloading.** The scene first settles one house with breaking off
  (`bonds.mjs preloadBonds`) and starts every copy from those poses.

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

- `asset.mjs`: bounds-checked GLB / schema-1 FlatBuffers reader; can be reused
  independently of WebGPU. Exposes full tensors, bond frames/areas/moments,
  composition, strength scales, reinforcement, boundary loops and child mappings.
  Interface material IDs share the material ID space:
  `library.interface_material[id - library.material.length]`; -1 means unspecified.
- `geometry.mjs`: selects a hierarchy level, converts full inertia to a principal
  basis, transforms both render and collision geometry to that basis, and cooks
  baked convex topology into the **packed hull ABI v2** (the same bytes as
  `engine-schema`'s Rust cooker; `hull-golden.json` pins them). Each compound
  child is recentred on its own bounds and placed by the collider's local
  position. CPU work is decoding/conversion, not hull generation.
- `simulation.mjs`: submits Rubble's **core** Slang AVBD kernels using WebGPU with
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

## Verification

```sh
nvm use   # .nvmrc: Node 22.22 (node:module stripTypeScriptTypes needs >= 22.13)
node --test engine/crates/engine-webgpu/fracture/asset.test.mjs engine/crates/engine-webgpu/fracture/convex.test.mjs engine/crates/engine-webgpu/fracture/simulation.test.mjs
cargo test --manifest-path engine/Cargo.toml -p engine-oracles --test convex_slang_cpu
```

`simulation.test.mjs` drives the browser host with a mock WebGPU device against
the exported WGSL (no GPU). `convex_slang_cpu` runs the production Slang convex
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
