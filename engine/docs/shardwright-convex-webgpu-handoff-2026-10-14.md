# Shardwright fractured GLB and Rubble convex Slang/WebGPU handoff

**Date:** 2026-10-14, America/Halifax.  
**Purpose:** Standalone transfer of project context, implementation state, evidence, known failures, and next actions. The incoming worker does not need the preceding chat.  
**Status:** The single-GLB workflow and experimental convex WebGPU path exist and have substantial validation. **The requested high-performance, 10,000+ active-body outcome is not achieved.** One final SAT dispatch optimization compiled successfully but its browser regression result was lost when the preceding work was interrupted.

## 1. User objective and scope

The user wants to:

1. Use **Shardwright offline** for slow pre-fracturing and collision-hull cooking.
2. Export a **single `.glb`** containing render geometry and the physics metadata needed by Rubble: hulls, mass/inertia, bonds, hierarchy, and materials. No separate `.fracphys` must be delivered to the browser.
3. Import that already-baked asset quickly into **Rubble's Slang AVBD implementation through WebGPU**. Browser import must not run fracture generation or convex decomposition.
4. Demonstrate the result in a web page with a larger building, a floor placed at the building's base, click-to-shoot sphere projectiles, reset, and separate physics TPS/render FPS/simulation-speed counters.
5. Support many compound convex bodies and hulls with **up to 64 vertices** correctly and efficiently. Repository instructions explicitly require assuming **10,000+ active rigid bodies** when evaluating scale.
6. Preserve openings such as doorways and arches through their constituent collision hulls. A whole compound must not be substituted with a sphere or a single convex enclosure.

The most recent request is to produce this handoff. No commit, PR, merge, or publication was requested. Do not interpret completion of this report as performance acceptance of the implementation.

## 2. Immediate takeover facts

- Rubble checkout: this repository.
- Shardwright checkout: a sibling `ShardWright` checkout (`../ShardWright`).
- Rubble HEAD when this report was prepared: `15ea7b122a2e879dc798dcd2d6abe3ed8ae50de0` (`WIP: harden full-scale AVBD WebGPU scenes and reduce ring cost`).
- Shardwright HEAD: `f10ce21970dba09530b0bd5687885945d81ef370`.
- Other work modified and committed shared Rubble code during this effort. Historical GPU results are from the evolving working tree, **not a clean, isolated benchmark of the current HEAD**. Regenerate all required shaders together before reproducing results.
- Much of this feature is **untracked or uncommitted in both repositories**. A fresh clone at those HEADs alone will not reproduce it. Preserve/transfer the files listed below before cleaning either checkout. This document is an architectural and operational handoff, not a replacement for those source files and binary assets.
- Demo URL used: `http://127.0.0.1:8777/fracture/`. Do not assume its old server or browser session remains alive.
- Last pending command was a browser run of `runConvexGpuTests({scale:10000})` after balancing SAT workgroup dispatch dimensions. At handoff, its exec session was gone and `/tmp/rubble-convex-balanced.json` was absent. **No result can be claimed for that last run.**
- Persistent evidence remains in `engine/crates/engine-webgpu/fracture/convex-validation.json`; its GPU results predate that final dispatch change.
- During preparation of this report, the current **11 Node tests passed again**, and `git diff --check` produced no errors. This does not replace the outstanding GPU rerun.

## 3. Files to preserve and review

Paths below are relative to the indicated repository.

### Shardwright

| File | State at handoff | Purpose |
|---|---|---|
| `crates/frac-io/src/fracture_glb.rs` | Untracked | Read/write the standalone fracture GLB extension |
| `crates/frac-io/src/lib.rs` | Modified | Export `FRACTURE_EXTENSION`, `write_fracture_glb`, `read_fracture_glb` |
| `crates/frac-cli/src/main.rs` | Modified | Add `prefracture export-glb` |
| `crates/frac-io/tests/fracture_glb.rs` | Untracked | Round-trip, invalid input, determinism, meshopt compatibility tests |
| `docs/GLB_FRACTURE.md` | Untracked | Extension and coordinate contract |
| `README.md` | Modified | Export workflow documentation |

### Rubble

The entire `engine/crates/engine-webgpu/fracture/` directory was untracked at handoff.

| File | Purpose |
|---|---|
| `engine/slang/kernels/fracture_step.slang` | Seven opt-in fracture collision compute entries; includes the existing AVBD world kernels |
| `engine/slang/THIRD_PARTY_LICENSES.txt` | MIT attribution for the three-avbd-derived algorithms; present in the current checkout |
| `engine/crates/engine-webgpu/src/bin/fracture_export.rs` | Compile seven Slang entries to WGSL and parse/validate each with Naga |
| `fracture/asset.mjs` | CPU-only GLB and schema-1 FlatBuffers reader |
| `fracture/geometry.mjs` | Hierarchy selection, full-inertia conversion, hull packing, exact bounds, grid sizing |
| `fracture/simulation.mjs` | WebGPU buffer layout, pipeline loading, AVBD scheduling, diagnostics, projectiles |
| `fracture/render.mjs` | WebGPU rendering from GPU body poses |
| `fracture/main.mjs` | Loading/UI/frame scheduling and test hooks |
| `fracture/index.html` | Demo page and controls |
| `fracture/rates.mjs` | TPS/FPS/simulation-speed accounting |
| `fracture/asset.test.mjs` | Six CPU tests, including real bundled assets and invalid finer levels |
| `fracture/convex.test.mjs` | Five CPU geometry/packing tests |
| `fracture/regression.mjs` | 840-step arch projectile/floor/wake/reset browser regression |
| `fracture/convex-regression.mjs` | GPU-vs-CPU SAT conformance, aperture/reuse checks, 10,000-body workload |
| `fracture/convex-validation.json` | Saved GPU evidence, reproduced numerically in this report |
| `fracture/README.md` | Run instructions, architecture, scope and benchmark caveats |
| `fracture/assets/masonry-arch.glb` | 4,894,676-byte standalone asset |
| `fracture/assets/veneer-house.glb` | 54,089,616-byte standalone asset |

`fracture/...` in that table means `engine/crates/engine-webgpu/fracture/...`.

Generated files under `engine/crates/engine-webgpu/generated/wgsl/` are ignored by Git. Regenerate them; do not pair a new importer with old fracture WGSL. Other dirty Rubble files, unrelated local screenshots, plans and benchmark outputs existed. Do not reset or sweep them into this feature indiscriminately.

## 4. Asset format and Shardwright exporter

### Chosen representation

The format remains a normal **glTF 2.0 GLB** with standard JSON and BIN chunks. A project/vendor extension, **`SHARDWRIGHT_fracture` version 1**, points to the original FlatBuffers FRAC physics payload in a standard `bufferView` and embeds the resolved physical material library in JSON.

This is **not a ratified Khronos extension or a registered vendor prefix**. It avoids delivering a separate sidecar, but the physics payload inside the GLB is still the existing FRAC schema; it is not an interoperable industry-standard destruction schema. OpenUSD/Houdini-specific interchange was not implemented.

Illustrative root JSON (bufferView index is an example):

```json
{
  "extensionsUsed": ["SHARDWRIGHT_fracture"],
  "extensions": {
    "SHARDWRIGHT_fracture": {
      "version": 1,
      "encoding": "flatbuffers",
      "physicsSchemaVersion": 1,
      "bufferView": 12,
      "coordinates": {"units": "meters", "up": "Y", "handedness": "right"},
      "materialLibrary": {
        "library_id": "...",
        "library_version": "...",
        "material": [],
        "interface_material": []
      }
    }
  }
}
```

The payload is an eight-byte-aligned slice of embedded buffer 0, has no graphics buffer target, and retains the original FRAC file identifier and bytes. The extension is in `extensionsUsed`, not `extensionsRequired`: ordinary viewers can render without interpreting physics. A destruction consumer must explicitly require and validate the extension. Generic GLB optimizers must preserve/remap its bufferView and the embedded fragment-to-node references.

### Exporter behavior

Rust APIs:

```rust
frac_io::write_fracture_glb(render_glb: &[u8], fracphys: &[u8], materials: &serde_json::Value)
    -> Result<Vec<u8>, IoError>;
frac_io::read_fracture_glb(glb: &[u8])
    -> Result<(Vec<u8>, serde_json::Value), IoError>;
```

Use `frac_io::fb` for the complete typed FlatBuffers schema; `read_physics` alone is a summary reader.

The writer validates the FRAC/schema header, material library identity/version, fragment/node matching, fragment material IDs and bond interface-material references. It rejects an already-bundled input, mismatched render/physics pairings, external buffer/image dependencies and invalid lengths. It preserves the original physics bytes, retains rendering resources, aligns and appends the payload, and emits the extension. URI-less meshopt virtual fallback buffers are supported by the exporter. The browser demo requires decoded render buffers.

Packaging **does not generate fractures**. Export command from the Shardwright root:

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

`--physics` defaults to the input path with `.fracphys` extension. Use `--materials path/to/materials.toml` if the bake used a custom library. `--uncompressed` expands meshopt for this demo's minimal accessor reader. The generic exporter can preserve meshopt for capable consumers.

### Data available to a future stress solver

The JS reader exposes:

- Fragments: ID/level/parent/children, mass, COM, full row-major 3×3 inertia, material IDs/fractions, hull ranges, volume, interior area, role/component/cell information, render node and particle-candidate flag.
- Hulls: asset-space vertices and polygon-face topology.
- Bonds: endpoints, level, area, centroid, normal, planarity, tangent frame, area moments (`iUU`, `iVV`, `iUV`, polar moment), extents, distances to bodies, interface composition, reinforcement count/area/direction, strength scale, parent/children, boundary loops, spawn ranges, anchor flag and interface count.
- Asset metadata: tool/schema version, seed, settings hash, library identity, hierarchy counts and mappings, boundary loops and spawn data.
- Full resolved physical material and interface-material library.

Interface material IDs share the material ID space: `interface_material[id - material.length]`; `-1` means unspecified. Missing optional properties do not mean zero strength. A bond's strength scale multiplies a constitutive capacity; it is not itself a breaking force in newtons.

### Coordinate/mass contract

Units are SI, right-handed, Y-up; quaternions are `(x,y,z,w)`. Hull vertices are in asset space. Inertia is about the fragment COM. Render meshes are COM-relative, with parent-relative node translations. `Fragment.gltf_node` addresses a node whose `extras.fragment_id` must match.

The importer diagonalizes the full inertia tensor into a principal basis and transforms render and collision geometry consistently into that basis. It retains off-diagonal inertia information through this change of coordinates. Compound mass/inertia belong to the rigid body, not separately to each child hull. Select one hierarchy frontier; this demo selects a single whole level and never instantiates a fragment together with its ancestors.

## 5. Demo content and limitations

| Asset/level | Fragments | Hulls/bonds where measured | Status |
|---|---:|---|---|
| Masonry arch level 0 | 23 | Asset has 751 fragments and 2,821 bonds across all levels | Impact regression passed |
| Arch levels 1 / 2 / 3 | 165 / 171 / 392 | CPU import checks cover all four arch levels | Not all individually stress-tested on GPU |
| Veneer house level 0 | 131 | 163 hulls, 454 bonds | Default, usable |
| Veneer house level 1 | 567 | 2,253 hulls, 2,247 bonds | Usable; heavily profiled |
| Veneer house level 2 | 3,158 | First rejected hull: fragment 1401, hull 3590 | Source zero-volume hull; needs recooking |
| Veneer house level 3 | 5,905 | First rejected hull: fragment 5162, hull 8777 | Source zero-volume hull; needs recooking |

The house is **`veneer_house`**, not the separate Shardwright `two_storey_building` benchmark. Source bake locations are `out/structs/building_lite/veneer_house/` and `out/structs/masonry_arch/`. Bundled exports use seed 1337 and built-in material library 1.0.0, with all hierarchy levels in one GLB.

The floor is at the minimum hull Y after asset placement. Fragments start asleep at baked poses and wake from projectile/neighbor contact. There are 16 projectile slots; click shoots, drag orbits, wheel zooms, and reset restores poses and ammunition. Sphere trajectories compensate for gravity toward approximate clicked scene depth. Collision is discrete, not CCD.

**Bonds are inspection lines, not physical constraints.** There is no stress solver, bond failure, damage accumulation, runtime fracture refinement or mixed hierarchy frontier support. Initial sleeping placement is not a solved bonded equilibrium, and anchor bonds do not become fixed world constraints. Thus unbonded building parts can collapse once awakened. Do not diagnose that behavior as failure to import bond data: the metadata exists, but the constraint model has not been implemented.

Other limits: global demo contact friction 0.5; glTF base-color shading; no texture support or arbitrary edited node transforms; sparse/meshopt accessors are unsupported by the browser reader. Decode with `--uncompressed`. Rendering reads GPU body transforms directly; it is not CPU triangle rendering or a JavaScript physics solver.

## 6. Why the earlier house was slow

The former broadphase derived a radius from hull vertices relative to the **compound body's COM**. Child hulls offset from that COM acquired large enclosing bounds. This caused many unnecessary candidate pairs and expensive convex SAT work. The radius was an acceleration proxy, not evidence that all final hull contacts were solved as sphere contacts.

The new radius is half the child hull's local AABB diagonal and is used **only to choose a grid size level**. Exact transformed vertices form the collision AABB. An additional oriented box is a conservative rejection proxy. Surviving candidates still use actual convex geometry. Compound openings survive because each child hull is tested independently.

Historical diagnostics:

- User screenshot: approximately **3.4 TPS, 0.8 FPS, 0.03× simulation speed**, with about 294.7 ms/tick and 1193.2 ms/frame.
- A previous impact measurement was about **195,122 candidates and 457 ms/tick**; isolated rendering was roughly 10 ms in that earlier investigation. These were different samples, not a controlled paired benchmark.
- Static CPU bound analysis counted **216,718 old sphere-derived AABB overlaps versus 3,815 vertex-AABB overlaps**: about 98.2% fewer overlapping proxy pairs. This is an offline pair-count comparison, **not a 98.2% runtime speedup**.
- Earlier frame scheduling could await four physics ticks serially before rendering. That amplified low graphics FPS even when drawing itself was not the dominant cost.

The importer-side COM-radius choice explains loose bounds; it does not imply the entire Shardwright export was bad. Separately, finer house levels really do contain zero-volume source hulls. Those are an asset-cooking issue and are now reported explicitly.

## 7. three-avbd analysis and port

Reference checkout: `inspiration/pinned/three-avbd`, pinned to **`2bb1a318e9aad03640b4bce5b5d222484f0925f7`**, upstream `sbobyn/three-avbd`. Relevant source files:

- `src/avbd3d/gpu/wgsl-collision.ts`: cheap rejection, grid organization and fine/coarse ownership.
- `src/avbd3d/gpu/wgsl-hull.ts`: hull SAT, adjacent-face Gauss-map pruning, face selection/clipping and contact reduction.
- `src/avbd3d/gpu/solver.ts` and radius/size-classification helpers: static upload and host layout.

The reference combines sphere rejection, oriented bounds, face-axis rejection and exact convex collision. It prunes edge-cross-edge axes using adjacent face normals, selects an incident face touching the deepest support vertex, and retains deep/spread contacts. Its fine/coarse ownership includes fine sleeping shapes querying coarse movers.

This implementation adapts those principles to **collider-level compound children**. It is not a literal copy of the reference's count/scan/scatter broadphase: Rubble's fracture path uses a hierarchical hashed linked list to reuse existing storage. The existing Slang AVBD solve/persistence remains shared. MIT attribution to Steven Bobyn (2026) is in the shader header and `engine/slang/THIRD_PARTY_LICENSES.txt`; retain it.

## 8. Current collision implementation

### Import and packed hull ABI

Limits are **64 vertices, 128 faces and 192 edges per hull**. A 64-vertex triangulated convex polyhedron can have 124 faces and 186 edges; the tests cover this case.

All hull records are padded `float3` values with a **16-byte storage stride**. Offsets/counts stored as floats must remain exact integers; the importer rejects an arena reaching `2^24` records.

For base record `B`, vertex count `V`, face count `F`, edge count `E`:

| Record range | Contents |
|---|---|
| `B` | `[V, F, signed E]`; negative E disables adjacency-based pruning |
| `B+1 .. B+V` | Original vertex coordinates/order in the body principal frame |
| Next F records | Outward face normals |
| Next F records | `[absolute face-loop start, face-loop length, 0]` |
| Next 2E records | Per edge: `[vertexA, vertexB, faceA]`, then `[faceB, 0, 0]` |
| Next 3 records | Interior point, local AABB minimum, local AABB maximum |
| Remaining records | Face-loop vertex coordinates |

Metadata starts at `B+1+V+2F+2E`. Collider parameters are `[B,V,F,radius]`, with radius used for grid classification only. There is no GLB extension-version change for this internal runtime ABI change.

The importer welds exact duplicate positions for adjacency only, normalizes coplanar face patches with planar convex loops, orients normals, and tests for nonzero 3D volume. It does not move original vertices or generate a new 3D decomposition. A closed, consistently oriented edge map enables the fast Gauss-map path.

When normalized faces cannot be formed (`Degenerate convex face`), it retries original face topology and uses exhaustive projections. Uncertified adjacency is marked negative; missing adjacency never enables Gauss-map pruning. This fallback is for imperfect baked face soups, **not a general concave-mesh collision algorithm or proof that arbitrary malformed topology is correct**. Tightening validation further remains worthwhile.

### Bounds and broadphase

1. `fracture_build_aabbs` transforms every convex vertex to compute exact world AABBs, padded by 0.0001 m, and caches the transformed local AABB as an OBB. Sleeping caches are initialized correctly on the first GPU step.
2. Each child collider is inserted once into a power-of-two size level of a hierarchical hash grid. Width covers its proxy diameter; the base is based on the median diameter with f32 padding.
3. Queries visit 27 cells at their own and coarser levels. Fine fixed/sleeping colliders also query coarse movers; omitting this loses contacts.
4. Stored exact cell coordinates and level are checked after hash lookup, avoiding hash-collision false ownership.
5. Same-body pairs are excluded. Pair ownership suppresses duplicates.
6. Exact AABB and six OBB face-axis rejection tests precede convex SAT. Six axes are a conservative early-out, not a replacement for full SAT.
7. Planes use a small explicit host index list. There is no fixed-capacity large-object list or silent 4,096-entry chain cutoff. Capacity/chain failures set overflow flags.

Hash heads reuse `gConstraintIds`, links reuse `gColorLists`; later CSR/color phases reuse those buffers after broadphase is finished. Link storage is sized for collider count as well as body/color storage, important for compounds.

### Narrowphase and contacts

- One **64-lane workgroup per convex pair** caches world vertices, face normals and edge adjacency in about 13.5 KiB shared memory.
- Lanes distribute face axes and edge-cross-edge tests. Closed adjacency permits Gauss-map rejection and support-edge separation; uncertified hulls use full vertex projections.
- Winner reduction has deterministic tie keys and prefers a face patch unless the edge result is materially better.
- SAT and clipping are separate pipelines so large clipping scratch arrays do not burden SAT.
- Face clipping has 128 temporary vertex slots to accommodate intersected high-vertex polygons. It chooses up to eight deepest/spread contacts with stable feature IDs.
- Incident faces must touch the deepest support vertex. If numerical/topology imperfections empty the clipped patch after SAT reports overlap, a support-point contact is retained, following the reference fallback.
- Plane contact selection also retains deepest/spread vertices instead of the first eight.
- Sphere–hull contact uses geometric closest-triangle/face information; sphere–plane and sphere–sphere paths remain available for projectiles.
- `fracture_prepare_candidates` attempts shared, generation-gated contact reuse before SAT/clipping. Cache keys use collider pairs, so children of a compound cannot alias one another.
- Hit states are `feature.w = 0` invalid, `1` valid SAT result, `2` reused manifold.

WebGPU uniformity is essential: storage-dependent early returns before workgroup barriers were rejected by browser validation. Current SAT makes every lane reach every barrier, including padding and non-convex/reused pairs; only valid pairs produce SAT output.

### Last change: balanced dispatch, runtime verification outstanding

Previously, more than 32,768 candidates launched rows of width 32,768. Around 35,900 pairs therefore launched 65,536 workgroups, with almost half doing padding work/barriers.

Current argument generation and SAT indexing both use:

```text
rows  = max(1, ceil(candidateCount / 32768))
width = ceil(candidateCount / rows)
index = group.x + group.y * width
```

This removes nearly all padding while retaining bounded dispatch dimensions. All seven entries compiled and passed Naga validation after this change. The final browser rerun was interrupted and its output is unavailable. Test counts around 0, 1, 32,768, 32,769, 65,535 and 65,536 explicitly, in addition to the standard scale test, before calling this verified.

## 9. WebGPU host integration and hazards

`fracture_export` generates these seven entry points:

1. `fracture_prepare_candidates`
2. `fracture_write_sat_args`
3. `fracture_sat`
4. `fracture_narrowphase`
5. `fracture_broadphase_query`
6. `fracture_build_aabbs`
7. `fracture_broadphase_insert`

The fracture host substitutes these for the corresponding broad/narrowphase entries and uses the shared kernels for contact persistence, CSR adjacency, graph coloring, prediction, primal/dual AVBD and sleeping.

Important ABI/scheduling details:

- `gFracHits` is binding **27**, 32 bytes per candidate; host allocates 28 buffer slots.
- Binding 24 is the shared world-OBB cache.
- Indirect slot **69** is fracture SAT workgroups; slots 64–68 retain shared work-count meanings. The buffer has at least 70 slots.
- **Integration collision:** newer shared code also uses indirect slot 69 for incremental coloring when `HOST_S(26)` is enabled. The fracture host leaves that mode off. Do not enable it without allocating distinct slots and updating both code paths.
- `HOST_S(17)` represents initialized/step generation, `(18)` grid level count, `(19)` plane count. Under the WebGPU path these are immutable host parameters, separate from scratch atomic counters that reuse indices 18–20 for contact-reuse diagnostics. Do not assume a native host has the same separation.
- The plane index list uses `gEvents`; joint-enabled code may need that storage. **This demo has no joints.** A generic joint/stress integration must resolve this ownership explicitly.
- Previous-contact hash sizing is capacity-based and consistent across host/clear passes. The previous GPU contact count is copied before counters are cleared. CPU `last.contact` is stale inside a batched submission and must not determine previous contact count.
- First initialization/reset forces GPU OBB/AABB construction even when initial fragments are sleeping; reset preserves immutable hull/plane data and invalidates cached initialization.
- Contact/candidate capacities are powers of two, initially about 32/64 entries per collider respectively, with a 4,096 minimum. These are storage budgets, not guaranteed sufficient for every pile. Buffer sizes are checked against adapter limits; overflow is surfaced rather than silently accepted.
- Coloring runs 32 rounds and up to 32 color dispatches per solver iteration. Empty colors use GPU-generated indirect counts; no stale CPU color estimate is used. This is a performance investigation target, not a reason to truncate colors.
- Solver defaults: 120 Hz, eight AVBD iterations in the demo, eight sleep-island propagation rounds. A `sleepRounds` config member exists but the current local scheduling variable is hard-coded to eight; audit before exposing tuning controls.
- Native Metal/CUDA host integration of the new hull ABI/binding/entries is **not implemented or validated** by this work. Slang source portability is not equivalent to a working native host integration.

### Frame scheduling and performance counters

`simulation.step(count)` accepts 1–4 ticks, submits ordered physics work and reads counters once per batch. The page chooses a smaller batch for slower ticks, queues rendering before awaiting physics readback, and skips physics submission for a fully sleeping world. A projectile invalidates that idle shortcut.

This is bounded batching, **not a fully independent render loop with asynchronous statistics rings**. GPU queue work and awaited readback can still depress FPS.

- TPS: completed physics ticks per wall-clock second, target 120.
- FPS: submitted/rendered demo frames per wall-clock second, not isolated GPU raster time or guaranteed display presentation rate.
- Simulation speed: TPS / 120.
- HUD ms/tick: CPU encode/submit plus GPU queue/readback wait, divided by batch size.
- Frame ms: interval between demo frames, including physics blocking.
- Hidden-tab/loading periods are excluded from the displayed rates.

Diagnostics:

- `inspectCollisions()` runs actual bounds/broadphase/SAT/contact kernels and reads pairs, bounds and contacts, without integrating poses. It disables reuse for that inspection. It still writes collision buffers/counters: use a dedicated/reset simulation before subsequent physics comparisons.
- `profileStep()` uses optional `timestamp-query`, **advances one real step**, and reports bounds, broadphase, narrowphase, persistence, topology, solve and sleep phases. It is not a passive getter. Narrowphase currently aggregates reuse/hash setup, SAT and clipping rather than timing each separately.
- `snapshot()` reads body state. Avoid it in the timed inner loop unless readback cost is intentionally part of the metric.

## 10. Validation evidence

### CPU and shader validation

Current Node command passed **11/11**, rerun during report preparation:

- Real asset hierarchy/material/bond preservation.
- Asymmetric inertia reconstruction and render/collision frame agreement.
- All arch levels and parent/child exclusion.
- Corrupt/plain/unsupported GLB rejection.
- Floor placement, sleeping initial state and inactive projectile slots.
- House levels 0 and 1, plus expected zero-volume rejection for levels 2 and 3.
- Closed 64-vertex prism adjacency.
- Compound child offset 50 m from body origin, rotated bounds, and appropriately small child classification radius.
- Duplicate face/vertex normalization without changing original vertex records.
- Flat-hull rejection.
- 64-vertex bipyramid retaining 124 faces and 186 edges.

Slang exporter runs compiled all seven entries and parsed/validated their generated WGSL through Naga. Browser validation also passed for the preceding cooperative/reuse implementation. The balanced dispatch change has compile/Naga evidence but lacks its final browser result.

Shardwright has three integration tests: byte-preserving/deterministic standalone round trip, rejection of wrong materials/model/corrupt data, and preservation/decompression of self-contained meshopt rendering. The round-trip optionally invokes a Khronos validator when installed. These tests were part of the earlier exporter work; **they were not rerun while preparing this handoff, and no fresh Rust test log is bundled here**. Run the command below before acceptance.

### Saved GPU conformance results

Adapter reported **vendor `apple`, architecture `metal-3`**. Exact chip/browser version was not saved with these samples; do not infer a specific chip solely from the repository's toolchain document. Tests were not hardware-isolated.

| Fixture | CPU SAT overlap pairs | GPU candidates | GPU contact points | GPU manifolds | Overflow |
|---|---:|---:|---:|---:|---:|
| 520 bodies, 8-vertex hulls | 127 | 130 | 501 | 127 | 0 |
| 64 bodies, 64-vertex prisms | 15 | 15 | 84 | 15 | 0 |
| 24 bodies, 64-vertex/124-face bipyramids | 6 | 6 | 23 | 6 | 0 |

Fixtures include arbitrary orientations, mixed sizes, static/sleeping/awake bodies and compound children. GPU bounds were compared with CPU transformed-vertex bounds. Candidate duplicates and missed/false contact pairs were checked against exhaustive CPU face/edge SAT. The inspection path does not build an awake list, so `awake:0` in these diagnostic records is not evidence that every fixture body was sleeping.

Additional saved passes:

- **Compound aperture:** a sphere inside a four-hull frame's opening does not contact it; a sphere on the jamb does.
- **Batched contact reuse:** eight individual steps versus two four-step batches; maximum tested pose-component difference **0**.
- **Arch impact/reset:** 120 idle + 720 impact steps = **840 steps**; peak 103 contact points and 24 awake bodies; maximum fragment displacement 21.7374 m; minimum hull Y **−0.0020093 m**, within the test's −0.025 m floor tolerance; finite poses/unit orientations, projectile floor support, wake-up and exact reset/ammunition checks passed. Final overflow 0. Large displacement is not proof of realistic bond mechanics, which are absent.

### Dense 10,000-body scale workload

The saved test uses 10,000 64-vertex prisms in a slightly overlapping/rotated 3D grid, gravity zero, eight AVBD iterations, **contact reuse disabled**, one warm-up and five timed steps. All 10,000 bodies stayed active. Snapshot readback is excluded from the saved 281.78 ms timing. This is a deliberately dense contact workload, not a representative performance guarantee for every scene.

| Metric | Saved value |
|---|---:|
| Wall-clock mean | **281.78 ms/tick** |
| Equivalent throughput | About **3.55 TPS** |
| Equivalent speed at 120 Hz | About **0.030×** |
| Active bodies | 10,000 |
| Candidate pairs | 35,923 |
| Contact points | 112,107 |
| Manifolds | 26,837 |
| Colors | 6 |
| Overflow | 0 |

A separate subsequent timestamped step (a different simulation state) recorded:

| Phase | GPU milliseconds |
|---|---:|
| Bounds | 0.1311 |
| Broadphase | 0.2621 |
| Narrowphase aggregate | **95.6170** |
| Persistence | 0.3277 |
| Topology | 2.5559 |
| AVBD solve aggregate | **106.0372** |
| Sleeping | 14.6145 |
| Sum of measured GPU phases | **219.5456** |
| Whole profiled call, wall clock | 236.1 |

That profiled step had 35,849 candidates, 112,040 contacts, 26,856 manifolds, 10,000 awake bodies and zero overflow. CPU-SAT oracle comparison was run on the smaller conformance fixtures; the scale run checked finite state, capacities and activity, **not exhaustive oracle comparison of every scale pair**. Only six ordinary ticks plus the profiling step were tested; this is not a long-duration stability qualification.

### House level-1 impact samples

After a projectile impact and wake propagation, five GPU timestamp samples contained about **7,600–7,700 candidates**, **7,700 contact points**, 568 awake bodies (567 fragments plus projectile), seven colors and zero overflow.

- Narrowphase: roughly **0.8–15.1 ms**.
- Solve: roughly **86.4–136.3 ms**.
- Whole profiled calls: roughly **140.8–218.9 ms**.
- The solver dominated these samples after contact reuse and tighter bounds.

These house samples are described here and in the demo README; their temporary raw output is not available at handoff. Their large variability means they should not be used as a precise speedup claim. Reproduce with fixed impact, step index, warm-up, visibility and competing GPU load controlled.

## 11. Experiments: what worked, what failed, and why

| Action/experiment | Outcome and interpretation |
|---|---|
| Single GLB containing original FRAC bytes and material library | Working packaging/import design; no browser fracture bake needed. It is custom metadata, not a universal destruction format. |
| Exact child bounds and size hierarchy | Removed COM-offset inflation and greatly reduced unnecessary house candidates. Offline overlap-count reduction was about 98.2%; real timings need separate measurement. |
| Closed adjacency plus Gauss-map pruning | Avoids most irrelevant edge axes; exhaustive CPU SAT conformance passed on valid fixtures. |
| Scalar optimized SAT | Earlier house sample around 198 ms/tick versus a historical 457 ms; earlier scale sample around 315 ms. Conditions differed; not controlled speedup ratios. |
| First split SAT implementation | One recorded 10k sample worsened to about **792 ms/tick**. Splitting alone did not remove repeated geometry work/private scratch costs; exact regression cause was not isolated. |
| Cooperative cached 64-lane SAT | Improved a subsequent 10k sample to about **200.62 ms/tick**. Later samples were about 303 and 282 ms. Useful direction, but contention and changes prevent declaring a stable multiplier. |
| Storage-dependent early return before SAT barriers | Browser rejected WGSL uniformity. Fixed by having all lanes execute every barrier and guarding actual work/output instead. |
| Deepest/spread contact patches and incident face touching deepest vertex | Better matches reference manifold construction for irregular/high-vertex hulls; tests passed. Avoid first-eight-vertices contact reduction. |
| Support fallback for an empty clipped patch | Retains collision after SAT establishes overlap; final arch and conformance runs passed before the dispatch-only change. |
| Contact reuse and corrected batch previous-contact bookkeeping | Arch regression and exact batched pose comparison passed; substantial collision-cost reduction in house profiles. Shared solver remains costly. |
| Increasing planar-loop area tolerance during normalization | Increased uncertifiable topology rather than improving it; reverted to the current small threshold. |
| Strict validation of every original raw face as a support plane | Rejected usable house level 1 at fragment 264 / hull 735 because its imperfect original face soup triggered the test. This added raw-face check was reverted. Fast-path normalized hull validation remains; fallback uses exhaustive projections. The source topology needs offline quality work rather than unsupported fast-path certification. |
| Zero-volume detection | Correctly identifies invalid finer-level source hulls. Those levels now fail visibly rather than inventing thickness or passing degenerate geometry into dynamics. |
| Failed level selection preserving current scene | Browser check confirmed level 1 remained intact after level 2 failed. The selector is restored. This specifically covers geometry-build failure; full transactional handling of all later GPU allocation failures could be improved. |
| 1–4 tick batching, draw queued before readback, sleeping fast path | Reduces serial waits and wasted sleeping-world physics. Does not remove queue/readback coupling or prove real-time rendering under load. |
| Balanced 2D SAT dispatch | Removes known padding overhead in code and compiles/validates. Final runtime result missing; first priority is rerun, not assumed success. |

## 12. Build, run and reproduce

### Required environment

Follow repository `AGENTS.md`: **every shell issuing Rust commands must export**:

```sh
export RUSTUP_TOOLCHAIN=stable
export WGPU_BACKEND=vulkan
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
```

These are repository instructions, including Cloud-specific Vulkan configuration. Browser JavaScript chooses its own WebGPU adapter; those variables do not make Chrome use Vulkan on macOS. Actual saved browser tests reported Apple/Metal. Software adapters/SwiftShader are suitable for correctness fallback, not comparable hardware performance results.

The documented Slang compiler is 2026.19, typically `~/.local/bin/slangc` symlinked to `~/.local/opt/slang-2026.19/bin/slangc`. Confirm `slangc -v` and PATH. Rust is stable. Node must support the modern JS used here, including `toReversed`.

### Export shaders and serve

From Rubble root:

```sh
export RUSTUP_TOOLCHAIN=stable
export WGPU_BACKEND=vulkan
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
cargo run --manifest-path engine/Cargo.toml -p engine-webgpu --bin wgsl_export
cargo run --manifest-path engine/Cargo.toml -p engine-webgpu --bin fracture_export
python3 -m http.server 8777 --bind 127.0.0.1 --directory engine/crates/engine-webgpu
```

Open `http://127.0.0.1:8777/fracture/` in a WebGPU-enabled browser. Avoid a stale cache after regeneration; reload with cache disabled. Confirm the adapter is hardware-backed and that there are no WebGPU validation/device-loss errors.

### CPU tests

```sh
node --test \
  engine/crates/engine-webgpu/fracture/asset.test.mjs \
  engine/crates/engine-webgpu/fracture/convex.test.mjs
git diff --check
```

### Shardwright exporter tests

From Shardwright root:

```sh
export RUSTUP_TOOLCHAIN=stable
export WGPU_BACKEND=vulkan
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
cargo test -p frac-io --test fracture_glb
```

If checkout write permissions are restricted, use an authorized writable Cargo target directory or obtain the required filesystem access. Do not claim a test ran merely because its source is present.

### Browser conformance and scale

In the fracture page console after modules are available:

```js
const convex = await import('./convex-regression.mjs');
const result = await convex.runConvexGpuTests({scale: 10000});
console.log(JSON.stringify(result, null, 2));
```

Use `{scale:0}` to run conformance/aperture/batch checks without the scale scene. The helper creates its own WebGPU device/simulations and destroys them. Save full results, hardware/browser/toolchain versions and source revision/hashes durably, not only under `/tmp`.

Arch regression:

```js
const demo = window.__FRACTURE_DEMO__;
await demo.load('arch');
const result = await (await import('./regression.mjs')).runFractureRegression(demo);
console.log(JSON.stringify(result, null, 2));
```

The hook exposes `ready`, `asset`, `scene`, `simulation`, `renderer`, `rates`, `load`, `selectLevel`, `shootRay`, `stepFrames`, `snapshot`, and `reset`. `stepFrames` and `shootRay` stop the automatic physics loop before proceeding.

A reproducible starting point for a house profile:

```js
const demo = window.__FRACTURE_DEMO__;
await demo.load('house');
await demo.selectLevel(1);
if (demo.scene.selected.length !== 567) throw new Error('House level 1 did not load');
const target = demo.scene.bodies[20].p;
await demo.shootRay([target[0], target[1], target[2] + 3], [0, 0, -1]);
await demo.stepFrames(72); // Let the projectile reach the building and wake pieces.
const profiles = [];
for (let i = 0; i < 5; i++) profiles.push(await demo.simulation.profileStep());
console.log(JSON.stringify(profiles, null, 2));
```

Exact step indices in earlier house samples included intervening profiling, so the snippet is a reproducible new baseline, not a promise to duplicate historical counts exactly. `selectLevel` catches load failures into the UI status; verify the selected scene after awaiting it.

For browser automation, the prior worker used the local `cdp` skill and `browser-harness-js` against an isolated Chrome tab. Do not reuse historical tab IDs, websocket URLs or exec-session IDs; discover current state. A reload invalidates old JS objects. Poll `window.__FRACTURE_DEMO__?.ready` afresh with a timeout rather than capturing an undefined `const demo` and waiting forever on it. Screenshots and temporary benchmark files from the earlier browser session may no longer exist.

## 13. Recommended next work, in order

### A. Preserve and establish a trustworthy baseline

1. Transfer/commit the feature files from both checkouts, including binary assets, license and generated-file regeneration instructions. Review unrelated dirty files separately.
2. Regenerate standard and fracture WGSL against the same current shared source. Shared AVBD changes have landed since earlier measurements.
3. Run 11 CPU tests, three Shardwright exporter tests and the browser conformance/arch suites.
4. Verify the last balanced SAT dispatch change, including boundary counts and nonmultiple row sizes. If it fails, revert only that small dimensions/index calculation to the last runtime-tested mapping while debugging.
5. Record a new isolated benchmark with exact chip, browser, adapter, Slang/Rust versions, commit plus dirty diff/source hashes, scene seed, iterations, reuse mode, active count, candidates, contacts, manifolds, colors, overflow, GPU timestamps and wall latency. Warm pipelines first. Keep other heavy GPU demos/tests stopped where authorized.

### B. Optimize both collision and shared solver, without changing correctness silently

1. Split narrowphase timestamp phases into hash/reuse, SAT and clipping. The current 95.6 ms aggregate does not prove which dominates the dense scale case.
2. Measure reuse hit/miss/generation-rejection counters; do not judge collision cost from sleeping worlds. Keep both a no-reuse worst case and a moving/reuse-enabled representative pile.
3. Profile clipping scratch pressure, duplicate parallel edge axes, geometry loads, shared-memory occupancy and irregular-hull fallback frequency. Consider cooperative clipping or precomputed edge/plane descriptors only with CPU-oracle and contact-quality tests.
4. Profile the **106 ms shared solve** and **14.6 ms sleeping** sample. Check dispatch overhead versus arithmetic/contact loads, color scheduling and repeated graph/sleep passes. Coordinate with work already optimizing shared AVBD. Do not reduce iterations/colors or skip valid constraints and describe it as an equivalent speedup.
5. Consider compacting only unreused convex pairs into a separate SAT work list if measurements show invalid/reused groups are material; compare extra compaction cost against saved work.
6. Evaluate asynchronous readback rings and bounded in-flight physics/render scheduling. Keep TPS based on completed work and simulation time honest; merely submitting more work is not throughput improvement.
7. Include many hulls per compound, not just one hull per body. Scale storage by collider and contact counts as well as rigid-body count.

At 120 Hz the physics budget is 8.33 ms/tick. The saved 281.78 ms scale result is roughly 34 times that budget. A headline "supports 10k" must distinguish finite/capacity correctness from real-time performance. Agree on target hardware and representative contact density before promising a frame rate.

### C. Repair offline hull quality in Shardwright

1. Investigate the level-2/3 zero-volume hulls at the IDs above and report their source fragments and bake settings.
2. Validate volume, convex support planes, closed reverse edge adjacency, winding, duplicate/T-junction/coplanar faces and 64-vertex limits in the offline cooker/export pipeline.
3. Prefer recooking valid collision pieces offline. Do not silently inflate flat geometry, merge across openings, or replace compounds with body-centered spheres.
4. For level 1, investigate the imperfect face soups requiring exhaustive fallback. Earlier counting found about 156/2,253 uncertified hulls, including 82 normalization failures labeled `Degenerate convex face`; recount on current code/assets before relying on those totals.
5. Add source-quality and malformed-topology fixtures, especially thin/sliver hulls, large coordinate offsets, tiny/large scales, near-coplanar edges, face-face/edge-edge/grazing cases and high-velocity impacts.

### D. Broaden correctness and backend integration

- Run long-duration 10k+ active-body tests; the saved six-step workload is insufficient for stability qualification.
- Add many children per body, dense compounds, explicit overflow tests, static/sleeping ownership across every grid size transition, hash collisions and rotated hole traversal.
- Check manifold normal/depth/support and resting stability, not only contact-pair overlap. Randomized SAT fixtures are useful but not an exhaustive proof.
- Validate sphere–hull edge/vertex/interior cases and CCD/tunneling behavior if projectiles require higher speeds.
- Test other hardware/backends and default WebGPU limits. Optional timestamp support must remain optional.
- Before native integration, extend Metal/CUDA host bindings, hull upload ABI, indirect slots and scratch ownership. Resolve plane/joint storage and incremental-color slot conflicts explicitly.

### E. Implement destruction semantics as a separate stage

Map embedded bond geometry/materials into a chosen stress/constraint model, including units, reinforcement, anchors, failure criteria and damage accumulation. Only then implement breakable bonds, topology changes and hierarchy activation. Current unbonded sleeping fragments do not meet that goal. Maintain the single-GLB transport contract and avoid conflating asset metadata availability with implemented physics behavior.

## 14. Source fingerprint at report preparation

SHA-256 values identify the uncommitted implementation described here, including balanced SAT dispatch:

```text
08186242314e220cff1f3d9a70e9612630a3074b8401dd9055d32530d98b3fc2  engine/slang/kernels/fracture_step.slang
7b8650ea929fd1cf1fdfa29030708a739b73c1c2610e5644d126afde3abbf6f5  engine/crates/engine-webgpu/fracture/geometry.mjs
d3b66f01a1d6b312936e01a87d8301dea467d82bc50aaa751cb1a130ddc83fa3  engine/crates/engine-webgpu/fracture/simulation.mjs
6b33d23ab2dde6afb770970cd5f4f852263efab003f9e993c6865fbed03e187f  engine/crates/engine-webgpu/fracture/convex-regression.mjs
```

These identify files, not a fully reproducible benchmark environment. The highest-priority unresolved item is a fresh end-to-end GPU run after regeneration against the current shared AVBD source, followed by measured work on collision **and** solver costs.
