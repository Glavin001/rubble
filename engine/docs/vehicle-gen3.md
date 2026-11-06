# gen3: the generated sedan's render model, rigged with crash physics

gen3 replaces gen2's lattice. The approved render model (`gen2/model.rs`, `surface.rs`,
the vehicle art spec) is rigged the way an artist's `.glb` would be. Each render vertex
is bound to the physics by construction: its weights are written with the export and
read verbatim by the loader. The mod keeps gen2's name and folder (`__mods/gen2`,
vehicle `gen2`, `gen2.pc`), so manifests and pages still work.

```bash
cargo run --release -p engine-softbody --example gen2_export     # __mods/gen2: JBeam, gen2.glb, gen2.rskin
cargo run --release -p engine-softbody --example gen3_rest -- 10 # in memory: per-part drift and warp parked
cargo test --release -p engine-softbody --test gen3_rig -- --nocapture
SURFACE_CONTACT=1 cargo run --release -p engine-softbody --example vehicle_compare -- --cars gen2,etkk
# page: softbody.html?solver=explicit&preset=wall56&mod=/__mods/gen2&pc=gen2/gen2.pc&wheels=raycast
```

## Files

| File | What |
|---|---|
| `gen2/rig.rs` | The lattice, placed from the surface's public API (section rays, belt, shoulder, arches, greenhouse corner), the part meshes' extents and the spec's markers. |
| `gen2/gen3.rs` | Which mesh rides which part, which pieces move rigidly, the skins, flexbody rows and deform groups. |
| `flexskin.rs` | Weighted skins: MLS weights, the `.rskin` sidecar, CPU evaluation (`eval`). |
| `gen2/build.rs` | The builder's rules: mass, springs, stability, lanes (shared with gen2). |
| `flexmesh::jbeam_visual_skinned` | The loader: `.glb` meshes and their `.rskin` skins, applied verbatim. |
| `showcase-car-render.mjs` `weightedVertex()` | The GPU twin of `flexskin::eval`. |
| `tests/gen3_rig.rs` | The required checks. |
| `examples/gen3_rest.rs` | Rest drift and warp per part, with `GEN3_MODES` (each closure's lowest free modes), `GEN3_SOFT` (beams the stability bound softened), `GEN3_DUMP`, `GEN3_NODE` and `GEN3_STRAIN`. |

## Decisions

### Binding

- **Weights are moving least squares (affine) over the vertex's own part's structural
  nodes.** No lane nodes, latch hardware or hubs are used. Each node has a C² compact
  kernel, (1 − r²/R²)³, with R = 1.45 × its 5th-neighbour spacing and at least
  0.42 m.
  - The weights form a partition of unity and reproduce any affine motion, so any rigid
    motion is exact.
  - **Choice:** MLS over B-spline patches. Patches need a tensor lattice per part, and
    the body shell has openings. MLS works on any part's node cloud, so the same code
    rigs an artist's `.glb`.
- **Each vertex stores** (node, A, ∂A/∂T, ∂A/∂B) for every node in reach, plus a detail
  offset and the rest normal's coefficients in the frame (U, V, Ŵ).
- **Normals** are F⁻ᵀ applied to the rest normal, recomputed every frame from the
  deformed field.
- **The frame's gradients come from kernels 1.6× wider than the position's.** With the
  position's own kernels, mm differences between close nodes turned normals 2°.
  λ = (5 mm)².
- **Rigid pieces** take one frame per piece, from their centroid's weights: lamps,
  mirrors, handles, wipers, grille, seats, steering wheel, dash, engine block,
  radiator, tyres, rims and discs.
- **Export:** a `.rskin` sidecar, named by a Rubble extension on the JBeam flexbody row
  (`"rubbleSkin"`, with `"pos"` placing the glTF mesh). It references nodes by name.
  152,528 vertices, 3.4 M entries (22.5 a vertex, at most 57), about 55 MB.

### Lattice

- **Members per side:** sill box (outer, top, bottom), floor rail, hinge pillar, A-, B-
  and C-pillar box sections, roof rails, roof bows (with inner partners), quarter grid,
  front rails low and high, aprons, strut tower, rear rails, firewall, cowl, radiator
  support, shelf and tail panel. Bonded windshield and backlight rows join them.
- **Panels have depth.**
  - Doors: an outer 3 × 3 grid on the bottom, shoulder crease and belt; an inner frame
    8 cm in on three rows; a window frame with an inner section, whose legs run down to
    the inner panel.
  - Hood: 5 × 4 outer grid with an inner grid 6 cm under it.
  - Trunk: a deck and its rear face, with inner nodes.
  - Fenders: top, crease and arch rows, plus a front bracket.
  - Bumpers: the cover plus a beam with a back flange.
- **Beams** form a neighbourhood graph within each part: 0.90 m on the body, 0.66–0.85 m
  on closures. None crosses an opening:
  - the door apertures, read from the door meshes;
  - the cabin volume, 8 cm in from the skin;
  - the engine bay top;
  - the trunk opening, read from the lid's mesh.
- **Hardware from markers:**
  - door hinges: 2 groups of 4 beams at 40 kN;
  - latches: couplers at 50 kN (doors), 2 × 30 kN (hood) and 45 kN (trunk);
  - bolts: fender 25 kN, bumper 20 kN, bumper sides 15 kN;
  - window frames seat inboard on the rails (SUPPORT).
- **Lane nodes** (≤ 0.20 m, 0.15 kg) sit only on side-facing edges of the outer skin.
  Each has two backing ties, spread at most 3 to a backing node. They have no
  self-collision: they meet the world, not their own car.
- **Mass** is each part's Accord mass by member kg/m plus the part's render surface,
  each node taking its Voronoi share. The floor is 0.7 × the mean.
  - Strut and shock tops get 6 kg and 4 kg, so the raycast wheels' arm beams stay
    stable.
  - Total 1,450 kg; CG 1,217 mm behind the front axle (band 1,001–1,248).
- **Stability:** ω_max·h ≤ 1.85 (measured, damping included). The softening now hits
  only the beams carrying the mode's strain energy. Before, it hit every beam of the
  nodes that move, which softened the doors to 0.34× of nominal.
- **Yield is the census's:** body 1.15 %, cabin × 1.2, crush zone × 1.0. gen2 needed
  × 4.5 and × 6. The sweep (surface contact on) put × 1.0 / × 1.2 nearest the frontal
  bands and the roof.
- **Surface contact** (spec §38) is on for every measurement below. The off column is
  for reference.

### Deviations from the census

- Node mass p50 is 0.86 kg against BeamNG's 1.2–4 kg. Hood and trunk nodes are
  0.33–0.36 kg, below the census's 0.55–1.6 kg. They have more nodes for depth.
- Body k·L p50 is 5.1e5 after the stability bound, against the census's 0.8–1.1e6.
- Spring ends per node p50: body 22, doors 14–16, fenders 8.

## Required checks (`tests/gen3_rig.rs`, CPU, raycast wheels, surface contact on)

| Check | Result | Required |
|---|---|---|
| Rest fidelity at spawn (vertex / normal) | 0.30 mm / 0.12° | ≤ 1 mm / 1° |
| Rest fidelity after 10 s parked: skin beyond its nodes' own rigid motion | 0.65 mm / 0.41° | ≤ 1 mm / 1° |
| Rest drift, every part, 10 s | ≤ 4.3 mm | ≤ 5 mm |
| Rest: broken / bent | 0 / 0 | 0 / 0 |
| wall56: stretch p99, fold p99 | 1.18, 3.1° | ≤ 1.6, ≤ 31° |
| pole32: stretch p99, fold p99 | 1.06, 1.1° | ≤ 1.6, ≤ 31° |
| Seams opened > 5 cm: wall56 / pole32 | **1,121 / 123 (fails; test ignored)** | 0 |

Rest fidelity after parking is measured against each vertex's own nodes' rigid motion,
weighted as the skin weights them. It counts only error beyond the most any of those
nodes is off that motion: the warp the skin adds. Against the design itself, the parked
mesh is 0.35° off in normal and about 100 mm in position, because the car sits on its
wheels.

## Measured: `vehicle_compare` (CPU, JBeam mode, pressure wheels)

Full tables: `measurements/vehicle-compare-gen3-surface-on.md` and `-off.md`.

| | gen3 on | gen3 off | ETK on | band |
|---|---|---|---|---|
| wall56 dynamic crush (mm) | 594 | 621 | 495 | 620–750 |
| t_m (ms) | 59.5 | 62 | 46.5 | 63–76 |
| peak, rear cabin (g) | 47.0 | 46.5 | 67.6 | 34–47 |
| residual max / avg (mm) | 597 / 335 | 642 / 374 | 230 / 193 | 380–580 / 261–512 |
| Campbell b1 (1/s) | 39.8 | 35.6 | 68.9 | 22.9–39.2 |
| restitution | 0.143 | 0.128 | 0.349 | 0.13–0.20 |
| feet / knees / chest intrusion (mm) | 94 / 60 / 53 | 106 / 50 / 70 | 63 / 130 / 40 | firewall 1–67 |
| side pole 32 km/h 75°: residual / dynamic (mm) | 168 / 301 | 199 / 355 | 101 / 212 | 297–445 residual |
| pole sweep 90°: worst penetration node / beam / triangle (mm) | 16 / 27 / 16 | 50 / 127 / 127 | 2.6 / 127 / 127 | 0 |
| MDB 214 residual side crush (mm) | 303 | 303 | 364 | 126–206 |
| roof, peak within 127 mm / weight | 5.49 | 5.49 | 6.14 | 4.48–5.32 |
| FMVSS 214 door: initial / intermediate / peak (kN) | 8.4 / 31.8 / 117 | same | 5.7 / 10.7 / 38.5 | ≥ 10.0 / 15.6 / 31.1 |

Against gen2 (pressure wheels, flag off): wall residual was 422 mm, roof 1.54× weight,
and MDB intrusion 4 m. Parts sagged 39–183 mm at rest and the mesh stretched p99 5.6
with 76 seams.

## In the browser

`softbody.html?...&preset=wall56&mod=/__mods/gen2` (the GPU solver) draws the skinned
`.glb` through `weightedVertex()`.

- At rest it matches the static model.
- After the wall, the front crushes smoothly with no stretched or torn panels. The hood,
  fender and door shut lines open, as the seam count says.
- A dark slab hangs under each wheel on raycast wheels. The ETK shows the same slab on
  the same page. It is the page's raycast-wheel drawing for imported cars, not gen3's.

## Open

- **Seams in the wall.** The hood against the fenders and grille, the front doors
  against the body and rear doors, and the bumper cover against the lamps open past
  5 cm. `SEAM_DEBUG=1` lists the pairs. The ETK opens none, but it also has 5× fewer
  seam pairs.
- **Too strong in the side tests:**
  - FMVSS 214 door peak 117 kN (band about 31);
  - MDB 303 mm (above 206);
  - pole residual 168 mm (below 297).
- **Frontal:** t_m and dynamic crush are just under their bands, residual max is just
  over, and feet intrusion is 94 mm.
- **Sidecar size:** about 55 MB. Weights could be f16 and nodes u16.
- **The model's marker data:**
  - `bolt_fender_FR_4` is at +X (the left side). The rig takes its side from the part.
  - `latch_door_R*` is mid-door (z −0.93), not on the rear door's shut line. The rig
    moves it to the shut line at the nearest height.
