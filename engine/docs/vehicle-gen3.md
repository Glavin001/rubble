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
| `examples/gen3_rest.rs` | Rest drift and warp per part, with `GEN3_MODES` (each closure's lowest free modes), `GEN3_SOFT` (beams the stability bound softened), `GEN3_DUMP`, `GEN3_NODE`, `GEN3_STRAIN` and `GEN3_KE` (where the kinetic energy is, and how much of it is the car's mean motion). |
| `examples/gen3_side.rs` | The oblique pole at five offsets along the car (−10 to +10 cm) and the MDB, on threads: the pole residual's spread with the pole's place against the lattice. |
| `examples/gen3_probe.rs` | The wall, named nodes every 10 ms; `PROBE_BENT=a,b` lists those nodes' most yielded beams at the end. |

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

## Round 2 (2026-11-06)

### Merges

- `merge/showcase-main` (1613f7d7): centred damping is the BeamNG-profile default, plus
  the energy ledger and surface contact r2.
- The styling branch (6a433a06). The fender-bolt marker is now on its own side, so the
  rig no longer corrects it.
- The rear doors' latch marker is on the glass line (z −0.96, y 0.70). The skin's rear
  edge at that height is about 30 cm further back, so the rig still moves the latch onto
  the edge. The front doors' marker is already on it.

### Stability under centred damping

Centred damping is stable below ω·h = 2 at any damping (§39). The bound is now the
springs' highest mode at 1.92, without counting damping (`Car::centred_damping`).

Measured ω·h is 1.918. Parked 30 s, raycast and pressure wheels: 0 broken, 0 bent;
KE 0.018 J (raycast) and 0.004 J (pressure).

### Sidecar v2

- Weights are quantised: A on a 2⁻¹³ grid, ∂A on a 2⁻⁹/m grid, rounded by largest
  remainder.
  - Σ A = 1 and Σ ∂A = 0 exactly, so every rigid motion stays exact.
  - The offset and normal are measured with the quantised values, so the rest pose stays
    exact.
- The sidecar stores i16 steps, u16 nodes and a u8 count per vertex, so it is lossless
  against the binding (`Skin::packable` checks this).
- Size: 30.0 MB, was 55; gzipped 15.8 MB. Rest fidelity is unchanged.

### CPU and GPU parity

`engine-world-wasm/tests/skin_parity_gpu.rs` runs the renderer's own `weightedVertex()`,
read from `showcase-car-render.mjs`, against `flexskin::eval`.

- Input: 4,000 vertices of a bent, crushed lattice, packed as the page packs them.
- Positions agree within 1.0e-6 m and unit normals within 1.5e-6. This is not
  bit-for-bit: 4 of 4,000 vertices agree bit for bit.

### Seams: what opens

Measured with `SEAM_DEBUG=1` and the `gen3_seams` example (the wall at 32, 40, 48 and
56 km/h, surface contact on).

| km/h | gen3 residual max (mm) | gen3 seams > 1 cm / > 5 cm (of 3,754) | ETK residual max (mm) | ETK seams > 1 cm / > 5 cm (of 785) |
|---|---|---|---|---|
| 32 | 436 | 1,100 / 528 | 111 | 0 / 0 |
| 56 | 629 | 1,624 / 1,153 | 234 | 1 / 0 |

- **The ETK's seams never open, even 1 cm.** It is not only that it crushes less.
- **gen3's seams open in two ways**, about half each:
  - **The parts separate.** The two parts' heaviest nodes move 5–11 cm apart.
    - The hood lifts off the fenders and the grille.
    - The front doors move against the A-pillar and sill and against the rear doors.
    - The bumper cover moves against the fenders.
  - **The skin reaches past its nodes.** The nodes stay within 1 cm but the skins are
    10–17 cm apart: the bumper cover against the lamps, the lamps against the fenders,
    the doors against each other and the doors against the fenders. MLS kernels about
    0.5 m wide cannot follow a fold between two nodes.
- **What I tried, and none of it helped:**
  - Lamps and grille skinned instead of rigid: kept.
  - Lane nodes added to the binding pools: 1,174 seams, and parked fidelity 2.6 mm.
  - Smaller kernels: 983 seams, but parked fidelity 1.25 mm.
  - Grille on the body plus a third bumper-cover row: 1,304.
- **Real cars** keep their shut lines within centimetres, because each closure's edge
  carries a flange and is bolted or stopped every 15–25 cm along it. gen3's closures
  have structural nodes about 0.5 m apart, and their edges are held only at the hinges,
  the latch and the seals.
- **Next:** shut-line edge nodes on both sides of each shut line, every 0.2–0.25 m, as
  structural (not lane) nodes. Each is tied to its own part's edge and to a SUPPORT stop
  against its neighbour, and is part of the binding pool, so a border vertex rides nodes
  within 10 cm. That is a lattice change for the next round.

### Side: the FMVSS 214 door reading

The coordinator's door figure was the FMVSS 214 static door test, read as "too stiff".
That test sets **minimums**: initial ≥ 10.0 kN, intermediate ≥ 15.6 kN, peak
≥ min(2 × curb weight, 31.1 kN). The door therefore passes intermediate and peak and
fails only the initial (9.7 kN), by a little.

The added door intrusion beams change the side results as follows:

| | before | with intrusion beams | band |
|---|---|---|---|
| MDB residual (mm) | 247 | 238 | 126–206 |
| Pole residual / dynamic (mm) | 178 / 301 | 178 / 296 | 297–445 residual |
| FMVSS 214 initial / intermediate / peak (kN) | 9.6 / 37.2 / 142.9 | 9.7 / 38.4 / 157.8 | ≥ 10.0 / 15.6 / 31.1 |

The pole and the MDB pull opposite ways: the pole residual is low (mostly elastic) and
the MDB residual is high. Not resolved this round.

### Frontal after round 2 (surface contact on)

| | gen3 | band |
|---|---|---|
| Dynamic crush (mm) | 583 | 620–750 |
| t_m (ms) | 60 | 63–76 |
| Peak, rear cabin (g) | 46.8 | 34–47 |
| Residual max / avg (mm) | 575 / 326 | 380–580 / 261–512 |
| Campbell b1 (1/s) | 41 | 22.9–39.2 |
| Restitution | 0.141 | 0.13–0.20 |
| Feet intrusion (mm) | 94 | 1–67 |
| Roof, peak / weight | 5.94 | 4.48–5.32 |

### Browser (GPU)

- The wall at 56 km/h draws the skinned car correctly through the crush: the hood folds,
  the windshield cracks, the roof stays.
- The side pole on this page differs from the CPU harness: 0 beams bent at 32 km/h on
  gen3, while the ETK runs normally. Round 3 traced it to the lab's impact detection.

## Round 3 (2026-11-06)

Branch `feat/vehicle-gen3-r3`. Every number below is surface contact on unless marked.

### The page's defects

| Defect | Cause | Fix |
|---|---|---|
| The roof not drawn in the wall56 side view at +76 ms | Physics, not rendering. The greenhouse racked 28 cm forward against the floor (the ETK 7 cm; pitch removed: gen3 18 cm, the ETK under 1.5 cm) and dropped 16 cm, so the door frames hid it. The A-pillar was a two-bay chain through its middle node with no depth against the cabin's inertia and buckled out; its top node coincided with the roof rail's front node and the two were never joined; and the reach graph ran struts from the aprons up to the A-pillar's middle and the windshield. | One node at the header; chord beams hinge pillar to header (outer and inner walls, inside the pillar's section); a hinge-pillar front wall behind the arch (a box about 20 cm deep); no beams from the aprons, rails and cross members to nodes above the belt. The roof is drawn at +76 ms (stills). |
| A magenta stub at the rear | `skin::color` had no colour for `GRAINED` (moulded-in black plastic: the rear diffuser, the front splitter), so it fell to the magenta debug colour. | Black. |
| The side pole reads 0 broken, 0 bent; no impact | The lab's impact was "a node inside a static collider". With surface contact the faces stop a 20 cm pole between nodes, so no node ever enters it: the time readout, slow motion and counts stayed at their pre-impact state. | Points across each collision triangle against the colliders grown by the contact range count as an impact too (`explicit_lab.rs`). The pole now reads an impact at 108 ms (step 216). |
| The trunk lid sits raised | At rest it is on its design (lid top 1.006–1.010 m against the deck sides' 1.003–1.005 m at the same sections; skin at spawn 0.15 mm). In the wall it lifted with the greenhouse's racking; it lifts about 3.5 cm against the tail lamps at 90 ms and returns. | The racking fix above. |

### Shut lines: edge nodes

- Where two parts' meshes meet (boundary vertices of different parts within 1 cm, as
  the harness finds seams), a structural node on each side every 22 cm, 2.5 cm in from
  the edge and 1 cm under the skin; 228 in all. Each is tied into its own panel's frame
  (its nearest structural nodes, an inner one among them, and more until the ties span
  all three directions), chained to its neighbours along the edge, and stopped against
  the node across the line (SUPPORT).
- They are in their part's binding pool, so border vertices ride nodes within about
  10 cm.
- Mass: 0.15 kg each from the part's budget. The panel's sheet stays with the nodes
  that carried it, so adding edge nodes does not lighten them.
- Stability: the bound softens the edge ties and stops (`Role::EdgeTie`, `EdgeStop`)
  before a panel's own springs. Without that, the hood, fenders and doors lost half
  their stiffness (k ratio p50 0.31–0.55). Now p50 is 0.85–1.0 and ω·h is 1.917.
- **Decision:** edge nodes with fixed light mass, not Voronoi-weighted ones. Weighted
  ones took 20 % of a panel's mass and the bound softened the panel.

| | before | edge nodes | edge nodes, round 3 yields |
|---|---|---|---|
| wall56 seams > 5 cm (of pairs) | 934 / 3,697 | 869 / 3,830 | 1,138 / 3,830 |
| pole32 | 168 | 132 | 237 |
| MDB | 304 | 210 | 385 |

**Not met: seams are not like the ETK's, and the seam test stays ignored.** The skins
now follow their edges: for doors against the body, the heaviest nodes' gap is most of
the skin gap. What still opens is the parts themselves separating:

- in the crush zone, the hood against the fenders and grille, and the bumper cover
  against the lamps (heaviest-node gaps 9–22 cm);
- the front doors against the hinge pillars (8–10 cm).

The round-3 yields add about 30 % more opened pairs, because the front now crushes
602 mm instead of 583. The ETK keeps its seams shut while crushing 504 mm dynamic and
234 mm residual; gen3's residual is 583 mm.

### Yields by zone

All are within the census's body band (p10 0.2–0.3 %, p50 1.0–1.3 %, p90 4.9–9.3 %).

| Zone | multiple | ε_y |
|---|---|---|
| Crush zones (rails, cans, aprons, rear rails, their webs) | 0.4 | 0.46 % |
| Footwell (floor, sills, hinge pillars, firewall ahead of the front seats, below the belt) | 3.0 | 4.1 % |
| Greenhouse (above the belt) | 0.45 | 0.62 % |
| Side (below the belt, behind the footwell), doors | 1.0 | 1.38 % / 1.45 % |

`GEN3_CRUSH_Y`, `GEN3_FLOOR_Y`, `GEN3_ROOF_Y`, `GEN3_SIDE_Y` and `GEN3_DOOR_Y`
override them for sweeps.

### Frontal and side after round 3

| | round 2 | round 3 | band |
|---|---|---|---|
| Dynamic crush (mm) | 583 | 602 | 620–750 |
| t_m (ms) | 60 | **70** | 63–76 |
| Peak, rear cabin (g) | 46.8 | 31.1 | 34–47 |
| Residual max / avg (mm) | 575 / 326 | 583 / 375 | 380–580 / 261–512 |
| Campbell b1 (1/s) | 41 | **35.5** | ≤ 39.2 |
| Restitution | 0.141 | 0.206 | 0.13–0.20 |
| Feet intrusion (mm) | 94 | **53** | ≤ 67 |
| Roof, peak / weight | 5.94 | **5.06** | 4.48–5.32 |
| Pole residual / dynamic (mm) | 178 / 296 | 188 / 308 | 297–445 residual |
| MDB residual (mm) | 238 | 209 | 126–206 |
| FMVSS 214 door initial (kN) | 9.7 | 9.1 | ≥ 10.0 |

Surface contact off, for reference: dynamic crush 630 mm, t_m 71.5 ms, b1 35.1 /s,
peak 30.9 g, residual 593 / 380 mm, feet 53 mm, pole 209 mm, MDB 209 mm, roof 5.06,
door 9.1 kN. Full tables: `measurements/vehicle-compare-gen3-surface-on.md` and
`-off.md`.

- **Frontal:** t_m, b1, feet and roof are now in band. Dynamic crush is 18 mm short.
  - The peak fell below its band as the front softened. A weaker crush zone means a
    longer, flatter pulse.
  - Restitution is 0.006 over its band. Residual max is 3 mm over its band.
- **Side: not met.** The pole and the MDB still pull opposite ways.

| Change | Pole residual (mm) | MDB residual (mm) | Door initial (kN) |
|---|---|---|---|
| Side yield × 0.5 | 273 | 274 | 8.9 |
| Side yield × 0.35, doors × 3 | 298 | 403 | 8.7 |
| Side yield × 1.4 | 107 | 181 | 9.1 |
| Doors × 0.5 | 210 | 280 | 7.8 |

  - Every yield that lets the pole crush deeper lets the MDB crush deeper too.
  - The door's initial force does not follow the door's yield (it stayed 7.8–9.1 kN).
  - Separating the two needs a structural answer, not a yield multiple: a sill and
    roof rail whose bending strength exceeds the B-pillar's and the door's local
    plastic hinge.

### Rear-door latch

The marker is 3 cm in from the rear door's shut line at 0.70 m, as the front door's is.

- The rear door's skin edge at that height is z −0.958, and the old marker was
  −0.960, so the marker was on the edge all along.
- The rig had compared it with the door mesh's rearmost point at any height: the
  window frame's corner at −1.44, at the belt. So it moved the latch up to the belt.
- The rig's correction is removed.

### Harness slowdown

- The scenes take under 10 s each. The 3–4 minutes were the page's `.truck` export
  rest run alone, with surface contact on or off: 230 s.
- That export goes non-finite at 21 ms. The rig's round-2 stability bound counts on
  centred damping, which the Rigs of Rods `.truck` rules do not apply. Stepping a NaN
  state costs about 50× a live step.
- The run now stops at the first non-finite node and reports `diverged_s`. The harness
  takes 9 s for gen3 again.
- `COMPARE_TIMES=1` prints each scene's time.
- The surface pass was not the cause and is unchanged.

### Checks (`gen3_rig`)

| Check | Round 3 |
|---|---|
| Rest fidelity at spawn | 0.15 mm / 0.06° |
| Parked 10 s, beyond the nodes' own motion | 0.54 mm / 0.41° |
| Drift, every part | ≤ 1.3 mm |
| Broken / bent at rest | 0 / 0 |
| wall56 stretch / fold p99 | 1.15 / 5.7° |
| pole32 stretch / fold p99 | 1.06 / 1.2° |
| Seams > 5 cm wall / pole | 1,138 / 237 (ignored test) |

Stills (GPU page, gen3 beside the ETK, same preset, camera and time):
`media/gen3-r3/`.

## Round 4 (2026-11-06)

Branch `feat/vehicle-gen3-r4`. Surface contact is on unless marked. Full tables:
`measurements/vehicle-compare-gen3-surface-on.md` and `-off.md`. Stills:
`media/gen3-r4/`.

### Decisions

- **Bumper covers ride clips, not through-bolts** (`Role::Clip`, exported as
  ANISOTROPIC, spec §20).
  - A clip has 5 % of its spring in compression and the bolt's spring and strength
    in tension. A cover pushed onto its beam slides back on it; pulled off, it
    holds.
  - Round 3's centre bolts broke in compression at 3 ms.
  - Round 3's side mounts had taken the hubs and wheel proxies, which are body
    nodes, so the wheels tore the covers' corners off at 11 ms.
  - Mounts now go to structural nodes only. The cover's top edge is clipped to the
    radiator support's upper tie bar (front) or the tail panel (rear). The front
    cover's side ends are clipped to the fender's front edge.
- **Fender bolt rows.**
  - Each fender node on the hood's shut line is bolted to the nearest apron or
    cowl node. Each rear-edge node is bolted to the nearest hinge-pillar node.
  - That is 11 bolts a side. Each is one M6 8.8 bolt at its ultimate tensile load:
    20.1 mm² × 800 MPa = 16 kN (ISO 898-1).
- **The hood's hardware was already complete.**
  - The hinges (cowl and apron) and the two latch couplers were in place.
  - The strikers already sit on the upper and lower tie bars, the rail tips and the
    apron ends. That is the latch support bracket.
  - No hood hardware breaks at 56 km/h. The peak hinge load is 34.5 kN of 160 kN.
- **The hood's seams open because it tents.**
  - Its middle rows rise 25–40 cm against the dropping body. The 1.35 m panel,
    shortened about 24 cm, folds once.
  - Six rows instead of four (`GEN3_HOOD_ROWS`) did not change this (983 against
    971 seams), so it stays at four.
- **The A-pillar's chords are hot-stamped:** 3 × the greenhouse's yield. Boron
  steel is 1,000–1,200 MPa; the greenhouse's yield stands for 300–400 MPa.
  - In the wall, the chords had shortened 6–9 cm at 4.4–6.7 kN.
  - The pillar's middle had bowed 19 cm off its chord.
  - The front door's widest gap to the A-pillar is now 131 mm (was 223).
- **The side structure** (step 2):
  - **Sill stations sit at the floor's cross members:**
    - the toe board;
    - the footwell's rear (the front seat's front mounts, 0.53 m ahead of the
      B-pillar);
    - the B-pillar;
    - the seat's rear mounts (0.35 m behind it);
    - the kick-up.
  - **Cross members and floor pan.** The cross members run sill to floor rail to
    tunnel at every station but the B-pillar's, at 3 × the census yield
    (`CROSS_MEMBER_YIELD`). The floor pan between them and its diagonals are at
    0.3 × (`FLOOR_PAN_YIELD`): 0.7 mm sheet, which buckles in plane. The footwell's
    floor keeps its own yield.
  - **The B-pillar is tailored.** Above 0.6 m it is 3 × the greenhouse's yield
    (hot-stamped). Below 0.6 m it is 0.5 × the side's (the soft zone).
  - **The door intrusion beam is a two-chord truss.**
    - Its inner chord is the middle row, 8 cm in. Its outer chord is 2 cm under the
      skin. Each has five stations: the three columns and the midpoints between
      them.
    - Round 3's beam was a straight chain of nodes. Such a chain carries no load
      across the door until it has sagged into a catenary, so its yield never
      mattered: × 0.5 to × 4 changed nothing.
    - As a 6 cm truss, its bending capacity is chord yield × depth: 0.029 ×
      0.9 MN × 0.06 m = 1.6 kN·m. A 32 × 2 mm UHSS tube's is 2.2 kN·m.
    - **The FMVSS 214 door's initial resistance rose from 9.1 to 10.8 kN.**
- **The powertrain sits where the 2011 Accord's does.**
  - Its front face is 0.39 m ahead of the front axle and its rear face 0.12 m
    behind it (DOT HS 811 666 fig. 87). Round 3 had the rear face at the axle.
  - The block's stops on the dash panel engage after the measured gap: the dash
    line is 0.27–0.30 m behind the axle, less the rear face, less 3 cm of manifold
    shield.
  - The mounts are four break groups, one per block corner, at 60 kN a beam. They
    hold. Making them tear at 20–30 kN raised restitution to 0.22 and added no
    crush.
  - Mass is 1,450 kg. The CG is 1,216 mm behind the front axle (band 1,001–1,248)
    and 595 mm up. The front axle carries 57.1 % (was 57.3 %).
- **Zone yields**, calibrated on the bands, each within the census body band:
  - **Crush zone: 0.28** (was 0.4).
  - **Greenhouse: 0.38** (was 0.45).
  - **The suspension towers' beams stay at 0.4.** At 0.28, two of them (rail to
    strut tower) yielded 1 mm under the parked car.
- **Lane-node ties.** A lane node whose ties lie in one plane now gets a backing tie
  across it. In the 0.65 m sill bay, one lane had its chain, its opposite corner and
  its one backing node in a plane, and it sagged 29 mm in 10 s.

### Required numbers (surface contact on)

| Measure | Round 3 | Round 4 | Band |
|---|---|---|---|
| Frontal dynamic crush (mm) | 602 | **626** | 620–750 ✓ |
| t_m (ms) | 70 | 71 | 63–76 ✓ |
| Peak, rear cabin (g) | 31.1 | 29.3 | 34–47 ✗ |
| Residual max / avg (mm) | 583 / 375 | **543** / 334 | 380–580 ✓ |
| Campbell b1 (1/s) | 35.5 | 39.9 | ≤ 39.2 ✗ (by 0.7) |
| Restitution | 0.206 | 0.212 | ≤ 0.20 ✗ |
| Feet intrusion (mm) | 53 | 60 | ≤ 67 ✓ |
| Roof / weight | 5.06 | 5.32 | 4.48–5.32 ✓ (at the edge) |
| Side pole residual (mm) | 188 | 153 (mean of 5 offsets: 173) | 297–445 ✗ |
| MDB residual (mm) | 209 | **191** | 126–206 ✓ |
| FMVSS 214 door: initial / intermediate / peak (kN) | 9.1 / 32.3 / 113 | **10.8** / 28.5 / 97 | ≥ 10.0 / 15.6 / 28.5 ✓ |
| Seams > 5 cm: wall / pole / MDB | 1,138 / 237 / 385 | 934 / 316 / 227 | ETK 0 / 0 / 0 ✗ |

**Surface contact off:**
- wall: dynamic crush 653 mm, t_m 73 ms, peak 29.5 g, residual 538 / 334 mm,
  b1 39.9, restitution 0.212, feet 61 mm, 1,150 seams;
- **pole: residual 255 mm, dynamic 353 mm;**
- MDB 191 mm, roof 5.32, door 10.8 kN.

**The ETK at the same settings (on):** dynamic crush 504 mm, peak 68 g, residual
234 mm, b1 67, restitution 0.36, pole 117 mm, MDB 244 mm, door 6.5 kN, roof 6.02.

### Seams at equal crush

`gen3_seams`, the wall at five speeds:

| km/h | gen3 dynamic / residual (mm) | gen3 seams > 5 cm (of 3,760) | ETK dynamic / residual (mm) | ETK seams > 5 cm (of 785) |
|---|---|---|---|---|
| 32 | 334 / 270 | 591 | 347 / 111 | 0 |
| 40 | 440 / 406 | 706 | 400 / 151 | 0 |
| 48 | 526 / 446 | 998 | 456 / 197 | 0 |
| 56 | 626 / 543 | 934 | 504 / 234 | 0 |
| 64 | 717 / 552 | 1,020 | 553 / 269 | 0 |

**Not met.**
- At equal dynamic crush, gen3 at 48 km/h opens 998 pairs and the ETK at 64 km/h
  opens none (both 526–553 mm).
- Normalised, gen3 opens 1.5–1.8 pairs per mm of dynamic crush; the ETK opens 0.

What still opens at 56 km/h (`SEAM_DEBUG`, the widest node pairs):

| Pair | Pairs > 5 cm | Widest node gap | Cause |
|---|---|---|---|
| hood / fenders | 163 | hd_o24–feL_23, 310 mm | The hood's middle rows tent up 25–40 cm while the fender stays bolted to the apron. All 22 row bolts hold (≤ 11 kN): the hardware holds and the panel folds. |
| bumper cover / headlamps | 145 | 341–354 mm | The cover goes back with the bumper beam and cans; the lamps ride the apron tips. |
| grille / hood | 113 | 286 mm | The hood's front corners lift (the latch is at the middle); the grille rides the cover. |
| fenders / headlamps | 124 | 333 mm | |
| front doors / body | 120 | ap1–door, 131 mm (was 223) | The hinge pillar's top is pushed 8 cm back against the A-pillar's top, and the door goes with its hinges. |
| hood / body | 95 | 38 mm (was 104) | Skin only: node gaps ≤ 4 cm. |

**Hardware peak loads in the wall** (`HW_DEBUG`):

| Hardware | Peak load | Strength |
|---|---|---|
| Hood hinges | 30–35 kN | 160 kN |
| Fender marker bolts | ≤ 30 kN | |
| Fender row bolts | ≤ 11 kN | 16 kN |
| Front door hinges | ≤ 27 kN | 160 kN |
| Engine mounts | 48–80 kN a group | |

Only the front cover's clips break, as the cover is crushed onto the wall:
- the side clips: 67 kN over 8 beams at 22 ms;
- one top-corner clip: 11 kN at 36 ms.

### Side: what limits the pole

- **The pole's energy goes into the floor, and the car slides along the pole.**
  - At 32 km/h the floor pan and its sill joints take 8.6 kJ of the 20.8 kJ of
    plastic work. The B-pillar takes 0.3 kJ.
  - With surface contact the car slides 0.7–1.3 m along the 75° pole. With node
    contact it slides 0.13 m and the residual is 255 mm.
  - **The ETK shows the same.** Its residual runs 22–423 mm across ±10 cm of pole
    offset: 423 mm where it does not slide (58 mm), 117 mm where it slides 1.7 m.
- **Every yield that let the pole crush deeper also let the MDB crush deeper.**
  These are the levers tried, each as a `GEN3_*` variable on `gen3_side` (five pole
  offsets):

  | Lever | Pole mean (mm) | MDB (mm) |
  |---|---|---|
  | Baseline (old intrusion beam) | 136 | 203 |
  | Floor longitudinals × 0.5 (`GEN3_LONG_Y`) | 175 | 206 |
  | Floor longitudinals × 0.3 | 180 | 222 |
  | Side zone × 0.5 | 220 | 255 |
  | Sill section × 0.3 | 151 | 199 |
  | B-pillar lower × 0.25 | 138 | 192 |
  | Door above the crease × 0.3 | 149 | 300 |
  | Cross members × 6 | 166–172 | 210–254 |

  - With the truss door beam, floor longitudinals × 0.5 moved the MDB more than the
    pole (177 / 254 mm), so it is not adopted.
  - The single pole number swings ±40 mm with the pole's place against the lattice.
    Read `gen3_side`'s mean.
- **The harness reports two new measures:**
  - `pole_slide_mm`: the pole's final shift along the car, in the far side's frame.
  - `residual_crush_imprint_mm`: the deepest residual within 0.15 m of the pole's
    path, from first contact to rest.

  For gen3 the imprint equals the line measure, because the deepest crush is at
  first contact. The band reading is unchanged.

### Rest kinetic energy

Round 3 reported 0.13–0.17 J and suspected the new edge ties or stops. Measured with
`GEN3_KE=1 gen3_rest`, it is not beams ringing:

- **88 % of it is the whole car's mean motion.** The car rolls fore and aft on its
  wheels: about a 3.3 s period on raycast wheels, about 1 s on pressure wheels,
  where the wheels' rotation adds to it.
- **The structure's own kinetic energy is unchanged by the edge nodes:** 0.019 J
  with them against 0.021 J without (raycast, 10 s).

**Round 4 on raycast wheels: 0.064 J at 10 s**, 0 broken, 0 bent. The lane fix
above removed one sagging lane's motion.

On pressure wheels it is 1.10 J at 10 s, decaying from 5.6 J at 7 s.
- 0.54 J of it is the car's mean motion.
- Most of the rest is the hubs and rims turning with it.
- The structure carries under 0.15 J.

This is a parked car on free wheels, with no parking brake in the scene. It is not
fixed here.

### Checks (`gen3_rig`)

| Check | Round 4 | Required |
|---|---|---|
| Rest fidelity at spawn | 0.16 mm / 0.07° | ≤ 1 mm / 1° |
| Parked 10 s, beyond the nodes' own motion | 0.52 mm / **0.96°** | ≤ 1 mm / 1° (close) |
| Drift, every part | ≤ 2.1 mm | ≤ 5 mm |
| Broken / bent at rest | 0 / 0 | 0 / 0 |
| wall56 stretch / fold p99 | 1.22 / 5.2° | ≤ 1.6 / ≤ 31° |
| pole32 stretch / fold p99 | 1.11 / 1.0° | ≤ 1.6 / ≤ 31° |
| Seams > 5 cm, wall / pole | 1,040 / 309 (test ignored) | 0 |

ω·h is 1.917 (bound 1.92).

### On the GPU page

`media/gen3-r4/` shows gen3 beside the ETK: same preset, camera and step, surface
contact on, raycast wheels.
- **wall56, side view:** at impact, +60 ms and settled (+1.5 s).
- **pole29, threeFront view:** +60 ms and settled.
- **mdb214:** +60 ms and settled.

At rest the page shows 0 broken and 0 bent.

**wall56:**
- The hood folds up at its middle and the fender front tears at the arch.
- The bumper cover hangs from its last clips.
- The doors stay on their hinges; the front door shows a gap at the A-pillar's
  middle.
- The page breaks 9 beams, the bumper cover's clips. The CPU harness breaks 8–9.

**pole29:** The pole stands at the B-pillar between the doors, and neither door
leaves its hinges.

**mdb214:**
- The page counts 221 broken beams by +1.5 s, 0 at +60 ms.
- The count covers both vehicles. The ETK's is 111 by the same time.
- The CPU MDB, whose barrier is `crash_rigs::mdb214_jbeam`, breaks no gen3 beam.

### Open

- **The pulse's peak (29 g against 34–47) and restitution (0.21 against ≤ 0.20).**
  - The force rises to about 430 kN within 10 ms and stays flat.
  - A late peak needs the block to stack on the dash. At the Accord's measured gap,
    the block moves only 5 cm toward the firewall before the wall stops it, so the
    stop never engages.
- **The pole residual.** See above: the car slides along the pole under surface
  contact. The contact's friction against a cylinder is a question for the solver,
  not the rig.
- **Seams.**
  - In the crush zone they come from the hood tenting and the cover and lamps
    parting.
  - At the doors they come from the hinge pillar moving back.
- **Frontal chest intrusion.** The truss door beam raised it in some configurations:
  60–194 mm across the sweep, 61 mm in the final build. The measure is noisy and has
  no band.

## Round 5 (2026-11-06)

Branch `feat/vehicle-gen3-r5`, merged with `feat/surface-contact-perf` (908e1187:
surface friction's true stick). Surface contact is on unless marked. Full tables:
`measurements/vehicle-compare-gen3-surface-on.md` and `-off.md`. Stills:
`media/gen3-r5/`.

### Decisions

- **The ETK's seams are not shut: our seam metric could not see them.**
  - The seam count pairs *boundary* vertices (open mesh edges) of two pieces
    within 1 cm. The ETK's panels are hemmed, closed meshes: its hood, fender,
    door and lamp lines have no open edge. Its 785 pairs all lie inside one group
    (body and carpet, dash and gauges, quarter panels), whose sides share their
    binding nodes. Its 0 is the metric, not the car.
  - Paired over **every vertex within 1.5 cm** (`shutline_*`, a new harness row),
    the ETK opens 796 pairs past 5 cm at 56 km/h, gen3 2,715.
  - **The ETK's hood tents too:** it rises 440 mm (gen3 683 mm) and its
    hood/fender line opens 509 mm, every pair past 5 cm at the end.
  - Its flexbodies never share nodes across a seam (0 shared group nodes on every
    pair below). Shared seam nodes are not how it keeps lines shut.
- **What keeps the ETK's other lines shut is SUPPORT beams between parts**,
  measured in its JBeam: the bumper fascia to the fender fronts and the hood; the
  headlamp (its own 6-node part, bolted to the radiator support) to the fascia, the
  hood and the fender; the doors to the hinge pillar. They are compression-only, so a
  part driven back pushes its neighbour back instead of sliding past it, and the two
  can still part.
- **gen3 takes the idiom: push links** (`push_links`, `Role::Bearing`): from each
  front-end part's nodes ahead of the front axle (cover, fenders, hood) to the
  nearest node of the part behind it (fenders, hood, body), 4–30 cm back and within
  35° of the car's axis. No tension, so a part whose hardware breaks still separates.
  - In round 4 the cover's corner dived 187 mm and went 522 mm back, while the lamp
    at the apron tip went 225 mm back and the fender front rose 360 mm.
  - Now the cover rises with its neighbours: hood/grille node gap 266 → 88 mm,
    lamp/fender 332 → 131–226 mm, lamp/cover 345 → 195–310 mm.
  - Shut lines at equal crush (gen3 at 48 km/h, 527 mm; the ETK at 64 km/h,
    550 mm): 2,042 against the ETK's 851 (round 4's model: 2,290).
  - Not taken: shared seam nodes (the ETK does not use them), and a stiffer hood
    inner frame (the ETK's hood tents as much).
- **Stacks:** BOUNDED stops from the block's front nodes to the bumper beam,
  engaging once the cooling module is crushed solid (`COOLING_STACK` 0.08 m, an
  estimate from the parts' depths), and from the front hubs to the hinge pillar's
  base (`WHEEL_STACK` 0.23 m: the rim plus flattened tyre). The dash stops were
  already there.
- **Crush zone yield 0.23** (was 0.28): the push links stiffen the front. The hinge
  pillar's front joints (`hpf`) keep the towers' 0.4: at 0.25 one (hpf0R–ar1R)
  yielded under the parked car.
- **Parking brake** (jbeam-semantics-spec.md §25.1): a pressure wheel row with
  `parkingTorque` and a `nodeArm` gets a Coulomb torque between its hub rings and its
  arm about the axle. Stick removes at most the two bodies' relative spin in one step.
  - CPU and GPU (`EB_AERO` modules), bit-identical: 0.0000 mm in
    `parking_brake_matches_cpu`, applied, and released through
    `ExplicitBeamsGpu::set_parking_brake`. Metal `vehicle_parity` and
    `explicit_parity` pass.
  - Released by default: content without brakes, or with them released, is
    unchanged bit for bit (test).
  - gen3's rear wheels carry 1,240 N·m (FMVSS 135's 20 % grade at the Accord's
    GVWR, × 2).
  - The harness parks every JBeam car 10 s on pressure wheels, applied and
    released.

### Required numbers (surface contact on)

| Measure | Round 4 | Round 5 | Band |
|---|---|---|---|
| Frontal dynamic crush (mm) | 626 | 629 | 620–750 ✓ |
| t_m (ms) | 71 | 72 | 63–76 ✓ |
| Peak, rear cabin (g) | 29.3 | 29.2 | 34–47 ✗ |
| Residual max / avg (mm) | 543 / 334 | 468 / 345 | 380–580 ✓ |
| Campbell b1 (1/s) | 39.9 | **38.6** | ≤ 39.2 ✓ |
| Restitution | 0.212 | 0.202 | ≤ 0.20 ✗ (by 0.002) |
| Feet intrusion (mm) | 60 | 66.8 | ≤ 67 ✓ (at the edge) |
| Roof / weight | 5.32 | 5.22 | 4.48–5.32 ✓ |
| Side pole residual (mm) | 153 | 132 (slide 1.28 m) | 297–445 ✗ |
| MDB residual (mm) | 191 | 193 | 126–206 ✓ |
| FMVSS 214 door: initial / intermediate / peak (kN) | 10.8 / 28.5 / 97 | 10.4 / 28.5 / 94 | ≥ 10.0 / 15.6 / 28.5 ✓ |
| Shut lines > 5 cm, wall (every vertex within 1.5 cm) | 2,200 | 2,715 | ETK 796 |
| Parked 10 s on pressure wheels: KE (J) | 1.10 | **0.003** | ETK 0.084 (released 3.63) |

Chest intrusion rose to 148 mm (61 mm in round 4). It has no band and swung 60–194 mm
across round 4's sweeps.

**Surface contact off:** dynamic crush 669 mm, peak 28.8 g, residual 456 / 334 mm, b1
39.9, restitution 0.23, pole 247 mm.

**The ETK (on):** dynamic crush 501 mm, peak 70 g, residual 233 mm, b1 68,
restitution 0.37, pole 119 mm (slide 1.70 m), MDB 244 mm, roof 6.02, door 6.5 kN.

### The ETK seam evidence (wall 56 km/h, every vertex within 1.5 cm)

The node gap is the separation, beyond rest, of the two vertices' heaviest nodes.
Each gap is the maximum over the crash / the value at 0.5 s. From
`gen3_seam_evidence`.

| Seam | ETK flexbody groups | Shared nodes | ETK pairs | ETK skin gap (mm) | ETK node gap (mm) | gen3 r5 skin gap | gen3 r5 node gap |
|---|---|---|---|---|---|---|---|
| hood / fender | `etkk_hood` / `etkk_fender_L` | 0 | 153 | 509 / 489 | 335 / 323 | 302–434 / 216–291 | 330–478 / 228–325 |
| hood / bumper (gen3: grille) | `etkk_hood` / `etkk_bumper_F` | 0 | 127 | 175 / 157 | 171 / 153 | 166 / 128 | 124 / 88 |
| bumper / headlamp | `etkk_bumper_F` / `etkk_headlight_R` | 0 | 44 (+2 housing) | 48 / 0 (housing 84 / 48) | 14 / 0 (58 / 44) | 270–382 / 242–377 | 210–314 / 195–310 |
| fender / headlamp | `etkk_fender_R` / `etkk_headlight_R` | 0 | 11 | 76 / 51 | 54 / 0.6 | 230–333 / 202–328 | 150–230 / 131–226 |
| door / A-pillar (body) | `etkk_body` / `etkk_door_L` | 0 | 61 | 23 / 10 | 9 / 3 | 175–186 / 107–115 | 172–196 / 131–139 |
| door / fender | `etkk_door_L` / `etkk_fender_L` | 0 | 35 | 37 / 18 | 9 / 5 | 224–244 / 89–106 | 98–118 / 13–17 |

Hood rise in the cabin's frame: the ETK 440 mm, gen3 683 mm.

### Pulse: what limits it

- The stops work when they engage. With the stack at 0.25–0.30 m they engage near
  20–30 ms and the force spikes to 530–660 kN. At the estimated 0.08 m they engage at
  the very end and change nothing.
- **The crush is not progressive.**
  - The block's front is 0.57 m behind the nose (the Accord's place). To close the
    0.31 m to the beam, the cans and rail tips ahead of it must crush first.
  - gen3's front crushes along its whole length at once, so the block meets the beam
    only as the car stops.
- A yield graded from a soft nose to a stiff firewall (`GEN3_GRADE_FRONT/REAR`,
  ungraded by default) raised the mid-crash force but cut the crush (575 mm at
  0.3–3.0) and the residual (b1 51). Not adopted.
- The plateau stays at 330–440 kN. 34 g needs about 485 kN held for 10 ms through
  the 100 Hz filter.

### Side pole after the friction fix

`feat/surface-contact-perf`'s stick changed the harness's 75° pole little:
- gen3: at the centre offset, slide 1.20 → 1.09 m and residual 153 → 137 mm (RoR
  friction, `RUBBLE_SURFACE_FRICTION=ror`, against the stick); 0.86–1.43 m across
  the offsets.
- the ETK: slide 1.66 → 1.70 m, residual 117 → 119 mm.
- Node contact: gen3 slides −0.10 to 1.07 m, mean residual 193 mm.

The friction branch measured its own scene (`surface_pole --scene oblique --cyl`),
where the ETK's slide halved. The harness's pole still slides 1–2 m. The side was not
tuned.

### Chrome

The merged surface pass put barriers under a branch on a storage-loaded flag
(`pairs`), which Chrome's uniformity analysis rejects, so the page did not load. The
flag now goes through workgroup memory (`gsUniformSurf`, read as
`workgroupUniformLoad`), as the other `gsUniform*` flags do.

Since the merge, Chrome takes about 8 minutes to compile the frame module on a first
load (the ETK and gen3 alike, raycast wheels). `crash-lab-record.mjs` now waits up to
15 minutes, and `CRASH_LAB_PROFILE` keeps its Chrome profile (and shader cache).

### On the GPU page

`media/gen3-r5/`: wall56 in the `three` view, gen3 beside the ETK, same preset,
camera and timing (surface contact on, raycast wheels): at impact, +60 ms and settled
(+1.5 s).
- Both hoods fold up at the wall. The ETK's tents the way gen3's does, as measured.
- gen3's bumper cover and lamps stay with the fenders' fronts instead of diving
  under them.
- The pole and MDB stills were not re-recorded: each page load costs the 8-minute
  compile.

### Checks (`gen3_rig`)

| Check | Round 5 | Required |
|---|---|---|
| Rest fidelity at spawn | 0.16 mm / 0.07° | ≤ 1 mm / 1° |
| Parked 10 s, beyond the nodes' own motion | 0.53 mm / **0.98°** | ≤ 1 mm / 1° (close) |
| Drift, every part | ≤ 2.3 mm | ≤ 5 mm |
| Broken / bent at rest | 0 / 0 | 0 / 0 |
| wall56 stretch / fold p99 | 1.27 / 5.2° | ≤ 1.6 / ≤ 31° |
| pole32 stretch / fold p99 | 1.11 / 1.0° | ≤ 1.6 / ≤ 31° |

ω·h is 1.915 (bound 1.92).

### Open

- The pulse's peak (29 g against 34–47) and restitution (0.202 against ≤ 0.20).
- The pole residual: the harness's pole still slides 1–2 m for both cars.
- Shut lines: the doors (the hinge pillar moving back), the lamp corners and the
  hood.

## Round 6 (2026-11-06)

Branch `feat/vehicle-gen3-r6`. Surface contact is on unless marked. Full tables:
`measurements/vehicle-compare-gen3-surface-on.md` and `-off.md`. Stills:
`media/gen3-r6/`.

### Decisions

- **The front is graded by station, from measured data.**
  - **The shape to match.**
    - The 2011 Accord's NCAP pulse (DOT HS 811 666 fig. 55, left rear sill) is
      12 g at 10 ms and 26 g at 20 ms, and peaks at 33 g at 78 ms.
    - The 2012 Camry's NCAP force against crush (CCSA's validation against
      NHTSA test 8545, fig. 10) is about 250 kN to 300 mm. It then rises to
      900 kN at 420 mm as the engine stacks.
    - gen3's force was flat at 330–440 kN from 10 ms to the end.
  - **Rails, aprons and cans** (`RAIL_ZONES`), each zone a gauge and grade:

    | Zone | Gauge, grade | Spring | Yield force |
    |---|---|---|---|
    | Crush cans (triggered) | 1.4 mm, 340 MPa | 0.875 | 0.62 |
    | Tip, bumper to the block's front (the reference) | 1.6 mm, 440 MPa | 1 | 1 |
    | To the strut tower (the gauge step) | 2.0 mm, 440 MPa | 1.25 | 1.45 |
    | Behind the tower, the kick-down under the dash | 2.0 mm, 590 MPa | 1.25 | 1.95 |

    - The grades are the Accord body's (270/340/440/980 MPa, DOT HS 811 666
      fig. 140).
    - The gauges are a mid-size sedan rail's usual range (1.4–2.0 mm), **not
      measured on an Accord**. I found no public gauge table for its rails.
    - A thin-walled section folding progressively carries a mean force ∝
      σ_y·t^(5/3) (Wierzbicki and Abramowicz), and its spring goes as t.
    - So the section (`Beam::section`) scales the spring by t / 1.6. The yield
      strain is scaled by (σ / 440)·(t / 1.6)^(2/3), so the yield force goes as
      σ·t^(5/3).
  - **Grading the rails alone did not move the plateau.** Crush yields from 0.12
    to 0.23 changed the force by only 10 %. `WALL_DEBUG` (plastic work by member
    every 10 ms) found two things carrying the front along its whole length:
    - **The body's shut-line edge ties ahead of the dash.** These are the radiator
      support's, aprons' and lamp surrounds' flanges (ebody–rh, ar–ebody).
      - They took 21 kJ of the 114 kJ of plastic work, half of it by 20 ms.
      - They kept the census sheet yield, 4.3 times the rails'.
      - They now take their station's rail yield.
    - **The towers' yield floor** (round 4's 0.4).
      - It held on every beam with a tower node. That included the radiator
        support's upper tie to the tower (rsu–tw) and the apron tip's (ar3–tw),
        which are the front of the upper load path.
      - The floor now holds only where the beam's middle is within 0.235 m of
        the axle. rh2–tw, at 0.15 m, bent parked when it did not.
  - **Crush yield 0.26** (the tip's).

- **The block's stack now engages.** Round 5's stops never did.
  - **Why:** a stop's closure was its own length's, but the targets were 0.14–0.29
    m to the side of the block's corners. A stop along a diagonal never closes to
    8 cm, so the block passed the beam.
  - **Fix 1:** the closure is along the car (`stack_close`). The stop engages
    when the gap along the car is the stack. The dash stops get the same fix.
  - **Fix 2:** the front bumper beam gets stations on the block's face lines
    (x = ±0.26). The stops go only to beam nodes within 0.12 m of the block's
    line (`STACK_REACH`).
  - **Fix 3:** the block still dived under the beam. It pitches down 6 cm on its
    mounts while the beam rides up 5 cm on the wall by 50 ms.
    - The block's front face is now a collision face (two triangles, part
      engine), and the front beam's nodes collide with it (self-collision).
    - A face meets the beam wherever the two overlap across the car and up the
      block.
  - **Result:** with all three fixes the block meets the beam at 40 ms, and the
    force rises to 500–530 kN at 55 ms.

- **Lane nodes keep their bound in any geometry.**
  - **The failure:** once the stack stops were added, the FMVSS 216 roof run
    diverged (NaN at 58 mm of platen travel).
    - The platen pressed lane node xbody47 (0.15 kg) into its backing and lined
      up its ties, which sum to 3.9e6 N/m. Aligned, that is ω·h 2.6.
    - The power iteration only sees the rest geometry's modes. In round 5's car
      the same node already rattled at 10–170 m/s under the platen.
  - **Fix:** each contact-only node's ties are scaled so that Σ k (1/m +
    1/m_far)·h² ≤ 1.92² (Gershgorin with its ties aligned).
    `GEN3_NO_LANE_BOUND` turns it off.

- **The pole scene is the NCAP oblique pole's.**
  - **Why the two scenes disagreed.**
    - `surface_pole --scene oblique` strikes the car's middle, about at its
      centre of mass, and runs 0.6 s.
    - `vehicle_compare` struck the generic head point, 0.64 of the wheelbase aft
      of the front axle, and ran 1.5 s. That is 0.29 m behind the ETK's centre of
      mass and 0.59 m behind gen3's.
    - The 75° impulse behind the centre of mass yaws the car about the pole. In
      `vehicle_compare`'s scene the ETK slid 0.99 m by 0.6 s and 1.70 m by 1.5 s.
      With the pole moved onto its centre of mass (+0.30 m), it slid 0.03 m.
    - `surface_pole`'s figures (624 → 324 mm across the friction fix) measure the
      centre of mass's move across the approach at 0.6 s. `pole_slide_mm` is the
      pole's place along the car in the far side's frame, a different measure.
    - Pole shape and friction source are not the difference. Both scenes use the
      254 mm cylinder (`--cyl`) and the nodes' own friction.
  - **The impact line now matches NHTSA's tests: `POLE_LINE`, 0.464 of the
    wheelbase aft of the front axle.**
    - On the 2018 Accord in TP-214P (reports v10841 and v10842), the line was
      1,303 and 1,313 mm aft of the axle on a 2,819–2,820 mm wheelbase. That is
      0.10 m behind the test weight's centre of mass.
    - A 2020 Veloster's was at 0.50 of its wheelbase and a 2024 Nautilus's at
      0.45.
    - In all of these tests the deepest crush lies on the line (0–150 mm from
      it): the cars do not slide along the pole. The Accord's residual was
      266–319 mm across the five heights.
  - **The car rides a carriage.**
    - The tests tow the car in on a platform (the MCW report measures heights
      "from top of platform").
    - The wheels do not turn, and the tyres sit on the carriage's low-friction
      pads (`CARRIAGE_MU` 0.1; PTFE on steel is 0.04–0.10).
  - `POLE_OLD=1` keeps round 5's scene.

### Required numbers (surface contact on)

| Measure | Round 5 | Round 6 | Band |
|---|---|---|---|
| Frontal dynamic crush (mm) | 629 | 651 | 620–750 ✓ |
| t_m (ms) | 72 | 72 | 63–76 ✓ |
| Peak, rear cabin (g) | 29.2 | **34.4** | 34–47 ✓ |
| Residual max / avg (mm) | 468 / 345 | 577 / 355 | 380–580 ✓ (at the edge) |
| Campbell b1 (1/s) | 38.6 | 37.5 | ≤ 39.2 ✓ |
| Restitution | 0.202 | **0.193** | ≤ 0.20 ✓ |
| Feet intrusion (mm) | 66.8 | **62.1** | ≤ 67 ✓ |
| Roof / weight | 5.22 | 5.25 | 4.48–5.32 ✓ |
| MDB residual (mm) | 193 | 198 | 126–206 ✓ |
| FMVSS 214 door: initial / intermediate / peak (kN) | 10.4 / 28.5 / 94 | 10.5 / 29.2 / 98 | ≥ 10.0 / 15.6 / 28.5 ✓ |
| Side pole residual / dynamic (mm), the new scene | – | 151 / 290 | 297–445 ✗ |
| Parked 10 s on pressure wheels, brake applied: KE (J) | 0.003 | 0.009 | |

- **Force every 5 ms from contact (kN):** 116, 369, 334, 328, 348, 340, 357, 361,
  348, 354, **507**, 378, 290…
- **The edges.** The peak is 0.4 g inside its band, and the residual max is 3 mm
  inside its band. Across crush yields 0.25 / 0.26 / 0.27 / 0.28:
  - peak 35.7 / 34.4 / 32.3 / 30.2 g;
  - restitution 0.193 / 0.193 / 0.191 / 0.218;
  - residual max 581 / 577 / 425 / 418 mm. It falls either near 430 or near 580
    mm, depending on whether one zone folds, within ±0.01 of the yield.

**Surface contact off:**
- wall: dynamic crush 672 mm, t_m 73.5 ms, peak 31.9 g, residual 454 / 313 mm,
  b1 42.7, restitution 0.225, feet 62 mm;
- pole 250 / 358 mm, but the pole reaches 504 mm past the skin (node contact lets
  it pass between nodes);
- MDB 198 mm.

**The ETK (on):** dynamic crush 501 mm, peak 70 g, residual 233 mm, b1 68,
restitution 0.37, pole 202 / 281 mm on the new scene, MDB 244 mm, roof 6.02, door
6.5 kN. With surface contact off its pole reads 438 mm, with the pole 555 mm past
its skin.

### The pole: not met

On the NCAP geometry, neither car reaches the band with surface contact on: gen3
reads 151 mm and the ETK 202 mm. Across ±10 cm of offset, gen3 reads 33–182 mm and
the ETK 57–260 mm.

- **What happens** (`POLE_TRACE`):
  - The dynamic crush stops at about 290 mm.
  - The car leaves the pole at 2.8 m/s of its 8.9: a restitution of 0.32. It then
    glides off on the carriage, turning slowly.
  - At the deepest crush the structure holds 15.5 kJ elastic of the 57 kJ. The
    largest shares are the front door's latch to striker (0.5 kJ), the seat's
    front cross member from sill to tunnel (0.5 kJ each), and the floor rail and
    sill joints.
  - The plastic work is 30 kJ, a third of it the floor pan's.
- **Yields cannot separate the pole from the MDB** (`gen3_side`, five offsets):

  | Lever | Pole mean (mm) | MDB (mm) |
  |---|---|---|
  | Round 6 | 124 | 198 |
  | Cross members × 1/3 (`GEN3_CM_Y` 1.0) | 131 | 199 |
  | Doors × 0.5 | 125 | 260 |
  | Side zone × 0.5 | 168 | 276 |
  | Side 0.3, doors 0.5, cross members 1.0, intrusion beam 0.5 | 206 | 364 |
  | Side 0.2, doors 0.3, cross members 0.5, intrusion beam 0.3, pan 0.15 | 227 | 512 |

  - The last row puts the MDB at 2.5 times its band. The pole still reads 227 mm,
    and the dynamic crush still stops near 375 mm.
  - So the limit is not the yields. It is how the pole meets the side:
    - With surface contact, the pole pushes whole skin triangles about 0.5 m
      across, so the side answers as a membrane spanning its nodes.
    - With node contact, the pole passes between the nodes instead.
  - Denting the side locally, as a real door dents, needs nodes closer than the
    pole's 254 mm across the door and sill. That is a lattice change for the next
    round and is not done here.

### Checks (`gen3_rig`)

| Check | Round 5 | Round 6 | Required |
|---|---|---|---|
| Rest fidelity at spawn | 0.16 mm / 0.07° | 0.16 mm / 0.07° | ≤ 1 mm / 1° |
| Parked 10 s, beyond the nodes' own motion | 0.53 mm / 0.98° | 0.53 mm / **1.03°** | ≤ 1 mm / 1° ✗ |
| Drift, every part | ≤ 2.3 mm | ≤ 2.1 mm | ≤ 5 mm |
| Broken / bent at rest | 0 / 0 | 0 / 0 | 0 / 0 |
| wall56 stretch / fold p99 | 1.27 / 5.2° | 1.28 / 6.0° | ≤ 1.6 / ≤ 31° |
| pole32 stretch / fold p99 (the new scene) | 1.11 / 1.0° | 1.06 / 1.2° | ≤ 1.6 / ≤ 31° |

ω·h is 1.919 (bound 1.92).

**Parked normal fidelity is at the limit, and its 10 s reading depends on the
phase of a slow bounce.**
- The worst vertex is at the body shell's tail corner.
- Its static warp is 0.99° after 20 s and 0.98° after 30 s, as in round 5.
- At 9.5 / 10 / 10.5 / 12 s it reads 0.94 / 1.03 / 1.04 / 1.00°: the car's slow
  parked bounce on raycast wheels adds about ±0.05°.
- With this round's changes switched off by their variables (all but the stack
  closure), it reads 1.05–1.11° at 9.5–10.5 s. Round 5's 0.98° was one phase of
  the same bounce.
- This round does not change the rear. Giving this check margin needs the tail
  corner's static warp reduced.
- About 0.2° of that warp is the lane nodes'. Without them (`GEN3_NOLANES`) the
  static warp is 0.80° at 30 s. A stricter bound on their ties (ω·h 1.5 or 1.0
  instead of 1.92) does not help: 0.98° at 30 s, and 1.05° and 1.11° at 10 s.
  The mechanism is not found yet.

### On the GPU page

`media/gen3-r6/` shows wall56 in the `three` view, gen3 beside the ETK: the same
preset and camera, both aligned to gen3's impact (570 ms), surface contact on,
raycast wheels. Each car is shown at impact, at +55–58 ms and settled (+1.5 s).
- Chrome's first load took about 6 minutes, with the kept profile.
- gen3's hood folds up at the wall, the front stacks behind it, and the doors stay
  shut on their hinges.
- The ETK's hood and front crumple more locally, and its driver's door springs
  open.

### Open

- The pole residual (151 mm against 297–445): the lattice's spacing against a
  254 mm pole (above).
- Parked normal fidelity: 0.98° static and 1.03° at 10 s, against 1°.
- The residual max (577 mm) and the peak (34.4 g) are each just inside their bands.

## Round 7 (2026-11-06)

Branch `feat/vehicle-gen3-r7`. Surface contact is on unless marked. Full tables:
`measurements/vehicle-compare-gen3-surface-on.md` and `-off.md`. Stills:
`media/gen3-r7/`.

### Decisions (deviations first)

- **The pole is not met, and the round-6 hypothesis is refuted by measurement.** A
  side lattice that folds locally at pole width (below) deepens the pole but takes
  the MDB with it: every lever moved the two together. It ships **opt-in**
  (`GEN3_SIDE_R7=1`), not as the default, because the default must keep the MDB,
  the frontal and the door in band.
- **Parked fidelity: the mechanism is found and fixed at the source.** The
  stability bound's row cap softened the members next to the lane nodes. The fix
  holds behind the rear doors only (where the warp is); everywhere, it moves the
  round-6 calibration out of band. The greenhouse yield is 0.37 (was 0.38) to keep
  the roof in band. `gen3_rig` passes. The 0.9° target is met at 10 s but not at
  every time from 5 to 30 s (below).
- **The pole harness started the pole inside the car on some lines.** It walked out
  from the head point to the first clear spot. When the head point itself was clear
  of nodes, the pole started there, 0.5 m in from the skin. Round 6's 33 mm reading
  at −10 cm was that. The pole now starts past the last node on its line. The
  harness's own pole (offset 0) is unchanged.

### Required numbers (surface contact on)

| Measure | Round 6 | Round 7 | Band |
|---|---|---|---|
| Frontal dynamic crush (mm) | 651 | 655 | 620–750 ✓ |
| t_m (ms) | 72 | 72.5 | 63–76 ✓ |
| Peak, rear cabin (g) | 34.4 | 35.6 | 34–47 ✓ |
| Residual max / avg (mm) | 577 / 355 | 569 / 386 | 380–580 ✓ |
| Campbell b1 (1/s) | 37.5 | 34.5 | ≤ 39.2 ✓ |
| Restitution | 0.193 | 0.190 | ≤ 0.20 ✓ |
| Feet intrusion (mm) | 62.1 | 60.8 | ≤ 67 ✓ |
| Roof / weight | 5.25 | 5.31 | 4.48–5.32 ✓ |
| MDB residual (mm) | 198 | 191 | 126–206 ✓ |
| FMVSS 214 door: initial / intermediate / peak (kN) | 10.5 / 29.2 / 98 | 10.7 / 28.8 / 96 | ≥ 10.0 / 15.6 / 28.5 ✓ |
| Side pole residual / dynamic (mm) | 151 / 290 | 147 / 271 | 297–445 ✗ |
| Parked 10 s: normal fidelity | 1.03° | 0.79° | ≤ 1° ✓ (target 0.9) |

- Mass 1,450 kg, centre of mass 1,216 mm behind the front axle; ω·h 1.919; rest
  drift ≤ 2.4 mm (body); 0 broken, 0 bent.
- **Surface contact off:** wall 672 mm, 73 ms, 32.5 g, residual 469 / 345 mm, b1
  38.6, restitution 0.236; pole 257 / 351 mm with the pole 517 mm past the skin
  (between nodes); MDB 191 mm.
- **The ETK (on):** pole 202 / 281 mm, MDB 244 mm, roof 6.02, door 6.5 kN; unchanged.

### The pole

**Measured with `gen3_side`, five pole offsets (−10 to +10 cm) and the MDB at five
offsets (−10 to +10 cm).** Single readings move ±10–20 mm with no change of physics:
reordering the door's beam list (summation order) moved the MDB 198 → 210 mm. Means
over offsets are the numbers to compare.

| | Pole mean (mm) | MDB mean (mm) |
|---|---|---|
| Round 6 structure, harness fixed | 136 | 202 |
| Round 7 default | 128 | 195 |
| `GEN3_SIDE_R7=1` | 207 | 215 |
| The ETK | 187 | 290 |

**What `GEN3_SIDE_R7` builds** (each part has its own switch inside it):
- Front doors at 0.20 m column pitch (`DOOR_PITCH`; 6 columns, not 3), their graph's
  reach 0.42 m (two bays along the door), not 0.66 m (`GEN3_OLD_DOOR`).
- The intrusion beam's outer chord on the skin, as the skin's middle row
  (`GEN3_NO_SKIN_ROW`). The skin between the bottom row (seated on the sill) and the
  shoulder was one triangle 0.49 m tall. So the pole pushed the sill and the
  shoulder and never the beam between them.
- The sill as its own box (`GEN3_OLD_SILL_BOX`): three chords through stations
  every 0.25 m (`SILL_PITCH`), with its section and wall webs. The body graph no
  longer joins sill nodes 0.5–0.9 m apart. Those joins braced every station against
  its neighbours, so a pole met the whole sill.
- The doors' inner panels bear inward on the hinge pillar, B- or C-pillar and sill
  (SUPPORT, `GEN3_NO_DOOR_STOPS`). The skin seals ran along the car from the
  door's edge to the pillar's face and held nothing inward.
- The window glass carries no collision faces (`GEN3_WINDOW_TRIS` keeps them).
  Tempered glass breaks at a few kN; as three triangles it was a rigid plate.
- Rear doors stay as in round 6 (`GEN3_DENSE_REAR`): densified, the light rear door
  softened under the bound to 1.3 kN yields and the MDB read 290–340 mm on it.

`GEN3_SIDE_R7` in the full harness: pole 216 / 340 mm, MDB 221 mm ✗, peak 34.2 g,
restitution 0.247 ✗, feet 106 mm ✗, door 10.3 / 18.4 / 75 kN.

**No lever separated the pole from the MDB** (`GEN3_SIDE_R7` on; pole and MDB means;
the ranges span this round's variants of the lattice as it was built up, some with the
rear doors dense too):

| Lever | Pole | MDB |
|---|---|---|
| none | 207 | 215 |
| Sill chords × 0.3 (`GEN3_SILL_CHORD_Y`) | 251–293 | 244–261 |
| Sill chords × 2 | 198 | 298 (3 offsets) |
| Doors × 1.5 / × 0.5 | 173 / 238 | 340 / 340 |
| Footwell × 0.2 (`GEN3_FLOOR_Y` 0.6) | 231 | 221 |
| Cross members × 0.5 or × 2 | 214–223 | 220–222 |
| B-pillar lower × 3 or × 5 | unchanged | unchanged |
| Greenhouse × 4 to 5 (`GEN3_ROOF_Y` 1.5–2; out of the roof's band) | 236–264 | 217–252 |
| Door pitch 0.25 m, reach 0.36–0.6 m, sill pitch 0.34 m | 202–228 | 226–247 |

**What limits it** (`POLE_TRACE=5`, `POLE_FORCES`, `POLE_HALF_H`, `MDB_TRACE`):
- **The car is stronger to the pole than to the MDB.** Force from the car's momentum:
  - the pole rises to 300 kN at 25–35 ms (230 mm crush) and the car stops at 62 ms;
  - the MDB peaks at 140–160 kN, about half the honeycomb's full-face crush force
    (290 kN, 49 CFR 587).
  - A real side is the other way round: stiff to a wide load, so the barrier's face
    crushes, and soft to a narrow one.
- **The top of the pole carries a third of it.** A pole cut off at 1.0 m reads
  275–293 mm, at 1.1 m 262 mm, at 1.3 m 206 mm, at full height 210 mm. At the
  pole's line the door's window frame corner (`door_FL_f0`, 1.18 m) and the
  A-pillar's middle (`ap1`, 1.15 m) lie within its radius. They are the hot-stamped
  A-pillar chords and the roof rail, which the roof and the frontal need strong.
- **The station under the pole is braced in every direction.** At 30 ms the most
  loaded beams are the sill's station 1 to the tunnel (the seat's front cross
  member, 22 kN each) and to the floor, firewall and front rail (16–20 kN).
- **The stability bound sets the springs by mass.** At 2 kHz a node's springs sum to
  about m · 3.7·10⁶ N/m. Denser nodes on the same door mass get softer springs (door
  k·L p50 1.6·10⁵ → 8·10⁴ N at 0.20 m pitch) and lower yield forces. So the door's
  local strength falls and the MDB with it, while a strong section stores
  F²/2k elastic: a sill chord at its real section (about 26 kN·m plastic moment) on
  a 3·10⁶ N/m spring would store 12 kJ before it yields.

### Parked fidelity: how lane nodes warp normals

- **Not their skin weight:** lane nodes are outside every mesh's pool, and the frame
  kernel's width (1.3, 1.6, 2.0 × the position's) changes the parked warp by 0.02°.
- **Their ties, through the stability bound's row cap** (`GEN3_TRACE_BEAM=a-b` traces
  one beam's spring):
  - A structural node's lane ties gave with its shut-line edge ties. The edge nodes
    carry the skin.
  - A node over the cap on its members alone scaled them by its whole row, lane ties
    included.
  - So the members next to lane nodes kept 0.19–0.38 of their no-lane springs:
    qt236–ebody93 5.2·10⁵ → 1.0·10⁵ N/m, qs185–qi218 3.9·10⁵ → 1.4·10⁵ N/m at the
    tail. The tail corner sagged parked.
- **The fix** (`Car::lane_ties_first`): only lane ties give first; members take the
  cap on their own row; the power iteration softens lane ties before members.
  - Everywhere, the parked normal reads 0.80–0.88° at every time from 5 to 30 s, but
    the frontal peak falls to 30.5 g and the roof rises to 5.66.
  - So it holds from 0.15 m ahead of the rear door's rear shut line backwards
    (`LANE_TIES_AHEAD`; `GEN3_LANE_TIES_FROM`, `GEN3_LANE_SOFTENS_PANEL`). The
    greenhouse yield 0.37 brings the roof back to 5.31.
- **Parked, the tail corner's normal** (worst vertex: body shell, 0.91, 0.71, −2.38 m):

  | t (s) | 5 | 7.5 | 9.5 | 10 | 12 | 15 | 20 | 30 |
  |---|---|---|---|---|---|---|---|---|
  | Round 6 | 1.11° | 0.98° | 0.94° | 1.03° | 1.00° | 0.97° | 0.99° | 0.98° |
  | Round 7 | 0.73° | 0.97° | 0.99° | 0.79° | 0.73° | 0.91° | 0.97° | 0.81° |

  - The static warp is about 0.85°, but the slow parked bounce is now ±0.13°. The
    0.9° target holds at 10, 12 and 30 s, not at 7.5, 9.5, 15 and 20 s.
  - The vertex error is 1.44–1.48 mm at 5 and 7.5 s, while the car settles. Round 6's
    was 1.19 mm at 5 s. It is 0.45–0.84 mm from 9.5 s on.

### Checks (`gen3_rig`)

| Check | Round 6 | Round 7 | Required |
|---|---|---|---|
| Rest fidelity at spawn | 0.16 mm / 0.07° | 0.16 mm / 0.07° | ≤ 1 mm / 1° |
| Parked 10 s, beyond the nodes' own motion | 0.53 mm / 1.03° ✗ | 0.74 mm / 0.79° | ≤ 1 mm / 1° |
| Drift, every part | ≤ 2.1 mm | ≤ 2.4 mm | ≤ 5 mm |
| Broken / bent at rest | 0 / 0 | 0 / 0 | 0 / 0 |
| wall56 stretch / fold p99 | 1.28 / 6.0° | 1.29 / 6.1° | ≤ 1.6 / ≤ 31° |
| pole32 stretch / fold p99 | 1.06 / 1.2° | 1.06 / 1.1° | ≤ 1.6 / ≤ 31° |

### On the GPU page

`media/gen3-r7/` shows the page's pole (`pole29`) in the `three` view, gen3 beside the
ETK. They use the same preset and camera, with the ETK aligned to gen3's impact
(110 ms). Surface contact is on, on raycast wheels. Each car is shown at impact, at
+53–64 ms and settled.
- At +64 ms gen3's front door is creased in at the pole, and its side glass has
  shattered.
- **No MDB stills.** The `mdb214` page did not start within the recorder's 15 min, twice,
  with the kept profile (`CRASH_LAB_PROFILE`). Each GPU run is held to 20 min.

### Diagnostics added

- `POLE_TRACE=<ms>`: the interval.
- `POLE_FORCES=<ms>`: the most loaded beams then.
- `POLE_WRAP`: the struck nodes outboard of the pole's axis.
- `POLE_HALF_H`: a shorter pole.
- `MDB_TRACE`: car and barrier velocity every 5 ms.
- `rebound_vx`: the car's fastest rebound.
- `scene_mdb_at` and `gen3_side`'s MDB offsets (`SIDE_MDB_SPREAD=wide`).
- `GEN3_RDOOR_Y`: the rear doors' yield.
- `GEN3_SILL_CHORD_Y`, `GEN3_SILL_SEC_Y`: the sill box's yields.
- `GEN3_NO_BODY_LANES`: no lane nodes on the body.

### Open

- **The pole** (147 mm against 297–445). Separating it from the MDB needs the side
  stiff to wide loads and soft to narrow ones. At the explicit bound the lattice
  couples the two. Candidates:
  - heavier structural nodes where the side's strength lives: the intrusion beam has
    no member mass, and the door's mass sits on its skin;
  - a pole line that does not land on the A-pillar's middle: the render model's
    A-pillar may sit further back than the Accord's;
  - the seat and its cross members as the wide load's path.
- **Parked fidelity at every time:** remove the slow parked bounce on raycast wheels.

## Round 8 (2026-11-06)

Branch `feat/vehicle-gen3-r8`. Surface contact is on unless marked. Tables:
`measurements/vehicle-compare-gen3-surface-on.md` and `-off.md` (the default car,
unchanged) and `measurements/vehicle-compare-gen3-r8-cabin.md` (`GEN3_CABIN_R8=1`,
beside the ETK).

### Decisions (deviations first)

- **The pole is not met.** Every lever this round measured left the pole's mean below
  200 mm (band 297–445) and moved the MDB with it. Where the side gives, the floor bends
  with it and springs back (below): the car wraps round the pole.
- **gen3's cabin sat 0.21 m aft of the Accord's. The fix ships opt-in
  (`GEN3_CABIN_R8=1`), not as the default.** It moves the firewall, cowl, windshield,
  A-pillars, front doors and B-pillars 0.21 m forward, and seats the driver on the
  Accord's H-point. With it the frontal falls out of band (restitution 0.264, b1 42.4,
  MDB 238 mm). Without the flag the export and every harness number are round 7's,
  bit for bit.
- **The default car is unchanged.** `gen3_rig` passes; ω·h 1.919; 0 broken, 0 bent at
  rest; 1,450 kg, centre of mass 1,216 mm behind the front axle.
- **Round 7's frontal is on a knife edge.** A 10 nm move of the dash's render boxes (a
  float rounding in the interior's placement) changed the stability bound's softening
  and moved b1 from 34.5 to 39.9 /s, the door's initial force from 10.7 to 9.8 kN and
  the roof from 5.31 to 5.37. The default now keeps round 7's literals exactly.

### The geometry against the Accord

NHTSA's oblique pole test of the 2018 Accord (v10841, data sheets 4, 7, 10 and 11;
v10842 alike) gives these positions, aft of the front axle (axle at 934 mm from the
front, the coordinates' origin):

| | Accord | gen3, round 7 | `GEN3_CABIN_R8` |
|---|---|---|---|
| front door's leading edge | 0.453 (impact line 1,304 − 851) | 0.672 | 0.462 |
| door striker on the B-pillar | 1.515 (H-point + 278, "hip point to striker") | 1.730 | 1.52 |
| B-pillar accelerometers: sill / lower / middle | 1.497 / 1.536 / 1.622 | nodes 1.71–1.80 | 1.50–1.59 |
| A-pillar at the roof rail | 1.15–1.30 (the window-top crush row starts at the line) | 1.495 | 1.285 |
| windshield angle | 30.8° | 29° | 29° |
| driver's H-point | 1.237 (seat track accelerometer at the H-point) | 1.915 (behind the B-pillar) | 1.237 |
| firewall centre | 0.154 (accelerometer); dash line 0.27–0.30 | 0.535 | 0.325 |

- **"A-pillar engagement: no direct engagement"** in both Accord tests. In gen3 the
  A-pillar's middle (`ap1`, 1.09 m aft, 1.16 m up) and the door frame's front corner
  (`door_FL_f0`, 1.22 m aft) lay within the pole's radius: round 7's "top of the pole
  carries a third".
- The render model's seats were 0.68 m aft of the Accord's, behind the B-pillar; the
  front seat's front cross member (`SEAT_CM_F`, 1.23 m) sat right under the pole line.
- `examples/gen3_geom.rs` prints the rig's nodes and every part's extent along the car.
- **What `GEN3_CABIN_R8` moves** (`surface::cabin_fwd`): the windshield's base and top
  and the roof ahead of its crown (`gh_shift`, blending back at z −0.70), the door shut
  lines `DOOR_F` and `DOOR_B` (the rear door's rear edge stays), the cowl and hood's
  rear edge, the firewall (render and rig), the front rails' rearmost station, the
  cabin and engine-bay guards, the window frame's front corner, and the interior from
  the Accord's dummy clearances (chest to steering wheel 349 mm, chest to dash 591 mm):
  dash, wheel, seats on the H-point, the rear seat 0.5 m forward, and the named masses
  (driver, seats, dash, HVAC, windshield, rear seat) with them. The centre of mass moves
  to 1,119 mm behind the axle (band 1,001–1,248; the Accord's test weight 1,213 mm).
- Wheelbase change after the pole, struck / far side: the Accord −54 / +12 mm (its side
  lengths −64 / +12); gen3 round 7 −15 / +38; `GEN3_CABIN_R8` −41 / −28; with
  `GEN3_SIDE_R7` too −75 / +7.

### Required numbers (surface contact on)

| Measure | Round 7 = round 8 default | `GEN3_CABIN_R8` | `GEN3_CABIN_R8` + `GEN3_SIDE_R7` | Band |
|---|---|---|---|---|
| Frontal dynamic crush (mm) | 655 | 643 | 666 | 620–750 |
| t_m (ms) | 72.5 | 70 | 71 | 63–76 |
| Peak, rear cabin (g) | 35.6 | 34.6 | 35.8 | 34–47 |
| Residual max / avg (mm) | 569 / 386 | 407 / 315 | 418 / 309 | 380–580 / 261–512 |
| Campbell b1 (1/s) | 34.5 | **42.4** | **43.2** | ≤ 39.2 |
| Restitution | 0.190 | **0.264** | **0.287** | ≤ 0.20 |
| Feet intrusion (mm) | 60.8 | 30.0 | 56.3 | ≤ 67 |
| Roof / weight | 5.311 | 4.56 | **4.04** | 4.48–5.32 |
| MDB residual (mm), offset 0 / mean of five | 191 / 195 | **238 / 239** | **263 / 270** | 126–206 |
| FMVSS 214 door: initial / intermediate / peak (kN) | 10.7 / 28.8 / 96 | 10.3 / 28.0 / 89 | **9.9** / 19.6 / 69 | ≥ 10.0 / 15.6 / 28.5 |
| Pole residual / dynamic (mm), offset 0 | 147 / 271 | 166 / 329 | 184 / 387 | 297–445 |
| Pole mean of five offsets (mm) | 128 | 136 | 196 | |
| Pole cut at 1.0 m (`POLE_HALF_H=0.5`), mean (mm) | – | – | 263 | |

The ETK on the same scene: pole 202 / 281 mm, MDB 244 mm.

### The pole: what limits it

- **Where the side gives, the car wraps round the pole and springs back as a whole**
  (`SIDE_DEBUG`, new: the
  far side's departure from its own rigid fit, and each side's wheelbase change):

  | | far side off its fit at the deepest crush, max / mean (mm) | at the end (mm) | springback (dynamic − residual, mm) |
  |---|---|---|---|
  | gen3 round 7 (the default) | 84 / 30 | 58 / 21 | 124 (271 − 147) |
  | gen3 `GEN3_CABIN_R8` | 313 / 93 | 164 / 58 | 164 (329 − 166) |
  | gen3 `GEN3_CABIN_R8` + `GEN3_SIDE_R7` | 332 / 126 | 141 / 56 | 203 (387 − 184) |
  | ETK | 78 / 35 | 37 / 11 | 79 (281 − 202) |

  - Round 7's car does not wrap: its far side stays as straight as the ETK's, but its
    side is too strong locally (271 mm dynamic, 300 kN).
  - Once the side gives (the pole line between the seat's cross member and the B-pillar
    with the cabin forward, and R7's side lattice), the floor bends with it instead of
    the side alone crushing.

  - The far side's rear (z −1.27 m: the rear door, the rear wheel) swings 300 mm toward
    the pole while its middle (z −0.35) bows 250 mm outward: the cross members push the
    far side out, and the rear half keeps moving past the pole.
  - Half of that bending is elastic. At the deepest crush the structure holds 11.4 kJ
    elastic, spread over the floor, tunnel, sills and both sides' latches: no member
    carries more than 0.3 kJ.
  - The dynamic crush, 330–390 mm, is about what a 297–445 mm residual needs. The
    residual is short by the springback.
- **The frontal shows the same flexibility.** With the cabin forward the car's centre of
  mass rebounds at 3.7 m/s (0.24) and the rear cabin rings ±0.4 m/s about it, peaking at
  4.1 m/s (0.264): `WALL_DEBUG` now prints both. Round 7's rebound: 2.67 / 3.16 m/s. The
  shorter bay leaves 26 kJ elastic at t_m (round 7 about 17 kJ), spread over the hood,
  the fenders and the floor's joints to the rails.
- **No side lever separated the pole from the MDB** (`GEN3_CABIN_R8` + `GEN3_SIDE_R7`;
  pole and MDB means of five offsets):

  | Lever | Pole | MDB | Also |
  |---|---|---|---|
  | none | 176 | 261 | |
  | the roof rail's and window frame's sections × 0.3 (`GEN3_RAIL_SEC_Y`) | 179 | 274 | roof 3.97 |
  | the same × 0.1, frame × 0.1 (`GEN3_FRAME_SEC_Y`) | 179 | 276 | roof 3.91 |
  | intrusion beam mass 3 kg/m (`GEN3_INTRUSION_KGM`) | 169 | 241 | door 11.9 kN |
  | intrusion 6 kg/m, door frame 2 kg/m (`GEN3_DOORFRAME_KGM`) | 171 | 231 | restitution 0.31 |
  | front sheet yield 1.5 % (`GEN3_FRONT_SHEET_EPS`, the frontal's rebound) | – | – | restitution 0.236, peak 32.9 g |

  (This table's rows ran before the engine-bay guard fix; its "none" reads 196 / 270
  after it.) The upper load path is not the limit once the A-pillar is out of the way:
  making the roof rail's and the window frame's sections crush moved the pole 3 mm and
  took the roof out of its band. Mass on the intrusion beam stiffened the side to both.
- **The seat path was not built.** The Accord's pole line is the seated driver's head
  (1.30 m against the H-point's 1.237): a seat that carries a wide load across the car
  is under the pole too, so it cannot separate the two tests.
- **What would be needed:** a side that gives locally on a floor that does not bend
  with it, as the ETK's (dynamic 281 mm, far side 35 mm mean) does. gen3 has either the
  side too strong (round 7: far side 30 mm, dynamic 271) or the floor too soft
  (93–126 mm once the side gives). Under
  the 2 kHz bound every spring is capped by its nodes' mass (92 % of gen3's nodes sit at
  the bound; node mass p50 0.46 kg against BeamNG's 1.2–4 kg). Fewer, heavier
  structural nodes along the floor, sills and tunnel, or a smaller step, would allow
  it; this round changed neither.

### Checks (`gen3_rig`)

- Default: passes (3 passed, the seam check ignored as before).
- `GEN3_CABIN_R8=1`: **fails** parked fidelity, 2.85 mm / 1.41° (≤ 1 mm / 1°; at spawn
  0.92 mm / 0.11°); rest drift ≤ 2.8 mm; 0 broken, 0 bent; wall56 stretch p99 1.35,
  fold 5.1°; pole32 1.18, 1.7°. A further reason it is not the default.

### On the GPU page

See below (`media/gen3-r8/`).

### Diagnostics added

- `examples/gen3_geom.rs`: the rig's nodes (CSV) and the parts' extents along the car.
- `SIDE_DEBUG`: the far side's departure from its rigid fit (worst ten nodes) at the
  deepest crush and at the end, and each side's wheelbase change.
- `WALL_DEBUG`: the rear cabin's and the centre of mass's velocity every 10 ms to
  300 ms, and the elastic energy by member at t_m.
- The intrusion readings' worst node (`max_node` in the JSON).
- `GEN3_RAIL_SEC_Y`, `GEN3_FRAME_SEC_Y`, `GEN3_INTRUSION_KGM`, `GEN3_DOORFRAME_KGM`,
  `GEN3_FRONT_SHEET_EPS`, `GEN3_SILL_BOX_FOOTWELL` (R7's sill box now keeps the
  footwell braced and unboxed: feet 135 → 117 mm).
- `CRASH_LAB_PROGRESS` in `crash-lab-record.mjs`: the page's load every 30 s.

### Open

- The pole (above): global stiffness under the bound.
- `GEN3_CABIN_R8` as the default: the frontal's rebound (0.264) and b1 (42.4) with the
  shorter engine bay, the MDB (238 mm) and parked fidelity (1.41°).
