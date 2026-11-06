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
